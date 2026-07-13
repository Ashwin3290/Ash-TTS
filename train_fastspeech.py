"""
FastSpeech2 training loop.

Trains the acoustic model (text -> mel spectrogram).
Run this before training HiFi-GAN.

Usage:
    python train_fastspeech.py                         # fresh start
    python train_fastspeech.py --resume                # auto-resumes from latest.pt
    python train_fastspeech.py --resume checkpoints/fastspeech2/step_50000.pt
"""

import argparse
import json
import os
import re
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from config import train as tcfg, paths, audio as acfg, model as mcfg
from data.dataset import get_loaders
from model.fastspeech2 import (FastSpeech2, masked_mse, masked_l1,
                               duration_loss, pitch_loss, energy_loss)

SAVE_EVERY = 1000   # save every 1000 steps
KEEP_LAST_N = 1      # step_*.pt checkpoints to retain besides latest.pt (each ~500MB)


def get_lr(step, d_model=mcfg.d_model, warmup=tcfg.warmup_steps):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def get_finetune_lr(step, warmup=tcfg.finetune_warmup):
    """Flat LR with a short linear re-warmup, used after unfreezing the backbone.
    Multiplies the optimizer's base lr (tcfg.finetune_lr)."""
    return min(1.0, max(step, 1) / warmup)


# During the frozen warm-start phase only these submodules train: the fresh
# PostNet, plus the pitch/energy predictors (tiny, and they need to relearn
# under the fixed pitch/energy losses anyway).
FROZEN_PHASE_TRAINABLE = ("postnet.",
                          "variance_adaptor.pitch_predictor.",
                          "variance_adaptor.energy_predictor.")


def set_frozen_phase(model, frozen: bool):
    for name, p in model.named_parameters():
        p.requires_grad = (not frozen) or name.startswith(FROZEN_PHASE_TRAINABLE)


def set_train_mode(model, phase):
    """model.train(), but during the frozen phase the frozen backbone stays in
    eval() so its dropout is off — the PostNet must see the same mel_before it
    will see at inference, not a noisier version of it."""
    model.train()
    if phase == "frozen":
        model.encoder.eval()
        model.decoder.eval()
        model.variance_adaptor.duration_predictor.eval()


def validate(model, val_loader, device):
    """Returns (val_before_l1, val_after_l1): per-element masked L1 on the
    decoder output and the PostNet output, teacher-forced like training."""
    model.eval()
    before, after = [], []
    with torch.no_grad():
        for vbatch in val_loader:
            vph   = vbatch["phonemes"].to(device)
            vmel  = vbatch["mel"].to(device)
            vf0   = vbatch["f0"].to(device)
            veng  = vbatch["energy"].to(device)
            vdur  = vbatch["durations"].to(device)
            vphl  = vbatch["ph_lens"].to(device)
            vmell = vbatch["mel_lens"].to(device)
            with autocast("cuda", enabled=tcfg.fp16):
                mel_before, mel_after, _, _, _, _ = model(
                    vph, vphl, durations_gt=vdur, f0_gt=vf0, energy_gt=veng)
                T = min(mel_before.size(1), vmel.size(1))
                lens = vmell.clamp(max=T)
                before.append(masked_l1(mel_before[:, :T], vmel[:, :T], lens).item() / acfg.n_mels)
                after.append(masked_l1(mel_after[:, :T],  vmel[:, :T], lens).item() / acfg.n_mels)
    return sum(before) / len(before), sum(after) / len(after)


def build_optimizer(model, lr, weight_decay):
    """
    AdamW with decoupled weight decay, excluding biases and norm parameters —
    decaying those doesn't regularise anything and is known to hurt training.
    Two param groups: only the first carries weight_decay.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.98), eps=1e-9,
    )


def save_checkpoint(step, model, optimizer, scheduler, scaler, ckpt_dir,
                     best_val_loss=None, is_best=False,
                     phase="full", lr_mode="noam", baseline=None):
    ckpt_dir = Path(ckpt_dir)
    state = {
        "step":          step,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
        # PostNet warm-start state — lets --resume restore the correct
        # freeze phase, LR regime, and unfreeze-gate baseline
        "phase":         phase,
        "lr_mode":       lr_mode,
        "baseline":      baseline,
    }
    ckpt_path = ckpt_dir / f"step_{step}.pt"
    torch.save(state, ckpt_path)
    # write latest.pt atomically — a crash mid-save must not corrupt the resume point
    tmp = ckpt_dir / "latest.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(ckpt_dir / "latest.pt")

    # best.pt tracks the lowest validation loss seen so far, independent of the
    # step_*.pt pruning below — training loss can keep dropping from overfitting
    # long after val loss bottoms out, so "most recent" != "best generalizing"
    if is_best:
        tmp_best = ckpt_dir / "best.pt.tmp"
        torch.save(state, tmp_best)
        tmp_best.replace(ckpt_dir / "best.pt")

    # prune old step_*.pt checkpoints — unbounded accumulation fills the disk
    # and corrupts the next torch.save mid-write (seen in practice on Vast.ai)
    numbered = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = re.fullmatch(r"step_(\d+)\.pt", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda t: t[0], reverse=True)
    for _, old_path in numbered[KEEP_LAST_N:]:
        old_path.unlink(missing_ok=True)

    # optional off-machine backup: set HF_CKPT_REPO to push best.pt to
    # HuggingFace whenever it improves. latest.pt stays local-only — it's
    # only needed to resume the current session, and best.pt is the one
    # actually worth keeping off-machine.
    repo = os.environ.get("HF_CKPT_REPO")
    if repo and is_best:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # delete before re-uploading — an in-place overwrite still leaves
            # the old chunk data retained by HF's storage backend even after
            # squashing history, so an explicit delete is needed to actually
            # release it (squash_history() alone was seen to leave ~7GB behind
            # despite only ~1GB of files being visible)
            try:
                api.delete_file("best.pt", repo_id=repo, repo_type="model")
            except Exception:
                pass  # first push ever — nothing to delete yet
            api.upload_file(
                path_or_fileobj=str(ckpt_dir / "best.pt"),
                path_in_repo="best.pt",
                repo_id=repo, repo_type="model",
            )
            api.super_squash_history(repo, repo_type="model")
        except Exception as e:
            print(f"\nHF checkpoint upload failed (continuing): {e}")

    return ckpt_path


def find_resume_path(resume_arg, ckpt_dir):
    """
    None        -> fresh start
    'auto'/'auto' -> load latest.pt if it exists
    path string -> load that specific checkpoint
    """
    if resume_arg is None:
        return None
    if resume_arg in ("", "auto"):
        latest = Path(ckpt_dir) / "latest.pt"
        if latest.exists():
            print(f"Auto-resuming from {latest}")
            return str(latest)
        print("No latest.pt found, starting fresh.")
        return None
    return resume_arg


def train(resume_path=None, init_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    paths.make_dirs()
    writer = SummaryWriter(log_dir=str(paths.log_dir / "fastspeech2"))

    train_loader, val_loader = get_loaders()
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    with open(paths.processed_dir / "stats.json") as f:
        stats = json.load(f)

    model = FastSpeech2().to(device)
    model.variance_adaptor.set_stats(**stats)
    print(f"Model parameters: {model.count_parameters():,}")

    # ---- phase / LR-regime bookkeeping (PostNet warm-start) -----------------
    # phase:   "frozen" = backbone frozen, only PostNet + pitch/energy predictors
    #          train; "full" = everything trains.
    # lr_mode: "noam" = transformer schedule (fresh runs and the frozen phase);
    #          "finetune" = flat low LR + short re-warmup (after unfreezing).
    # baseline: frozen backbone's own val mel_before L1 — the unfreeze gate
    #          compares val mel_after L1 against it.
    phase, lr_mode, baseline = "full", "noam", None
    start_step = 0
    best_val_loss = float("inf")
    ckpt = None

    if init_path:
        # warm start: model weights only (no optimizer/step state) from a
        # checkpoint that may predate the PostNet
        init_ckpt = torch.load(init_path, map_location=device)
        state = init_ckpt.get("model", init_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("postnet.")]
        if bad_missing or unexpected:
            raise ValueError(f"--init checkpoint mismatch beyond the PostNet: "
                             f"missing={bad_missing} unexpected={unexpected}")
        if missing:
            phase = "frozen"
            print(f"Initialized from {init_path} — PostNet is fresh "
                  f"({len(missing)} new keys), entering frozen warm-start phase.")
        else:
            print(f"Initialized from {init_path} (PostNet weights present — "
                  f"training everything from step 0).")
    elif resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        phase    = ckpt.get("phase", "full")
        lr_mode  = ckpt.get("lr_mode", "noam")
        baseline = ckpt.get("baseline")

    set_frozen_phase(model, frozen=(phase == "frozen"))

    def build_opt_sched():
        if lr_mode == "finetune":
            opt = build_optimizer(model, lr=tcfg.finetune_lr, weight_decay=tcfg.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, get_finetune_lr)
        else:
            opt = build_optimizer(model, lr=1.0, weight_decay=tcfg.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
        return opt, sched

    optimizer, scheduler = build_opt_sched()
    scaler = GradScaler("cuda", enabled=tcfg.fp16)

    if ckpt is not None:
        # optimizer.load_state_dict maps saved state to current param_groups
        # *positionally* (flattened parameter order), not by name — if the
        # number of param_groups differs from what's saved (e.g. switching
        # from a single-group Adam to this two-group decay/no-decay AdamW),
        # a naive load silently assigns momentum buffers to the wrong
        # parameters instead of erroring. Only restore optimizer state when
        # the shape actually matches; otherwise start it fresh.
        saved_opt = ckpt["optimizer"]
        n_saved_params = sum(len(g["params"]) for g in saved_opt["param_groups"])
        n_curr_params  = sum(len(g["params"]) for g in optimizer.param_groups)
        structure_matches = (len(saved_opt["param_groups"]) == len(optimizer.param_groups)
                             and n_saved_params == n_curr_params)
        if structure_matches:
            optimizer.load_state_dict(saved_opt)
            # override the saved weight_decay so a config change takes effect
            # on resume — group 0 = decay params, group 1 = no-decay (fixed
            # order from build_optimizer)
            optimizer.param_groups[0]["weight_decay"] = tcfg.weight_decay
            optimizer.param_groups[1]["weight_decay"] = 0.0
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(f"Optimizer structure changed — starting fresh optimizer state "
                  f"(model weights still restored normally).")
            # scheduler.load_state_dict would hit the same positional mismatch
            # (LambdaLR's base_lrs is sized to the OLD param_group count) — skip
            # it and manually fast-forward the scheduler/optimizer LR instead
            resume_step = ckpt["step"]
            scheduler.last_epoch = resume_step
            lr_fn = get_finetune_lr if lr_mode == "finetune" else get_lr
            base = tcfg.finetune_lr if lr_mode == "finetune" else 1.0
            for group in optimizer.param_groups:
                group["lr"] = base * lr_fn(resume_step)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_val_loss") is not None:
            best_val_loss = ckpt["best_val_loss"]
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step} (phase={phase}, lr_mode={lr_mode}, "
              f"best val loss so far: {best_val_loss:.4f})")

    # unfreeze-gate baseline: the frozen backbone's own val mel_before L1.
    # Measured once, before any training — this is "the current best
    # checkpoint's" quality that the PostNet has to beat.
    if phase == "frozen" and baseline is None:
        print("Measuring frozen-backbone baseline (val mel_before L1)...")
        baseline, base_after = validate(model, val_loader, device)
        print(f"Baseline val mel_before L1/element: {baseline:.5f} "
              f"(mel_after at init: {base_after:.5f}) — unfreeze when mel_after "
              f"<= {baseline * tcfg.unfreeze_threshold:.5f}")

    running_loss = 0.0
    set_train_mode(model, phase)
    train_iter = iter(train_loader)

    pbar = tqdm(range(start_step, tcfg.max_steps), initial=start_step, total=tcfg.max_steps)
    for step in pbar:
        is_best = False

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        phonemes  = batch["phonemes"].to(device)
        mel_gt    = batch["mel"].to(device)
        f0_gt     = batch["f0"].to(device)
        energy_gt = batch["energy"].to(device)
        durations = batch["durations"].to(device)
        ph_lens   = batch["ph_lens"].to(device)
        mel_lens  = batch["mel_lens"].to(device)

        optimizer.zero_grad()

        with autocast("cuda",enabled=tcfg.fp16):
            mel_before, mel_after, log_dur_pred, pitch_pred, energy_pred, _ = model(
                phonemes, ph_lens,
                durations_gt=durations,
                f0_gt=f0_gt,
                energy_gt=energy_gt,
            )

            T           = min(mel_before.size(1), mel_gt.size(1))
            capped_lens = mel_lens.clamp(max=T)

            # MSE on the raw decoder output (same objective the backbone
            # converged under), L1 on the PostNet output (less prone to
            # conditional-mean blur — this is the sharpening signal)
            loss_mel_before = masked_mse(mel_before[:, :T], mel_gt[:, :T], capped_lens)
            loss_mel_after  = masked_l1(mel_after[:, :T],  mel_gt[:, :T], capped_lens)
            loss_mel    = (loss_mel_before + loss_mel_after) * tcfg.mel_loss_weight
            loss_dur    = duration_loss(log_dur_pred, durations, ph_lens) * tcfg.duration_loss_weight
            loss_pitch  = pitch_loss(pitch_pred[:, :T], f0_gt[:, :T], capped_lens) * tcfg.pitch_loss_weight
            loss_energy = energy_loss(energy_pred[:, :T], energy_gt[:, :T], capped_lens) * tcfg.energy_loss_weight
            loss        = loss_mel + loss_dur + loss_pitch + loss_energy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        if not (loss_val != loss_val):  # skip nan
            running_loss += loss_val

        if (step + 1) % tcfg.log_every == 0:
            avg_loss = running_loss / tcfg.log_every
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "mel":  f"{loss_mel.item():.3f}",
                "dur":  f"{loss_dur.item():.3f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })
            writer.add_scalar("train/loss",   avg_loss,           step + 1)
            writer.add_scalar("train/mel",    loss_mel.item(),    step + 1)
            writer.add_scalar("train/mel_before_mse", loss_mel_before.item(), step + 1)
            writer.add_scalar("train/mel_after_l1",   loss_mel_after.item(),  step + 1)
            # per-element view — masked losses sum over the 80 mel channels
            writer.add_scalar("train/mel_after_l1_per_elem",
                              loss_mel_after.item() / acfg.n_mels, step + 1)
            writer.add_scalar("train/dur",    loss_dur.item(),    step + 1)
            writer.add_scalar("train/pitch",  loss_pitch.item(),  step + 1)
            writer.add_scalar("train/energy", loss_energy.item(), step + 1)
            writer.add_scalar("train/lr",     optimizer.param_groups[0]["lr"], step + 1)
            running_loss = 0.0

        if (step + 1) % tcfg.val_every == 0:
            val_before, val_after = validate(model, val_loader, device)
            print(f"\n[step {step+1}] val mel L1/elem — before: {val_before:.5f}  "
                  f"after: {val_after:.5f}")
            writer.add_scalar("val/mel_before_l1", val_before, step + 1)
            writer.add_scalar("val/mel_after_l1",  val_after,  step + 1)

            # best.pt tracks the PostNet output — that's what inference uses
            if val_after < best_val_loss:
                best_val_loss = val_after
                is_best = True
                print(f"New best val loss: {best_val_loss:.5f}")

            # metric-gated unfreeze: leave the frozen phase once the PostNet
            # output beats the frozen backbone's own baseline (or the safety
            # cap is hit), then fine-tune everything at a low LR
            if phase == "frozen":
                gate_hit = baseline is not None and val_after <= baseline * tcfg.unfreeze_threshold
                cap_hit  = (step + 1) >= tcfg.max_freeze_steps
                if gate_hit or cap_hit:
                    if cap_hit and not gate_hit:
                        print(f"WARNING: unfreeze gate never triggered — PostNet "
                              f"stalled at {val_after:.5f} vs target "
                              f"{baseline * tcfg.unfreeze_threshold:.5f}. Forcing "
                              f"unfreeze at the {tcfg.max_freeze_steps}-step cap.")
                    else:
                        print(f"Unfreeze gate hit at step {step+1}: mel_after "
                              f"{val_after:.5f} <= baseline {baseline:.5f} * "
                              f"{tcfg.unfreeze_threshold}. Unfreezing backbone.")
                    phase, lr_mode = "full", "finetune"
                    set_frozen_phase(model, frozen=False)
                    optimizer = build_optimizer(model, lr=tcfg.finetune_lr,
                                                weight_decay=tcfg.weight_decay)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_finetune_lr)

            set_train_mode(model, phase)

        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = save_checkpoint(
                step + 1, model, optimizer, scheduler, scaler,
                paths.fastspeech_ckpt_dir,
                best_val_loss=best_val_loss, is_best=is_best,
                phase=phase, lr_mode=lr_mode, baseline=baseline,
            )
            suffix = "  (new best.pt saved)" if is_best else ""
            print(f"\nSaved: {ckpt_path.name}  (latest.pt updated){suffix}")

    writer.close()
    print("FastSpeech2 training complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume training. No value = auto-find latest.pt. "
             "Or pass a specific checkpoint path."
    )
    parser.add_argument(
        "--init", default=None,
        help="Warm-start from a checkpoint (model weights only, fresh step "
             "counter/optimizer). If the checkpoint predates the PostNet, the "
             "backbone is frozen until the PostNet beats the checkpoint's own "
             "val mel L1 (metric-gated unfreeze), then everything fine-tunes "
             "at a low LR. Mutually exclusive with --resume."
    )
    args = parser.parse_args()
    if args.init and args.resume:
        parser.error("--init and --resume are mutually exclusive: --init starts "
                     "a new run from a checkpoint's weights; --resume continues "
                     "an existing run (and restores its phase/baseline itself).")
    resume_path = find_resume_path(args.resume, paths.fastspeech_ckpt_dir)
    train(resume_path=resume_path, init_path=args.init)