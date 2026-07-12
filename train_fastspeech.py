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
from model.fastspeech2 import FastSpeech2, masked_mse, duration_loss

SAVE_EVERY = 1000   # save every 1000 steps
KEEP_LAST_N = 1      # step_*.pt checkpoints to retain besides latest.pt (each ~500MB)


def get_lr(step, d_model=mcfg.d_model, warmup=tcfg.warmup_steps):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


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
                     best_val_loss=None, is_best=False):
    ckpt_dir = Path(ckpt_dir)
    state = {
        "step":          step,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
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


def train(resume_path=None):
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

    optimizer = build_optimizer(model, lr=1.0, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    scaler    = GradScaler("cuda", enabled=tcfg.fp16)

    start_step = 0
    best_val_loss = float("inf")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        # optimizer.load_state_dict maps saved state to current param_groups
        # *positionally* (flattened parameter order), not by name — if the
        # number of param_groups differs from what's saved (e.g. switching
        # from a single-group Adam to this two-group decay/no-decay AdamW),
        # a naive load silently assigns momentum buffers to the wrong
        # parameters instead of erroring. Only restore optimizer state when
        # the shape actually matches; otherwise start it fresh.
        saved_opt = ckpt["optimizer"]
        structure_matches = len(saved_opt["param_groups"]) == len(optimizer.param_groups)
        if structure_matches:
            optimizer.load_state_dict(saved_opt)
            # override the saved weight_decay so a config change takes effect
            # on resume — group 0 = decay params, group 1 = no-decay (fixed
            # order from build_optimizer)
            optimizer.param_groups[0]["weight_decay"] = tcfg.weight_decay
            optimizer.param_groups[1]["weight_decay"] = 0.0
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(f"Optimizer structure changed ({len(saved_opt['param_groups'])} -> "
                  f"{len(optimizer.param_groups)} param groups) — starting fresh "
                  f"optimizer state (model weights still restored normally).")
            # scheduler.load_state_dict would hit the same positional mismatch
            # (LambdaLR's base_lrs is sized to the OLD param_group count) — skip
            # it and manually fast-forward the scheduler/optimizer LR instead
            resume_step = ckpt["step"]
            scheduler.last_epoch = resume_step
            for group in optimizer.param_groups:
                group["lr"] = get_lr(resume_step)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_val_loss") is not None:
            best_val_loss = ckpt["best_val_loss"]
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step} (best val loss so far: {best_val_loss:.4f})")

    running_loss = 0.0
    model.train()
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
            mel_pred, log_dur_pred, pitch_pred, energy_pred, _ = model(
                phonemes, ph_lens,
                durations_gt=durations,
                f0_gt=f0_gt,
                energy_gt=energy_gt,
            )

            T           = min(mel_pred.size(1), mel_gt.size(1))
            capped_lens = mel_lens.clamp(max=T)

            loss_mel    = masked_mse(mel_pred[:, :T], mel_gt[:, :T], capped_lens) * tcfg.mel_loss_weight
            loss_dur    = duration_loss(log_dur_pred, durations, ph_lens) * tcfg.duration_loss_weight
            loss_pitch  = F.mse_loss(pitch_pred[:, :T] * (f0_gt[:, :T] != 0).float(),
                                     f0_gt[:, :T]) * tcfg.pitch_loss_weight
            loss_energy = F.mse_loss(energy_pred[:, :T], energy_gt[:, :T]) * tcfg.energy_loss_weight
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
            writer.add_scalar("train/dur",    loss_dur.item(),    step + 1)
            writer.add_scalar("train/pitch",  loss_pitch.item(),  step + 1)
            writer.add_scalar("train/energy", loss_energy.item(), step + 1)
            writer.add_scalar("train/lr",     optimizer.param_groups[0]["lr"], step + 1)
            running_loss = 0.0

        if (step + 1) % tcfg.val_every == 0:
            model.eval()
            val_losses = []
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
                        vmel_pred, _, _, _, _ = model(
                            vph, vphl, durations_gt=vdur, f0_gt=vf0, energy_gt=veng)
                        T     = min(vmel_pred.size(1), vmel.size(1))
                        vloss = masked_mse(vmel_pred[:, :T], vmel[:, :T], vmell.clamp(max=T))
                    val_losses.append(vloss.item())
            val_mel_loss = sum(val_losses) / len(val_losses)
            print(f"\n[step {step+1}] val mel loss: {val_mel_loss:.4f}")
            writer.add_scalar("val/mel_loss", val_mel_loss, step + 1)
            if val_mel_loss < best_val_loss:
                best_val_loss = val_mel_loss
                is_best = True
                print(f"New best val loss: {best_val_loss:.4f}")
            model.train()

        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = save_checkpoint(
                step + 1, model, optimizer, scheduler, scaler,
                paths.fastspeech_ckpt_dir,
                best_val_loss=best_val_loss, is_best=is_best,
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
    args = parser.parse_args()
    resume_path = find_resume_path(args.resume, paths.fastspeech_ckpt_dir)
    train(resume_path=resume_path)