"""
HiFi-GAN training loop.

Trains the vocoder (mel spectrogram → waveform).
Run after FastSpeech2 is trained, or train in parallel on ground-truth mels.

The generator and discriminators have separate optimisers and update alternately:
  1. Update discriminators on real vs generated audio
  2. Update generator using adversarial + feature matching + mel loss

Usage:
    python train_hifigan.py
    python train_hifigan.py --resume-g checkpoints/hifigan/g_step_50000.pt \
                             --resume-d checkpoints/hifigan/d_step_50000.pt
"""

import os
import argparse
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from config import hifigan as hcfg, audio as acfg, paths
from vocoder.generator      import Generator
from vocoder.discriminator  import MPD, MSD
from vocoder.losses         import (discriminator_loss, generator_adversarial_loss,
                                    feature_matching_loss, mel_reconstruction_loss)


class WavMelDataset(Dataset):
    """
    Loads raw wavs + ground truth mels for HiFi-GAN training.
    HiFi-GAN trains on random fixed-length chunks, not full utterances.
    """
    def __init__(self, manifest_path, data_root, processed_dir):
        import json
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.wav_dir = Path(data_root) / "wavs"
        self.mel_dir = Path(processed_dir) / "mel"
        self.segment = hcfg.segment_length

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        import soundfile as sf
        utt_id = self.manifest[idx]["id"]

        wav, sr = sf.read(str(self.wav_dir / f"{utt_id}.wav"), dtype="float32")
        if sr != acfg.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=acfg.sample_rate)

        mel = np.load(self.mel_dir / f"{utt_id}.npy")  # (T, 80)

        # random crop to segment_length samples
        if len(wav) >= self.segment:
            start = np.random.randint(0, len(wav) - self.segment)
            wav = wav[start:start + self.segment]
            # align mel: start frame = start_sample // hop_length
            mel_start = start // acfg.hop_length
            mel_frames = self.segment // acfg.hop_length
            mel = mel[mel_start:mel_start + mel_frames]
        else:
            # pad short clips
            wav = np.pad(wav, (0, self.segment - len(wav)))
            mel_frames = self.segment // acfg.hop_length
            mel = mel[:mel_frames]
            if len(mel) < mel_frames:
                mel = np.pad(mel, ((0, mel_frames - len(mel)), (0, 0)))

        return {
            "wav": torch.from_numpy(wav).unsqueeze(0),   # (1, segment)
            "mel": torch.from_numpy(mel.T).float(),       # (80, T) channel-first for conv
        }


def get_mel_fn(device):
    """Returns a function: wav (B, T) → mel (B, 80, T_mel) for loss computation."""
    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=acfg.sample_rate,
        n_fft=acfg.n_fft,
        hop_length=acfg.hop_length,
        win_length=acfg.win_length,
        n_mels=acfg.n_mels,
        f_min=acfg.fmin,
        f_max=acfg.fmax,
    ).to(device)

    def mel_fn(wav):
        mel = mel_transform(wav)
        return torch.log(mel.clamp(min=1e-5))

    return mel_fn


def train(resume_g=None, resume_d=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths.make_dirs()

    train_ds = WavMelDataset(
        paths.processed_dir / "train_manifest.json",
        paths.data_root, paths.processed_dir,
    )
    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_ds,
        batch_size=hcfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train batches: {len(train_loader)}")

    generator = Generator().to(device)
    mpd = MPD().to(device)
    msd = MSD().to(device)

    g_params  = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params  = sum(p.numel() for p in list(mpd.parameters()) + list(msd.parameters()) if p.requires_grad)
    print(f"Generator params: {g_params:,}  Discriminator params: {d_params:,}")

    opt_g = torch.optim.AdamW(generator.parameters(), hcfg.learning_rate,
                               betas=(hcfg.adam_b1, hcfg.adam_b2))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()),
                               hcfg.learning_rate, betas=(hcfg.adam_b1, hcfg.adam_b2))

    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, hcfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, hcfg.lr_decay)

    scaler_g = GradScaler(enabled=hcfg.fp16)
    scaler_d = GradScaler(enabled=hcfg.fp16)

    start_step = 0
    if resume_g:
        ckpt = torch.load(resume_g, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        sched_g.load_state_dict(ckpt["sched_g"])
        if "scaler_g" in ckpt:
            scaler_g.load_state_dict(ckpt["scaler_g"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed generator from step {start_step}")
    if resume_d:
        ckpt = torch.load(resume_d, map_location=device)
        mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])
        opt_d.load_state_dict(ckpt["opt_d"])
        sched_d.load_state_dict(ckpt["sched_d"])
        if "scaler_d" in ckpt:
            scaler_d.load_state_dict(ckpt["scaler_d"])

    mel_fn = get_mel_fn(device)
    step = start_step
    train_iter = iter(train_loader)

    pbar = tqdm(range(start_step, hcfg.max_steps), initial=start_step, total=hcfg.max_steps)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            sched_g.step()
            sched_d.step()

        wav_real = batch["wav"].to(device)   # (B, 1, segment)
        mel      = batch["mel"].to(device)   # (B, 80, T)

        # generate fake waveform
        with autocast(enabled=hcfg.fp16):
            wav_fake = generator(mel)

        # --- Discriminator update ---
        opt_d.zero_grad()
        with autocast(enabled=hcfg.fp16):
            mpd_real, mpd_fake = mpd(wav_real, wav_fake.detach())
            msd_real, msd_fake = msd(wav_real, wav_fake.detach())
            d_loss = discriminator_loss(mpd_real, mpd_fake) + discriminator_loss(msd_real, msd_fake)

        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), 1000.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Generator update ---
        opt_g.zero_grad()
        with autocast(enabled=hcfg.fp16):
            mpd_real, mpd_fake = mpd(wav_real, wav_fake)
            msd_real, msd_fake = msd(wav_real, wav_fake)

            loss_adv = generator_adversarial_loss(mpd_fake) + generator_adversarial_loss(msd_fake)
            loss_fm  = feature_matching_loss(mpd_real, mpd_fake) + feature_matching_loss(msd_real, msd_fake)
            loss_mel = mel_reconstruction_loss(wav_fake, wav_real, mel_fn)

            # weights from original HiFi-GAN paper: fm=2, mel=45
            g_loss = loss_adv + 2 * loss_fm + 45 * loss_mel

        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1000.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        if (step + 1) % hcfg.log_every == 0:
            pbar.set_postfix({
                "g":   f"{g_loss.item():.3f}",
                "d":   f"{d_loss.item():.3f}",
                "mel": f"{loss_mel.item():.4f}",
                "fm":  f"{loss_fm.item():.3f}",
            })

        if (step + 1) % hcfg.save_every == 0:
            g_path = paths.hifigan_ckpt_dir / f"g_step_{step+1}.pt"
            d_path = paths.hifigan_ckpt_dir / f"d_step_{step+1}.pt"
            g_state = {
                "step": step + 1,
                "generator": generator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "sched_g": sched_g.state_dict(),
                "scaler_g": scaler_g.state_dict(),
            }
            d_state = {
                "step": step + 1,
                "mpd": mpd.state_dict(),
                "msd": msd.state_dict(),
                "opt_d": opt_d.state_dict(),
                "sched_d": sched_d.state_dict(),
                "scaler_d": scaler_d.state_dict(),
            }
            torch.save(g_state, g_path)
            torch.save(d_state, d_path)
            # atomic latest checkpoints for --resume auto
            for state, name in ((g_state, "g_latest.pt"), (d_state, "d_latest.pt")):
                tmp = paths.hifigan_ckpt_dir / f"{name}.tmp"
                torch.save(state, tmp)
                tmp.replace(paths.hifigan_ckpt_dir / name)
            print(f"\nSaved: {g_path.name}, {d_path.name}  (g_latest/d_latest updated)")

    print("HiFi-GAN training complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from g_latest.pt / d_latest.pt")
    parser.add_argument("--resume-g", default=None)
    parser.add_argument("--resume-d", default=None)
    args = parser.parse_args()

    if args.resume == "auto" and not (args.resume_g or args.resume_d):
        g_latest = paths.hifigan_ckpt_dir / "g_latest.pt"
        d_latest = paths.hifigan_ckpt_dir / "d_latest.pt"
        if g_latest.exists():
            args.resume_g = str(g_latest)
            print(f"Auto-resuming generator from {g_latest}")
        if d_latest.exists():
            args.resume_d = str(d_latest)
            print(f"Auto-resuming discriminators from {d_latest}")
        if not g_latest.exists():
            print("No g_latest.pt found, starting fresh.")

    train(resume_g=args.resume_g, resume_d=args.resume_d)
