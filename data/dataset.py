"""
PyTorch Dataset and DataLoader for the preprocessed LJSpeech data.

Each sample contains:
  - phonemes:  (L,)        int32 — phoneme token ids
  - mel:       (T, 80)     float32 — normalised log mel spectrogram
  - f0:        (T,)        float32 — pitch in Hz (0 = unvoiced)
  - energy:    (T,)        float32 — RMS energy
  - durations: (L,)        int32 — phoneme durations in mel frames (if MFA available)
  - mel_len:   int — actual mel length (pre-padding)
  - ph_len:    int — actual phoneme sequence length (pre-padding)

Batches are padded to the longest sequence in the batch.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import audio as acfg, model as mcfg, train as tcfg, paths


class LJSpeechDataset(Dataset):
    def __init__(self, manifest_path, processed_dir, stats=None):
        self.processed_dir = Path(processed_dir)
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # load normalisation stats for pitch and energy
        stats_path = self.processed_dir / "stats.json"
        if stats is not None:
            self.stats = stats
        elif stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)
        else:
            # if stats don't exist yet, use identity (no normalisation)
            self.stats = {"f0_mean": 0, "f0_std": 1, "energy_mean": 0, "energy_std": 1}

    def __len__(self):
        return len(self.manifest)

    def _normalise_f0(self, f0):
        """Normalise voiced frames to zero mean unit variance. Unvoiced frames stay 0."""
        voiced = f0 > 0
        if voiced.any():
            f0[voiced] = (f0[voiced] - self.stats["f0_mean"]) / (self.stats["f0_std"] + 1e-8)
        return f0

    def _normalise_energy(self, energy):
        return (energy - self.stats["energy_mean"]) / (self.stats["energy_std"] + 1e-8)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        utt_id = item["id"]

        phonemes = np.load(self.processed_dir / "phoneme" / f"{utt_id}.npy")
        mel      = np.load(self.processed_dir / "mel"     / f"{utt_id}.npy")
        f0       = np.load(self.processed_dir / "f0"      / f"{utt_id}.npy")
        energy   = np.load(self.processed_dir / "energy"  / f"{utt_id}.npy")

        f0     = self._normalise_f0(f0)
        energy = self._normalise_energy(energy)

        dur_path = self.processed_dir / "duration" / f"{utt_id}.npy"
        durations = np.load(dur_path) if dur_path.exists() else np.zeros(len(phonemes), dtype=np.int32)

        return {
            "id": utt_id,
            "phonemes":  torch.from_numpy(phonemes).long(),
            "mel":       torch.from_numpy(mel).float(),
            "f0":        torch.from_numpy(f0).float(),
            "energy":    torch.from_numpy(energy).float(),
            "durations": torch.from_numpy(durations).long(),
            "mel_len":   mel.shape[0],
            "ph_len":    len(phonemes),
        }


def collate_fn(batch):
    """
    Pads all sequences to the longest in the batch.
    mel:       (B, T_max, 80)
    phonemes:  (B, L_max)
    f0:        (B, T_max)
    energy:    (B, T_max)
    durations: (B, L_max)
    """
    ph_lens  = [b["ph_len"]  for b in batch]
    mel_lens = [b["mel_len"] for b in batch]
    max_ph  = max(ph_lens)
    max_mel = max(mel_lens)

    B = len(batch)
    phonemes  = torch.zeros(B, max_ph,  dtype=torch.long)
    mel       = torch.zeros(B, max_mel, acfg.n_mels)
    f0        = torch.zeros(B, max_mel)
    energy    = torch.zeros(B, max_mel)
    durations = torch.zeros(B, max_ph,  dtype=torch.long)
    ph_lens_t  = torch.tensor(ph_lens,  dtype=torch.long)
    mel_lens_t = torch.tensor(mel_lens, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["ph_len"]
        T = b["mel_len"]
        phonemes[i, :L]  = b["phonemes"]
        mel[i, :T]       = b["mel"]
        f0[i, :T]        = b["f0"]
        energy[i, :T]    = b["energy"]
        durations[i, :L] = b["durations"]

    return {
        "phonemes":  phonemes,
        "mel":       mel,
        "f0":        f0,
        "energy":    energy,
        "durations": durations,
        "ph_lens":   ph_lens_t,
        "mel_lens":  mel_lens_t,
    }


def get_loaders(processed_dir=None, batch_size=None):
    processed_dir = processed_dir or paths.processed_dir
    batch_size    = batch_size    or tcfg.batch_size

    train_ds = LJSpeechDataset(processed_dir / "train_manifest.json", processed_dir)
    val_ds   = LJSpeechDataset(processed_dir / "val_manifest.json",   processed_dir,
                                stats=train_ds.stats)

    num_workers = 6
    val_workers = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=val_workers,
        pin_memory=True,
    )
    return train_loader, val_loader