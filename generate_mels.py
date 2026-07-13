"""
Generate FastSpeech2-predicted mels for HiFi-GAN fine-tuning.

The vocoder fine-tuned on ground-truth mels never sees the blurrier mels
FastSpeech2 actually produces at inference — the documented HiFi-GAN
fine-tuning procedure closes that gap by training the generator on
(predicted mel -> real audio) pairs.

For each utterance in the train+val manifests:
  - forward best.pt with GROUND-TRUTH durations (keeps the predicted mel
    frame-aligned with the real audio) but PREDICTED pitch/energy (keeps
    the mel distribution close to real inference conditions)
  - save the normalised [-1,1] predicted mel to data/processed/mel_pred/

Uses cached phoneme ids from data/processed/phoneme/ — no phonemizer or
espeak needed, so this runs on the cloud box as-is.

Usage:
    python generate_mels.py
    python generate_mels.py --ckpt checkpoints/fastspeech2/best.pt
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import paths
from model.fastspeech2 import FastSpeech2, load_fs2_state


def main(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(paths.processed_dir / "stats.json") as f:
        stats = json.load(f)
    model = FastSpeech2().to(device)
    model.variance_adaptor.set_stats(**stats)
    ckpt = torch.load(ckpt_path, map_location=device)
    load_fs2_state(model, ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (step {ckpt.get('step', '?')}, "
          f"best val loss {ckpt.get('best_val_loss')})")

    manifest = []
    for name in ("train_manifest.json", "val_manifest.json"):
        with open(paths.processed_dir / name) as f:
            manifest.extend(json.load(f))
    print(f"{len(manifest)} utterances")

    out_dir = paths.processed_dir / "mel_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    with torch.no_grad():
        for item in tqdm(manifest, desc="predicting mels"):
            utt_id = item["id"]
            ph_path = paths.processed_dir / "phoneme" / f"{utt_id}.npy"
            dur_path = paths.processed_dir / "duration" / f"{utt_id}.npy"
            if not ph_path.exists() or not dur_path.exists():
                skipped += 1
                continue

            phonemes = torch.from_numpy(np.load(ph_path)).long().unsqueeze(0).to(device)
            durations = torch.from_numpy(np.load(dur_path)).long().unsqueeze(0).to(device)
            ph_lens = torch.tensor([phonemes.size(1)], dtype=torch.long, device=device)

            # GT durations for frame alignment; pitch/energy left to the model
            _, mel_pred, _, _, _, mel_lens = model(   # PostNet-refined mel_after
                phonemes, ph_lens, durations_gt=durations,
                f0_gt=None, energy_gt=None,
            )
            mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{utt_id}.npy", mel)

    print(f"Done. Saved to {out_dir}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/fastspeech2/best.pt")
    args = parser.parse_args()
    main(ckpt_path=args.ckpt)
