#!/usr/bin/env python3
"""Download model and data from HuggingFace for vast.ai training."""

import os
import sys
from pathlib import Path

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        sys.exit(1)

    from huggingface_hub import hf_hub_download, snapshot_download

    HF_CKPT_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"
    HF_DATA_REPO = "Ashwin-C9/tts-data-silence-fix"

    # Download model checkpoint
    print(f"Downloading model from {HF_CKPT_REPO}...")
    ckpt_dir = Path("checkpoints/fastspeech2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=HF_CKPT_REPO,
            filename="best.pt",
            local_dir=str(ckpt_dir),
            token=token,
        )
        # Rename best.pt to latest.pt for auto-resume
        best_path = ckpt_dir / "best.pt"
        latest_path = ckpt_dir / "latest.pt"
        if best_path.exists():
            best_path.replace(latest_path)
        print("✓ Model checkpoint ready for resume")
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        sys.exit(1)

    # Download dataset
    print(f"Downloading dataset from {HF_DATA_REPO}...")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the entire repo (manifests + phoneme + duration)
        snapshot_download(
            repo_id=HF_DATA_REPO,
            local_dir=str(processed_dir),
            repo_type="dataset",
            token=token,
        )
        print("✓ Dataset downloaded successfully")
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        sys.exit(1)

    # Verify critical files exist
    required_files = [
        processed_dir / "train_manifest.json",
        processed_dir / "val_manifest.json",
    ]
    for fpath in required_files:
        if not fpath.exists():
            print(f"ERROR: {fpath} not found after download")
            sys.exit(1)

    # Verify at least some phoneme and duration files exist
    phoneme_files = list((processed_dir / "phoneme").glob("*.npy"))
    duration_files = list((processed_dir / "duration").glob("*.npy"))
    if not phoneme_files or not duration_files:
        print(f"ERROR: Missing phoneme ({len(phoneme_files)} files) or duration ({len(duration_files)} files)")
        sys.exit(1)

    print(f"✓ Dataset complete: {len(phoneme_files)} phoneme files, {len(duration_files)} duration files")

if __name__ == "__main__":
    main()
