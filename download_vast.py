#!/usr/bin/env python3
"""Download model and compressed data from HuggingFace for vast.ai training."""

import os
import sys
import tarfile
from pathlib import Path

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        sys.exit(1)

    from huggingface_hub import hf_hub_download, list_repo_files

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

    # Download and extract dataset tar.gz files
    print(f"Downloading dataset from {HF_DATA_REPO}...")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("data/.hf_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # List all tar.gz files in the repo
        files = list_repo_files(HF_DATA_REPO, repo_type="dataset", token=token)
        tar_files = sorted([f for f in files if f.startswith("processed_part") and f.endswith(".tar.gz")])

        if not tar_files:
            print("ERROR: No processed_part*.tar.gz files found in dataset")
            sys.exit(1)

        print(f"Found {len(tar_files)} tar.gz files, downloading and extracting...")

        for tar_file in tar_files:
            print(f"  Downloading {tar_file}...")
            tar_path = hf_hub_download(
                repo_id=HF_DATA_REPO,
                filename=tar_file,
                local_dir=str(temp_dir),
                repo_type="dataset",
                token=token,
            )

            print(f"  Extracting {tar_file}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path="data")

        # Clean up temp dir
        import shutil
        shutil.rmtree(temp_dir)

        # Verify critical files exist
        if not (processed_dir / "train_manifest.json").exists():
            print("ERROR: train_manifest.json not found after extraction")
            sys.exit(1)
        if not (processed_dir / "val_manifest.json").exists():
            print("ERROR: val_manifest.json not found after extraction")
            sys.exit(1)

        # Verify data files
        phoneme_files = list((processed_dir / "phoneme").glob("*.npy"))
        duration_files = list((processed_dir / "duration").glob("*.npy"))
        if not phoneme_files or not duration_files:
            print(f"ERROR: Missing phoneme ({len(phoneme_files)} files) or duration ({len(duration_files)} files)")
            sys.exit(1)

        print(f"✓ Dataset complete: {len(phoneme_files)} phoneme files, {len(duration_files)} duration files")

    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
