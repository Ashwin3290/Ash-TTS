"""
Download the processed features (multiple tar.gz parts) from HuggingFace and
extract into data/. Counterpart to upload_data.py — run this on the cloud box
(Kaggle/RunPod) instead of re-running preprocess.py.

Usage:
    python download_processed.py
"""

import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "Ashwin-C9/tts-ljspeech-processed"
PART_PREFIX  = "processed_part"
DEST = Path("data")


def main():
    if (DEST / "processed" / "train_manifest.json").exists():
        print(f"{DEST / 'processed'} already exists, skipping download.")
        return

    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    parts = sorted(f for f in files if f.startswith(PART_PREFIX) and f.endswith(".tar.gz"))
    if not parts:
        raise RuntimeError(f"No {PART_PREFIX}*.tar.gz files found in {DATASET_REPO}")
    print(f"Found {len(parts)} archive parts.")

    DEST.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        print(f"[{i+1}/{len(parts)}] Downloading {part}...")
        archive = hf_hub_download(DATASET_REPO, part, repo_type="dataset")
        print(f"  Extracting...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(DEST)  # archive root is "processed/"

    print(f"Done. Features at {DEST / 'processed'}")


if __name__ == "__main__":
    main()
