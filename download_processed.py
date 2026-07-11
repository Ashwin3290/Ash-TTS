"""
Download the processed features archive from HuggingFace and extract into data/.
Counterpart to upload_data.py — run this on the cloud box (Kaggle/RunPod)
instead of re-running preprocess.py.

Usage:
    python download_processed.py
"""

import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_REPO = "Ashwin-C9/tts-ljspeech-processed"
DEST = Path("data")


def main():
    if (DEST / "processed" / "train_manifest.json").exists():
        print(f"{DEST / 'processed'} already exists, skipping download.")
        return

    print(f"Downloading processed.tar.gz from {DATASET_REPO}...")
    archive = hf_hub_download(DATASET_REPO, "processed.tar.gz", repo_type="dataset")
    print(f"Downloaded: {archive}")

    DEST.mkdir(parents=True, exist_ok=True)
    print("Extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DEST)  # archive root is "processed/"
    print(f"Done. Features at {DEST / 'processed'}")


if __name__ == "__main__":
    main()
