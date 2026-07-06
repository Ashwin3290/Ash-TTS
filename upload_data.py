"""
Upload processed data as a single tar.gz to HuggingFace.
Much faster than uploading 12K individual files.

Usage:
    python upload_data.py
"""

import tarfile
from pathlib import Path
from huggingface_hub import HfApi

PROCESSED_DIR = Path("data/processed")
ARCHIVE_PATH  = Path("data/processed.tar.gz")
DATASET_REPO  = "Ashwin-C9/tts-ljspeech-processed"
MODEL_REPO    = "Ashwin-C9/tts-fastspeech2-ckpt"


def create_archive():
    print(f"Creating archive from {PROCESSED_DIR}...")
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        tar.add(PROCESSED_DIR, arcname="processed")
    size_gb = ARCHIVE_PATH.stat().st_size / 1e9
    print(f"Archive created: {ARCHIVE_PATH} ({size_gb:.2f} GB)")


def reset_and_upload_dataset():
    api = HfApi()

    print(f"Deleting repo {DATASET_REPO}...")
    try:
        api.delete_repo(DATASET_REPO, repo_type="dataset")
        print("  deleted.")
    except Exception:
        print("  repo did not exist, skipping delete.")

    print(f"Creating repo {DATASET_REPO}...")
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True)
    print("  created.")

    print(f"Uploading {ARCHIVE_PATH}...")
    api.upload_file(
        path_or_fileobj=str(ARCHIVE_PATH),
        path_in_repo="processed.tar.gz",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Done. Dataset at: https://huggingface.co/datasets/{DATASET_REPO}")


def ensure_model_repo():
    api = HfApi()
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo exists: {MODEL_REPO}")
    except Exception:
        api.create_repo(MODEL_REPO, repo_type="model", private=True)
        print(f"Model repo created: {MODEL_REPO}")


if __name__ == "__main__":
    create_archive()
    reset_and_upload_dataset()
    ensure_model_repo()
    print("\nAll done. You can now delete data/processed.tar.gz if you want to save disk space.")