"""
Push regenerated data and model to HuggingFace.

Usage:
    export HF_TOKEN="hf_xxxxx"
    python push_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

DATA_REPO = "Ashwin-C9/tts-data-silence-fix"
MODEL_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"

def push_data():
    """Push regenerated phoneme/duration files."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        return False

    api = HfApi(token=token)
    processed = Path("data/processed")

    print(f"Pushing to {DATA_REPO}...")

    # Push phoneme files
    ph_files = list((processed / "phoneme").glob("*.npy"))
    print(f"Uploading {len(ph_files)} phoneme files...")
    for i, ph_file in enumerate(sorted(ph_files)):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(ph_files)}")
        try:
            api.upload_file(
                path_or_fileobj=str(ph_file),
                path_in_repo=f"phoneme/{ph_file.name}",
                repo_id=DATA_REPO,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  ⚠️ {ph_file.name}: {e}")

    # Push duration files
    dur_files = list((processed / "duration").glob("*.npy"))
    print(f"Uploading {len(dur_files)} duration files...")
    for i, dur_file in enumerate(sorted(dur_files)):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(dur_files)}")
        try:
            api.upload_file(
                path_or_fileobj=str(dur_file),
                path_in_repo=f"duration/{dur_file.name}",
                repo_id=DATA_REPO,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  ⚠️ {dur_file.name}: {e}")

    # Push manifests
    for manifest in ["train_manifest.json", "val_manifest.json"]:
        print(f"Uploading {manifest}...")
        api.upload_file(
            path_or_fileobj=str(processed / manifest),
            path_in_repo=manifest,
            repo_id=DATA_REPO,
            repo_type="dataset",
        )

    print("\n✓ Data push complete!")
    return True


def push_model():
    """Push current best.pt checkpoint."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        return False

    api = HfApi(token=token)
    ckpt_path = Path("checkpoints/fastspeech2/best.pt")

    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return False

    print(f"Pushing checkpoint to {MODEL_REPO}...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="best.pt",
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    print("✓ Model push complete!")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--model-only":
        push_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--data-only":
        push_data()
    else:
        push_data()
        push_model()
