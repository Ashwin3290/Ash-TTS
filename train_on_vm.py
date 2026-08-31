"""
Pull model & data from HuggingFace, resume training.

Usage on vast.ai VM:
    export HF_TOKEN="hf_xxxxx"
    python train_on_vm.py

Downloads:
- Model checkpoint from Ashwin-C9/tts-fastspeech2-ckpt
- Dataset from Ashwin-C9/tts-data-silence-fix
- Runs train_fastspeech.py --resume
"""

import os
import subprocess
from pathlib import Path

MODEL_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"
DATA_REPO = "Ashwin-C9/tts-data-silence-fix"

def download_model():
    """Download model weights from HuggingFace."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        return False

    from huggingface_hub import hf_hub_download

    ckpt_dir = Path("checkpoints/fastspeech2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from {MODEL_REPO}...")
    try:
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="best.pt",
            local_dir=str(ckpt_dir),
            token=token,
        )
        # Rename best.pt to latest.pt so --resume auto-finds it
        best_path = ckpt_dir / "best.pt"
        latest_path = ckpt_dir / "latest.pt"
        if best_path.exists():
            best_path.rename(latest_path)
        print("✓ Model checkpoint downloaded and ready for resume")
        return True
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        return False


def download_data():
    """Download dataset from HuggingFace."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        return False

    from huggingface_hub import hf_hub_download

    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {DATA_REPO}...")

    # Download manifests
    for manifest in ["train_manifest.json", "val_manifest.json"]:
        print(f"  {manifest}...")
        try:
            hf_hub_download(
                repo_id=DATA_REPO,
                filename=manifest,
                local_dir=str(processed),
                token=token,
            )
        except Exception as e:
            print(f"  ⚠️ {manifest}: {e}")

    # Download phoneme files
    print("  Downloading phoneme files...")
    try:
        import subprocess
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                DATA_REPO,
                "--include",
                "phoneme/*",
                "--local-dir",
                str(processed),
            ],
            env={**os.environ, "HF_TOKEN": token},
            check=True,
        )
    except Exception as e:
        print(f"  Note: {e}")

    # Download duration files
    print("  Downloading duration files...")
    try:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                DATA_REPO,
                "--include",
                "duration/*",
                "--local-dir",
                str(processed),
            ],
            env={**os.environ, "HF_TOKEN": token},
            check=True,
        )
    except Exception as e:
        print(f"  Note: {e}")

    print("✓ Dataset downloaded")
    return True


def train():
    """Resume training."""
    print("\nStarting training...")
    result = subprocess.run(
        ["python", "train_fastspeech.py", "--resume"],
        check=False,
    )
    return result.returncode == 0


def main():
    print("="*80)
    print("TTS SILENCE FIX - VAST.AI TRAINING")
    print("="*80)

    # Step 1: Download model
    if not download_model():
        print("❌ Failed to download model")
        return False

    # Step 2: Download data
    if not download_data():
        print("⚠️ Data download had issues, but continuing...")

    # Step 3: Train
    print("\n" + "="*80)
    if train():
        print("\n✓ Training completed!")
        return True
    else:
        print("\n❌ Training failed")
        return False


if __name__ == "__main__":
    import sys

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: Set HF_TOKEN environment variable first:")
        print("  export HF_TOKEN='hf_xxxxx'")
        sys.exit(1)

    success = main()
    sys.exit(0 if success else 1)
