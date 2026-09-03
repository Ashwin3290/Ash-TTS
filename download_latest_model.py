
import os
import sys
import tarfile
from pathlib import Path


token = os.environ.get("HF_TOKEN")

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