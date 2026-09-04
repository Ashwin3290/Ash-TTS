"""
Upload regenerated data to HuggingFace.

Set HF_TOKEN environment variable first:
    $env:HF_TOKEN = "hf_xxx..."

Then run:
    python upload_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, Repository

# Configuration
HF_REPO_ID = "Ashwin-C9/tts-data-silence-fix"  # Change to your repo
REPO_TYPE = "dataset"
DATA_DIR = Path("data/processed")

def upload_to_hf():
    """Upload regenerated phoneme/duration files to HuggingFace."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable first")
        print("  $env:HF_TOKEN = 'hf_xxx...'")
        exit(1)

    api = HfApi(token=token)

    print(f"Uploading to {HF_REPO_ID}...")

    # Upload phoneme directory
    phoneme_dir = DATA_DIR / "phoneme"
    if phoneme_dir.exists():
        print(f"\nUploading {len(list(phoneme_dir.glob('*.npy')))} phoneme files...")
        for ph_file in sorted(phoneme_dir.glob("*.npy")):
            try:
                api.upload_file(
                    path_or_fileobj=str(ph_file),
                    path_in_repo=f"phoneme/{ph_file.name}",
                    repo_id=HF_REPO_ID,
                    repo_type=REPO_TYPE,
                    commit_message=f"Update phoneme file with silence tokens: {ph_file.name}",
                )
            except Exception as e:
                print(f"  ⚠️  {ph_file.name}: {e}")

    # Upload duration directory
    duration_dir = DATA_DIR / "duration"
    if duration_dir.exists():
        print(f"\nUploading {len(list(duration_dir.glob('*.npy')))} duration files...")
        for dur_file in sorted(duration_dir.glob("*.npy")):
            try:
                api.upload_file(
                    path_or_fileobj=str(dur_file),
                    path_in_repo=f"duration/{dur_file.name}",
                    repo_id=HF_REPO_ID,
                    repo_type=REPO_TYPE,
                    commit_message=f"Update duration file with silence: {dur_file.name}",
                )
            except Exception as e:
                print(f"  ⚠️  {dur_file.name}: {e}")

    # Upload manifests
    for manifest_file in ["train_manifest.json", "val_manifest.json"]:
        manifest_path = DATA_DIR / manifest_file
        if manifest_path.exists():
            print(f"\nUploading {manifest_file}...")
            api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo=manifest_file,
                repo_id=HF_REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Update {manifest_file}",
            )

    print("\n✓ Upload complete!")

if __name__ == "__main__":
    upload_to_hf()
