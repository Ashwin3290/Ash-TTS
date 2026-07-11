"""
One-time upload of the official jik876/hifi-gan LJ_V1 checkpoint (generator_v1
+ its config.json) to HuggingFace, so cloud_train.sh can pull it on any fresh
instance instead of you manually copying the "GAN model/" folder over.

Usage:
    python push_pretrained_hifigan.py
"""

from pathlib import Path
from huggingface_hub import HfApi

LOCAL_DIR = Path("GAN model")
REPO = "Ashwin-C9/tts-hifigan-ckpt"


def main():
    api = HfApi()
    try:
        api.repo_info(repo_id=REPO, repo_type="model")
        print(f"Repo exists: {REPO}")
    except Exception:
        api.create_repo(REPO, repo_type="model", private=True)
        print(f"Repo created: {REPO}")

    for fname in ("generator_v1", "config.json"):
        local_path = LOCAL_DIR / fname
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} not found")
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fname,
            repo_id=REPO, repo_type="model",
        )

    print(f"Done. https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
