"""
Upload processed data as multiple tar.gz parts to HuggingFace, compressed in
parallel across CPU workers using native `tar` (Python's tarfile module is
drastically slower — file-by-file gzip overhead, ~100x slower in practice).

Usage:
    python upload_data.py
    python upload_data.py --workers 8
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

PROCESSED_DIR = Path("data/processed")
DATA_DIR      = Path("data")
PART_PREFIX   = "processed_part"
DATASET_REPO  = "Ashwin-C9/tts-ljspeech-processed"
MODEL_REPO    = "Ashwin-C9/tts-fastspeech2-ckpt"


def create_archive_parts(n_workers):
    if shutil.which("tar") is None:
        raise RuntimeError("native `tar` not found on PATH — required for fast archiving")

    # exclude dotfiles/dotdirs (e.g. a stray .cache/ left by a prior huggingface_hub
    # upload_folder call) — only the five feature dirs + top-level manifests belong here
    files = sorted(str(p.relative_to(DATA_DIR)).replace("\\", "/")
                   for p in PROCESSED_DIR.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.parts))
    print(f"Archiving {len(files)} files across {n_workers} workers...")

    chunks = [files[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]  # drop empty chunks if files < workers

    part_paths = []
    procs = []
    for i, chunk in enumerate(chunks):
        filelist = DATA_DIR / f".filelist_{i}.txt"
        filelist.write_text("\n".join(chunk), encoding="utf-8")
        part_path = DATA_DIR / f"{PART_PREFIX}{i:03d}.tar.gz"
        part_paths.append(part_path)
        proc = subprocess.Popen(
            ["tar", "-czf", str(part_path), "-C", str(DATA_DIR), "-T", str(filelist)],
        )
        procs.append((proc, filelist))

    for proc, filelist in procs:
        ret = proc.wait()
        filelist.unlink(missing_ok=True)
        if ret != 0:
            raise RuntimeError(f"tar exited with code {ret}")

    total_size = sum(p.stat().st_size for p in part_paths) / 1e9
    print(f"Done. {len(part_paths)} parts, {total_size:.2f} GB total.")
    return part_paths


def reset_and_upload_dataset(part_paths, n_workers):
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

    def upload_one(part_path):
        api.upload_file(
            path_or_fileobj=str(part_path),
            path_in_repo=part_path.name,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        return part_path.name

    print(f"Uploading {len(part_paths)} parts...")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(upload_one, p) for p in part_paths]
        for f in as_completed(futures):
            print(f"  uploaded: {f.result()}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel tar/upload workers (default: cpu_count - 1)")
    args = parser.parse_args()

    import os
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    parts = create_archive_parts(n_workers)
    reset_and_upload_dataset(parts, n_workers)
    ensure_model_repo()
    print("\nAll done. You can delete data/processed_part*.tar.gz if you want to save disk space.")
