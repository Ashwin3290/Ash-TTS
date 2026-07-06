"""
Downloads and extracts LJSpeech-1.1 dataset.
Run once before preprocessing.

LJSpeech: ~2.6GB download, ~13000 single-speaker English clips, total ~24hrs audio.
Source: https://keithito.com/LJ-Speech-Dataset/
"""

import urllib.request
import tarfile
import os
from pathlib import Path

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DEST = Path("data")
ARCHIVE = DEST / "LJSpeech-1.1.tar.bz2"
EXPECTED_DIR = DEST / "LJSpeech-1.1"


def reporthook(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    print(f"\r  downloading... {percent}%", end="", flush=True)


def download():
    DEST.mkdir(parents=True, exist_ok=True)

    if EXPECTED_DIR.exists():
        print(f"LJSpeech already exists at {EXPECTED_DIR}, skipping download.")
        return

    print(f"Downloading LJSpeech-1.1 (~2.6GB)...")
    urllib.request.urlretrieve(URL, ARCHIVE, reporthook=reporthook)
    print("\nExtracting...")
    with tarfile.open(ARCHIVE, "r:bz2") as tar:
        tar.extractall(DEST)
    print(f"Done. Dataset at: {EXPECTED_DIR}")

    print("Cleaning up archive...")
    os.remove(ARCHIVE)

    # quick sanity check
    wavs = list((EXPECTED_DIR / "wavs").glob("*.wav"))
    meta = EXPECTED_DIR / "metadata.csv"
    print(f"  {len(wavs)} wav files found")
    print(f"  metadata.csv exists: {meta.exists()}")


if __name__ == "__main__":
    download()
