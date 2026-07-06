"""
Generates .lab files required by Montreal Forced Aligner from LJSpeech metadata.

MFA expects each wav file to have a matching .lab file in the same directory
containing the plain text transcript.

Run this BEFORE running MFA alignment:
    python make_labs.py
    python make_labs.py --data-root C:/path/to/LJSpeech-1.1

Then run MFA in WSL:
    mfa align /mnt/c/.../LJSpeech-1.1/wavs english_us_arpa english_us_arpa /mnt/c/.../TextGrid --clean
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from config import paths


def make_labs(data_root):
    data_root = Path(data_root)
    wav_dir   = data_root / "wavs"
    metadata  = data_root / "metadata.csv"

    if not metadata.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"wavs directory not found at {wav_dir}")

    lines = open(metadata, encoding="utf-8").readlines()
    written = 0
    skipped = 0

    for line in tqdm(lines, desc="writing .lab files"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        utt_id = parts[0]
        text   = parts[2]   # normalised text (column 3)

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            skipped += 1
            continue

        lab_path = wav_dir / f"{utt_id}.lab"
        lab_path.write_text(text.strip(), encoding="utf-8")
        written += 1

    print(f"Done. {written} .lab files written, {skipped} skipped (missing wavs).")
    print(f"Lab files are in: {wav_dir}")
    print(f"\nNext step — run in WSL:")
    print(f"  mfa align /mnt/c/.../LJSpeech-1.1/wavs english_us_arpa english_us_arpa /mnt/c/.../TextGrid --clean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(paths.data_root))
    args = parser.parse_args()
    make_labs(data_root=args.data_root)
