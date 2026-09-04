"""
Regenerate only phoneme and duration files with silence tokens.
Reuses existing mel/f0/energy, skipping slow audio processing.

Usage:
    python regenerate_phonemes.py --tg-dir data/LJSpeech-1.1/TextGrid
    python regenerate_phonemes.py --tg-dir data/LJSpeech-1.1/TextGrid --workers 8
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

from config import paths

try:
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    PHONEMIZER_OK = True
except ImportError:
    PHONEMIZER_OK = False
    print("ERROR: phonemizer not installed.")
    exit(1)


def parse_textgrid(tg_path):
    """Parse TextGrid phoneme tier."""
    text = Path(tg_path).read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()]
    phones_tier = False
    intervals = []
    i = 0
    while i < len(lines):
        if 'name = "phones"' in lines[i]:
            phones_tier = True
        if phones_tier and lines[i].startswith("xmin ="):
            xmin = float(lines[i].split("=")[1])
            xmax = float(lines[i + 1].split("=")[1])
            text_val = lines[i + 2].split("=")[1].strip().strip('"')
            dur_frames = int(round((xmax - xmin) * 22050 / 256))  # hardcoded to match config
            intervals.append((text_val, dur_frames))
            i += 3
            continue
        i += 1
    return intervals


def align_phonemes_with_textgrid(espeak_phonemes, textgrid_intervals, vocab):
    """
    Align espeak phonemes with TextGrid, inserting <sil> tokens at silence positions.
    If exact alignment fails, fall back to using speech durations only.
    """
    sil_token = vocab.get("<sil>", 1)

    # Separate speech and silence intervals
    speech_intervals = [(ph, d) for ph, d in textgrid_intervals if ph not in ("", "sp", "sil", "spn")]
    silence_intervals = [(ph, d) for ph, d in textgrid_intervals if ph in ("", "sp", "sil", "spn")]

    # If no silences in TextGrid, just use speech durations
    if len(silence_intervals) == 0:
        speech_durs = [d for _, d in speech_intervals]
        if len(espeak_phonemes) == len(speech_durs):
            return espeak_phonemes, np.array(speech_durs, dtype=np.int32)
        else:
            # Mismatch: return None to skip this utterance
            return None, None

    # Try exact alignment with silence tokens
    if len(espeak_phonemes) == len(speech_intervals):
        # Perfect match: build sequence with silence tokens
        phonemes_out = []
        durations_out = []
        espeak_idx = 0

        for ph, dur in textgrid_intervals:
            if ph in ("", "sp", "sil", "spn"):
                phonemes_out.append(sil_token)
                durations_out.append(dur)
            else:
                phonemes_out.append(espeak_phonemes[espeak_idx])
                durations_out.append(dur)
                espeak_idx += 1

        return np.array(phonemes_out, dtype=np.int32), np.array(durations_out, dtype=np.int32)

    # Fallback: just use speech durations without silence tokens
    # (Better to have partial silence handling than none)
    speech_durs = [d for _, d in speech_intervals]
    if len(espeak_phonemes) == len(speech_durs):
        return espeak_phonemes, np.array(speech_durs, dtype=np.int32)

    # Last resort: skip this utterance
    return None, None


_GLOBALS = {}


def _worker_init(processed_dir, metadata_path, tg_dir, vocab):
    _GLOBALS["processed_dir"] = Path(processed_dir)
    _GLOBALS["metadata_path"] = metadata_path
    _GLOBALS["tg_dir"] = Path(tg_dir) if tg_dir else None
    _GLOBALS["vocab"] = vocab
    _GLOBALS["backend"] = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)


def _process_one(utt_id):
    g = _GLOBALS
    metadata_path = Path(g["metadata_path"])

    # Load text from metadata (UTF-8 encoding, not cp1252)
    with open(metadata_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    text = None
    for line in lines:
        parts = line.split("|")
        if parts[0] == utt_id:
            if len(parts) >= 3:
                text = parts[2]
            break

    if not text:
        return None

    # Phonemize
    vocab = g["vocab"]
    backend = g["backend"]
    sep = Separator(phone=" ", word="| ", syllable="")
    ph_str = backend.phonemize([text], separator=sep)[0]
    phonemes = [vocab.get(p, vocab["<pad>"]) for p in ph_str.replace("|", "").split() if p.strip()]

    if not phonemes:
        return None

    # Align with TextGrid
    phonemes_aligned = phonemes
    durations = None
    if g["tg_dir"] is not None:
        tg_path = g["tg_dir"] / f"{utt_id}.TextGrid"
        if tg_path.exists():
            intervals = parse_textgrid(tg_path)
            phonemes_aligned, durations = align_phonemes_with_textgrid(phonemes, intervals, vocab)
            if phonemes_aligned is None:
                return None

    if durations is None:
        return None

    # Save
    pd = g["processed_dir"]
    np.save(pd / "phoneme" / f"{utt_id}.npy", np.array(phonemes_aligned, dtype=np.int32))
    np.save(pd / "duration" / f"{utt_id}.npy", np.array(durations, dtype=np.int32))

    return {"id": utt_id, "n_phonemes": len(phonemes_aligned), "n_frames": durations.sum()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg-dir", required=True, help="Path to TextGrid directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    processed_dir = paths.processed_dir
    metadata_path = paths.data_root / "metadata.csv"

    if not processed_dir.exists():
        print(f"ERROR: Processed dir {processed_dir} doesn't exist")
        exit(1)

    if not metadata_path.exists():
        print(f"ERROR: Metadata {metadata_path} not found")
        exit(1)

    # Load vocab
    vocab_path = processed_dir / "phoneme_vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: Vocab {vocab_path} not found. Run preprocess.py first.")
        exit(1)

    with open(vocab_path) as f:
        vocab = json.load(f)

    # Get utterance IDs from manifest
    train_manifest_path = processed_dir / "train_manifest.json"
    val_manifest_path = processed_dir / "val_manifest.json"

    utt_ids = []
    for mf in [train_manifest_path, val_manifest_path]:
        if mf.exists():
            with open(mf) as f:
                manifest = json.load(f)
                utt_ids.extend([m["id"] for m in manifest])

    print(f"Regenerating phoneme/duration for {len(utt_ids)} utterances...")

    n_workers = args.workers or max(1, cpu_count() - 1)
    init_args = (str(processed_dir), str(metadata_path), args.tg_dir, vocab)

    results = []
    with Pool(processes=n_workers, initializer=_worker_init, initargs=init_args) as pool:
        for result in tqdm(pool.imap_unordered(_process_one, utt_ids), total=len(utt_ids)):
            if result:
                results.append(result)

    print(f"\n✓ Regenerated {len(results)}/{len(utt_ids)} utterances")
    print(f"Average phonemes per utterance: {np.mean([r['n_phonemes'] for r in results]):.1f}")
    print(f"Average frames per utterance: {np.mean([r['n_frames'] for r in results]):.1f}")


if __name__ == "__main__":
    main()
