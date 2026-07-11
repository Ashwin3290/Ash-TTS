"""
Re-extract F0 only, after the pitch_ceiling fix in preprocess.py.

The original extraction used pitch_ceiling=2093 Hz, which made Praat track
harmonics and inflated the dataset f0 stats (std ~284 Hz vs a plausible ~50 Hz).
This overwrites data/processed/f0/*.npy and recomputes stats.json.
Mel/energy/phoneme/duration files are untouched.

Usage:
    python reextract_f0.py
    python reextract_f0.py --workers 8
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from config import audio as acfg, paths
from preprocess import extract_f0, compute_pitch_energy_stats


def _one(utt_id):
    wav_path = paths.data_root / "wavs" / f"{utt_id}.wav"
    mel_path = paths.processed_dir / "mel" / f"{utt_id}.npy"
    if not wav_path.exists() or not mel_path.exists():
        return utt_id, False
    try:
        n_frames = np.load(mel_path, mmap_mode="r").shape[0]
        wav, sr = librosa.load(str(wav_path), sr=acfg.sample_rate, mono=True)
        f0 = extract_f0(wav, sr, n_frames)
        np.save(paths.processed_dir / "f0" / f"{utt_id}.npy", f0)
        return utt_id, True
    except Exception:
        return utt_id, False


def main(n_workers=None):
    utt_ids = [f.stem for f in (paths.processed_dir / "f0").glob("*.npy")]
    print(f"Re-extracting F0 for {len(utt_ids)} utterances...")

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    failed = 0
    with Pool(processes=n_workers) as pool:
        for _, ok in tqdm(pool.imap_unordered(_one, utt_ids, chunksize=16),
                          total=len(utt_ids), desc="f0"):
            if not ok:
                failed += 1
    print(f"Done. {failed} failed.")

    # stats.json early-returns if it exists — remove so it recomputes
    stats_path = paths.processed_dir / "stats.json"
    if stats_path.exists():
        stats_path.unlink()
    stats = compute_pitch_energy_stats(paths.processed_dir)
    print(f"New stats: f0 mean={stats['f0_mean']:.1f}Hz std={stats['f0_std']:.1f}Hz "
          f"max={stats['f0_max']:.1f}Hz")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    main(n_workers=args.workers)
