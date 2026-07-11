"""
Preprocessing pipeline for LJSpeech.

Speedups over v1:
  - Multiprocessing: processes N files in parallel across CPU cores
  - F0: parselmouth (Praat) instead of pyin — 5-10x faster, comparable quality
  - Phonemizer: batched per worker to avoid repeated subprocess spawning

Run:
    python preprocess.py --tg-dir data/LJSpeech-1.1/TextGrid
    python preprocess.py --tg-dir data/LJSpeech-1.1/TextGrid --workers 8
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Windows: set espeak DLL path before phonemizer import
if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

from config import audio as acfg, paths

try:
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    PHONEMIZER_OK = True
except ImportError:
    PHONEMIZER_OK = False
    print("WARNING: phonemizer not installed.")

try:
    import parselmouth
    PARSELMOUTH_OK = True
except ImportError:
    PARSELMOUTH_OK = False
    print("WARNING: parselmouth not installed. Falling back to pyin (slow).")
    print("         pip install praat-parselmouth")


# ---- TextGrid parser ----
def parse_textgrid(tg_path):
    text  = Path(tg_path).read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()]
    phones_tier = False
    intervals   = []
    i = 0
    while i < len(lines):
        if 'name = "phones"' in lines[i]:
            phones_tier = True
        if phones_tier and lines[i].startswith("xmin ="):
            xmin      = float(lines[i].split("=")[1])
            xmax      = float(lines[i+1].split("=")[1])
            text_val  = lines[i+2].split("=")[1].strip().strip('"')
            dur_frames = int(round((xmax - xmin) * acfg.sample_rate / acfg.hop_length))
            intervals.append((text_val, dur_frames))
            i += 3
            continue
        i += 1
    return intervals


# ---- Feature extraction ----
def extract_mel(wav, sr):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=acfg.n_fft, hop_length=acfg.hop_length,
        win_length=acfg.win_length, n_mels=acfg.n_mels,
        fmin=acfg.fmin, fmax=acfg.fmax,
    )
    log_mel = np.log(np.clip(mel, 1e-5, None))
    log_mel = np.clip(log_mel, acfg.mel_min, acfg.mel_max)
    log_mel = (log_mel - acfg.mel_min) / (acfg.mel_max - acfg.mel_min) * 2 - 1
    return log_mel.T.astype(np.float32)


def extract_f0_parselmouth(wav, sr, n_frames):
    """Fast F0 via Praat. ~5-10x faster than pyin."""
    snd   = parselmouth.Sound(wav, sampling_frequency=sr)
    # ceiling 600 Hz: speech range for a female speaker — a music-range ceiling
    # (e.g. 2093 Hz) makes Praat track harmonics, inflating f0 stats ~5x
    pitch = snd.to_pitch(time_step=acfg.hop_length / sr,
                         pitch_floor=65.0, pitch_ceiling=600.0)
    xs    = pitch.xs()
    f0    = np.array([pitch.get_value_at_time(t) for t in xs], dtype=np.float32)
    f0    = np.nan_to_num(f0, nan=0.0)
    if len(f0) >= n_frames:
        return f0[:n_frames]
    return np.pad(f0, (0, n_frames - len(f0)))


def extract_f0_pyin(wav, sr, n_frames):
    """Fallback F0 via pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=acfg.hop_length, fill_na=0.0,
    )
    f0 = f0[:n_frames] if len(f0) >= n_frames else np.pad(f0, (0, n_frames - len(f0)))
    vf = voiced_flag[:n_frames] if len(voiced_flag) >= n_frames else np.pad(voiced_flag, (0, n_frames - len(voiced_flag)))
    return np.where(vf, f0, 0.0).astype(np.float32)


def extract_f0(wav, sr, n_frames):
    if PARSELMOUTH_OK:
        return extract_f0_parselmouth(wav, sr, n_frames)
    return extract_f0_pyin(wav, sr, n_frames)


def extract_energy(wav, n_frames):
    energy = librosa.feature.rms(
        y=wav, frame_length=acfg.win_length, hop_length=acfg.hop_length)[0]
    energy = energy[:n_frames] if len(energy) >= n_frames else np.pad(energy, (0, n_frames - len(energy)))
    return energy.astype(np.float32)


# ---- Vocab ----
def build_phoneme_vocab(metadata_path, use_phonemizer=True):
    vocab_path = paths.processed_dir / "phoneme_vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            return json.load(f)

    print("Building phoneme vocabulary...")
    lines = open(metadata_path, encoding="utf-8").readlines()
    texts = [l.strip().split("|")[2] for l in lines if l.strip()]

    if use_phonemizer and PHONEMIZER_OK:
        backend      = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
        all_phonemes = set()
        batch_size   = 256
        sep = Separator(phone=' ', word='| ', syllable='')
        for i in tqdm(range(0, len(texts), batch_size), desc="phonemizing vocab"):
            for seq in backend.phonemize(texts[i:i+batch_size], separator=sep):
                all_phonemes.update(p for p in seq.replace('|', '').split() if p.strip())
    else:
        all_phonemes = set("".join(texts))

    vocab = {"<pad>": 0, "<sil>": 1}
    for ph in sorted(all_phonemes):
        if ph not in vocab:
            vocab[ph] = len(vocab)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Phoneme vocab size: {len(vocab)}")
    return vocab


# ---- Per-utterance worker ----
_GLOBALS = {}

def _worker_init(processed_dir, wav_dir, tg_dir, vocab, use_phonemizer):
    _GLOBALS["processed_dir"]  = Path(processed_dir)
    _GLOBALS["wav_dir"]        = Path(wav_dir)
    _GLOBALS["tg_dir"]         = Path(tg_dir) if tg_dir else None
    _GLOBALS["vocab"]          = vocab
    _GLOBALS["use_phonemizer"] = use_phonemizer

    if use_phonemizer and PHONEMIZER_OK:
        if sys.platform == "win32":
            os.environ.setdefault(
                "PHONEMIZER_ESPEAK_LIBRARY",
                r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
            )
        _GLOBALS["backend"] = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
    else:
        _GLOBALS["backend"] = None


def _process_one(args):
    utt_id, text = args
    g = _GLOBALS

    wav_path = g["wav_dir"] / f"{utt_id}.wav"
    if not wav_path.exists():
        return None

    try:
        wav, sr = librosa.load(str(wav_path), sr=acfg.sample_rate, mono=True)
    except Exception:
        return None

    if len(wav) / sr > acfg.max_wav_length:
        return None

    mel      = extract_mel(wav, sr)
    n_frames = mel.shape[0]
    f0       = extract_f0(wav, sr, n_frames)
    energy   = extract_energy(wav, n_frames)

    vocab   = g["vocab"]
    backend = g["backend"]
    if backend is not None:
        sep = Separator(phone=' ', word='| ', syllable='')
        ph_str = backend.phonemize([text], separator=sep)[0]
        # split on spaces but filter out the word boundary marker
        phonemes = [vocab.get(p, vocab["<pad>"]) for p in ph_str.replace('|', '').split() if p.strip()]
    else:
        phonemes = [vocab.get(c, vocab["<pad>"]) for c in text]

    if not phonemes:
        return None

    durations = None
    if g["tg_dir"] is not None:
        tg_path = g["tg_dir"] / f"{utt_id}.TextGrid"
        if tg_path.exists():
            intervals    = parse_textgrid(tg_path)
            ph_intervals = [(ph, d) for ph, d in intervals if ph not in ("", "sp", "sil", "spn")]
            durs         = [d for _, d in ph_intervals]
            if len(durs) == len(phonemes):
                durations = durs

    pd = g["processed_dir"]
    np.save(pd / "mel"     / f"{utt_id}.npy", mel)
    np.save(pd / "f0"      / f"{utt_id}.npy", f0)
    np.save(pd / "energy"  / f"{utt_id}.npy", energy)
    np.save(pd / "phoneme" / f"{utt_id}.npy", np.array(phonemes, dtype=np.int32))
    if durations is not None:
        np.save(pd / "duration" / f"{utt_id}.npy", np.array(durations, dtype=np.int32))

    return {
        "id":           utt_id,
        "text":         text,
        "n_phonemes":   len(phonemes),
        "n_frames":     n_frames,
        "has_duration": durations is not None,
    }


# ---- Stats ----
def compute_pitch_energy_stats(processed_dir):
    processed_dir = Path(processed_dir)
    stats_path    = processed_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)

    print("Computing pitch/energy statistics...")
    all_f0, all_energy = [], []
    for f in (processed_dir / "f0").glob("*.npy"):
        f0 = np.load(f)
        all_f0.extend(f0[f0 > 0].tolist())
    for f in (processed_dir / "energy").glob("*.npy"):
        all_energy.extend(np.load(f).tolist())

    stats = {
        "f0_mean":     float(np.mean(all_f0)),
        "f0_std":      float(np.std(all_f0)),
        "f0_min":      float(np.min(all_f0)),
        "f0_max":      float(np.max(all_f0)),
        "energy_mean": float(np.mean(all_energy)),
        "energy_std":  float(np.std(all_energy)),
        "energy_min":  float(np.min(all_energy)),
        "energy_max":  float(np.max(all_energy)),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: f0 mean={stats['f0_mean']:.1f}Hz std={stats['f0_std']:.1f}Hz")
    return stats


# ---- Main ----
def process_dataset(data_root, processed_dir, tg_dir=None, use_phonemizer=True, n_workers=None):
    data_root     = Path(data_root)
    processed_dir = Path(processed_dir)
    wav_dir       = data_root / "wavs"
    metadata_path = data_root / "metadata.csv"

    for subdir in ["mel", "f0", "energy", "duration", "phoneme"]:
        (processed_dir / subdir).mkdir(parents=True, exist_ok=True)

    vocab = build_phoneme_vocab(metadata_path, use_phonemizer)

    lines = [l.strip() for l in open(metadata_path, encoding="utf-8") if l.strip()]
    jobs  = [(p[0], p[2]) for l in lines if len(p := l.split("|")) >= 3]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    print(f"Processing {len(jobs)} utterances with {n_workers} workers...")
    print(f"F0 backend: {'parselmouth (fast)' if PARSELMOUTH_OK else 'pyin (slow)'}")

    init_args = (str(processed_dir), str(wav_dir), tg_dir, vocab, use_phonemizer)

    manifest = []
    skipped  = 0

    with Pool(processes=n_workers, initializer=_worker_init, initargs=init_args) as pool:
        for result in tqdm(pool.imap_unordered(_process_one, jobs, chunksize=16),
                           total=len(jobs), desc="processing"):
            if result is None:
                skipped += 1
            else:
                manifest.append(result)

    compute_pitch_energy_stats(processed_dir)

    random.seed(42)
    random.shuffle(manifest)
    split          = int(0.95 * len(manifest))
    train_manifest = manifest[:split]
    val_manifest   = manifest[split:]

    with open(processed_dir / "train_manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(processed_dir / "val_manifest.json", "w") as f:
        json.dump(val_manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} processed, {skipped} skipped.")
    print(f"Train: {len(train_manifest)}  Val: {len(val_manifest)}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     default=str(paths.data_root))
    parser.add_argument("--processed-dir", default=str(paths.processed_dir))
    parser.add_argument("--tg-dir",        default=None)
    parser.add_argument("--no-phonemizer", action="store_true")
    parser.add_argument("--workers",       type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    args = parser.parse_args()

    process_dataset(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        tg_dir=args.tg_dir,
        use_phonemizer=not args.no_phonemizer,
        n_workers=args.workers,
    )