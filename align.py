"""
CTC Forced Aligner using torchaudio MMS_FA.

MMS_FA is a character-level model — it aligns raw text characters (a, b, c...)
against audio, NOT IPA phonemes. The strategy here is:

  1. Align words to audio using MMS_FA character alignment
  2. Phonemize each word separately to get its phoneme sequence
  3. Distribute the word's frames across its phonemes proportionally

This gives us per-phoneme durations without needing MFA or Kaldi.

Usage:
    python align.py
    python align.py --overwrite
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as BUNDLE
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from config import audio as acfg, paths

ALIGN_SR      = BUNDLE.sample_rate  # 16000
LABELS        = BUNDLE.get_labels()
LABEL2IDX     = {l: i for i, l in enumerate(LABELS)}
BLANK_IDX     = LABEL2IDX.get('|', 0)
W2V_HOP       = 320
W2V_FRAME_SEC = W2V_HOP / ALIGN_SR
MEL_FRAME_SEC = acfg.hop_length / acfg.sample_rate


def get_phonemizer():
    return EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)


def phonemize_words(words, backend):
    """
    Phonemize all words at once using word boundary separator.
    This matches what preprocess.py does, avoiding coarticulation mismatches.
    """
    sep = Separator(phone=' ', word='| ', syllable='')
    full_text = ' '.join(words)
    ph_str = backend.phonemize([full_text], separator=sep)[0]
    # split on word boundary marker
    word_segments = ph_str.split('|')
    results = []
    for seg in word_segments:
        phones = [p for p in seg.split() if p.strip()]
        results.append(phones if phones else ['_'])
    # pad or trim to match word count
    while len(results) < len(words):
        results.append(['_'])
    return results[:len(words)]


def text_to_char_tokens(text):
    words = text.lower().split()
    token_ids     = []
    word_of_token = []
    valid_words   = []

    for w_idx, word in enumerate(words):
        # strip punctuation for alignment, keep only alpha chars in MMS vocab
        # critically exclude index 0 (blank token '-') from tokens
        chars = [c for c in word if c in LABEL2IDX and LABEL2IDX[c] != BLANK_IDX]
        if not chars:
            continue
        valid_words.append(word)
        for c in chars:
            token_ids.append(LABEL2IDX[c])
            word_of_token.append(len(valid_words) - 1)

    return token_ids, word_of_token, valid_words


def align_utterance(wav_path, text, n_phones, model, device, backend):
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != ALIGN_SR:
        waveform = torchaudio.functional.resample(waveform, sr, ALIGN_SR)
    waveform = waveform.mean(dim=0, keepdim=True).to(device)

    with torch.inference_mode():
        emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)

    token_ids, word_of_token, words = text_to_char_tokens(text)
    if not token_ids:
        return None

    try:
        token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emissions, token_ids_t.unsqueeze(0), blank=BLANK_IDX
        )
        spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])
    except Exception:
        return None

    if not spans:
        return None

    # spans[i].token is the label ID, not the position in token_ids list
    # we need to map each span back to its position sequentially
    # since forced_align is monotonic, span i corresponds to token_ids[i]
    n_words    = len(words)
    word_start = [None] * n_words
    word_end   = [None] * n_words

    for span_idx, span in enumerate(spans):
        if span_idx >= len(word_of_token):
            continue
        w_idx = word_of_token[span_idx]
        if word_start[w_idx] is None:
            word_start[w_idx] = span.start
        word_end[w_idx] = span.end

    # fill missing word spans from neighbours
    for i in range(n_words):
        if word_start[i] is None:
            prev_end = word_end[i-1] if i > 0 and word_end[i-1] is not None else 0
            word_start[i] = prev_end
            word_end[i]   = prev_end + 1

    # phonemize each word to get phone count per word
    word_phones = phonemize_words(words, backend)

    # distribute word frames across phonemes
    durations = []
    for w_idx, w_phones in enumerate(word_phones):
        start_sec   = word_start[w_idx] * W2V_FRAME_SEC
        end_sec     = word_end[w_idx]   * W2V_FRAME_SEC
        word_frames = max(len(w_phones), round((end_sec - start_sec) / MEL_FRAME_SEC))
        n_ph = len(w_phones)
        base = word_frames // n_ph
        rem  = word_frames % n_ph
        for j in range(n_ph):
            durations.append(base + (1 if j < rem else 0))

    # match length to n_phones
    if len(durations) != n_phones:
        if not durations:
            return None
        total = sum(durations)
        base  = total // n_phones
        rem   = total % n_phones
        durations = [base + (1 if i < rem else 0) for i in range(n_phones)]

    return durations


def run_alignment(data_root, processed_dir, overwrite=False, device=None):
    data_root     = Path(data_root)
    processed_dir = Path(processed_dir)
    wav_dir       = data_root / "wavs"
    dur_dir       = processed_dir / "duration"
    mel_dir       = processed_dir / "mel"
    ph_dir        = processed_dir / "phoneme"
    dur_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not ph_dir.exists():
        raise RuntimeError("Run preprocess.py first — phoneme files not found.")

    utt_ids = [f.stem for f in ph_dir.glob("*.npy")]
    print(f"Found {len(utt_ids)} preprocessed utterances")

    if not overwrite:
        utt_ids = [u for u in utt_ids if not (dur_dir / f"{u}.npy").exists()]
        print(f"Need alignment for {len(utt_ids)} utterances")

    if not utt_ids:
        print("All utterances already aligned.")
        return

    metadata_path = data_root / "metadata.csv"
    text_map = {}
    for line in open(metadata_path, encoding="utf-8"):
        parts = line.strip().split("|")
        if len(parts) >= 3:
            text_map[parts[0]] = parts[2]

    print("Loading MMS_FA model...")
    model = BUNDLE.get_model().to(device)
    model.eval()
    print("Model loaded.")

    backend  = get_phonemizer()
    success  = 0
    fallback = 0
    failed   = 0

    for utt_id in tqdm(utt_ids, desc="aligning"):
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            failed += 1
            continue

        text = text_map.get(utt_id, "")
        if not text:
            failed += 1
            continue

        ph_ids   = np.load(ph_dir / f"{utt_id}.npy")
        n_phones = len(ph_ids)

        try:
            durations = align_utterance(wav_path, text, n_phones, model, device, backend)
        except Exception:
            durations = None

        if durations is None:
            mel_path = mel_dir / f"{utt_id}.npy"
            if mel_path.exists():
                n_frames = np.load(mel_path).shape[0]
                base = n_frames // n_phones
                rem  = n_frames % n_phones
                durations = [base + (1 if i < rem else 0) for i in range(n_phones)]
                fallback += 1
            else:
                failed += 1
                continue

        np.save(dur_dir / f"{utt_id}.npy", np.array(durations, dtype=np.int32))
        success += 1

    print(f"\nDone. {success} saved ({success - fallback} real alignments, "
          f"{fallback} fallback), {failed} failed.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     default=str(paths.data_root))
    parser.add_argument("--processed-dir", default=str(paths.processed_dir))
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--device",        default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    run_alignment(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        overwrite=args.overwrite,
        device=device,
    )