"""
Phone-level CTC forced aligner using facebook/wav2vec2-lv-60-espeak-cv-ft.

The previous version of this file aligned WORDS (via character-level MMS_FA)
and split each word's frames uniformly across its phonemes. That destroyed
real duration structure: 75% of all phoneme durations landed in a 7-10 frame
band (true speech has stops at ~3-5 frames and stressed vowels at 10-25),
so the duration predictor collapsed to a near-constant ~8.8 frames and
synthesis came out with a halting, evenly-paced, robotic rhythm.

This version force-aligns our exact espeak phoneme sequences directly
against a CTC model that emits espeak IPA phones — same inventory, verified
1:1 coverage after stripping stress marks. Durations come from real acoustic
evidence per phoneme, not uniform splitting.

Reads phoneme ids straight from data/processed/phoneme/*.npy (no
phonemizer/espeak needed at alignment time).

Usage:
    python align.py
    python align.py --overwrite
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from config import audio as acfg, paths

ALIGNER_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ALIGN_SR      = 16000
W2V_HOP       = 320                      # wav2vec2 conv stack: 20ms per emission frame
W2V_FRAME_SEC = W2V_HOP / ALIGN_SR
MEL_FRAME_SEC = acfg.hop_length / acfg.sample_rate

STRESS_MARKS = ("ˈ", "ˌ")      # ˈ primary, ˌ secondary


def strip_stress(phone):
    for m in STRESS_MARKS:
        phone = phone.replace(m, "")
    return phone


def load_aligner(device):
    """CTC model + its phone vocab.

    Avoids the HF tokenizer class (it insists on initialising a phonemizer
    backend, which needs espeak, just to hold a vocab dict we can read from
    vocab.json directly) and avoids from_pretrained for the weights (the
    repo only ships pytorch_model.bin, which newer transformers refuses to
    torch.load on torch<2.6 — loading with weights_only=True ourselves is
    safe and version-agnostic)."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
    config = Wav2Vec2Config.from_pretrained(ALIGNER_MODEL)
    model = Wav2Vec2ForCTC(config)
    bin_path = hf_hub_download(ALIGNER_MODEL, "pytorch_model.bin")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tolerate only cosmetic mismatches (e.g. masked_spec_embed buffers)
    real_missing = [k for k in missing if not k.endswith("masked_spec_embed")]
    if real_missing or unexpected:
        raise RuntimeError(f"aligner weight mismatch: missing={real_missing} "
                           f"unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    vocab_path = hf_hub_download(ALIGNER_MODEL, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        aligner_vocab = json.load(f)
    blank_id = aligner_vocab["<pad>"]
    return model, aligner_vocab, blank_id


def distribute_to_frames(boundaries_sec, n_frames):
    """
    Convert per-phoneme boundary times (seconds, len n_phones+1, last entry
    is total audio duration) into integer mel-frame durations that sum to
    exactly n_frames, each at least 1.
    """
    n_phones = len(boundaries_sec) - 1
    total_sec = boundaries_sec[-1]
    if total_sec <= 0:
        return None
    # scale boundaries onto [0, n_frames] then round — cumulative rounding
    # guarantees the sum is exact, unlike rounding each span independently
    bounds = np.round(np.array(boundaries_sec) / total_sec * n_frames).astype(np.int64)
    bounds[0], bounds[-1] = 0, n_frames
    durations = np.diff(bounds)
    # enforce min 1 frame per phoneme, stealing from the largest spans
    for i in np.where(durations < 1)[0]:
        need = 1 - durations[i]
        donor = int(np.argmax(durations))
        if durations[donor] - need < 1:
            return None  # utterance shorter than its phoneme count — hopeless
        durations[donor] -= need
        durations[i] += need
    return durations.tolist()


def align_utterance(wav_path, aligner_ids, n_mel_frames, model, device, blank_id):
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != ALIGN_SR:
        waveform = torchaudio.functional.resample(waveform, sr, ALIGN_SR)
    waveform = waveform.mean(dim=0, keepdim=True)
    # lv60 models are trained on zero-mean/unit-variance input
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
    waveform = waveform.to(device)

    with torch.inference_mode():
        emissions = model(waveform).logits
    emissions = torch.log_softmax(emissions.float(), dim=-1)

    tokens = torch.tensor(aligner_ids, dtype=torch.long, device=device).unsqueeze(0)
    try:
        aligned, scores = torchaudio.functional.forced_align(
            emissions, tokens, blank=blank_id
        )
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])
    except Exception:
        return None

    if len(spans) != len(aligner_ids):
        return None

    # boundary between phones i-1 and i = start of span i; inter-phone gaps
    # (CTC blanks) therefore attach to the preceding phone. Head silence goes
    # to the first phone, tail silence to the last.
    n_emission = emissions.size(1)
    boundaries_sec = [0.0]
    for span in spans[1:]:
        boundaries_sec.append(span.start * W2V_FRAME_SEC)
    boundaries_sec.append(n_emission * W2V_FRAME_SEC)

    return distribute_to_frames(boundaries_sec, n_mel_frames)


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

    with open(processed_dir / "phoneme_vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    utt_ids = [f.stem for f in ph_dir.glob("*.npy")]
    print(f"Found {len(utt_ids)} preprocessed utterances")

    if not overwrite:
        utt_ids = [u for u in utt_ids if not (dur_dir / f"{u}.npy").exists()]
        print(f"Need alignment for {len(utt_ids)} utterances")

    if not utt_ids:
        print("All utterances already aligned.")
        return

    print(f"Loading {ALIGNER_MODEL}...")
    model, aligner_vocab, blank_id = load_aligner(device)
    print("Model loaded.")

    # our phoneme id -> aligner token id (stress marks stripped; coverage of
    # the full vocab was verified 1:1 before this aligner was adopted)
    def to_aligner_ids(ph_ids):
        out = []
        for pid in ph_ids:
            phone = strip_stress(inv_vocab.get(int(pid), ""))
            aid = aligner_vocab.get(phone)
            if aid is None:
                return None
            out.append(aid)
        return out

    success  = 0
    fallback = 0
    failed   = 0

    for utt_id in tqdm(utt_ids, desc="aligning"):
        wav_path = wav_dir / f"{utt_id}.wav"
        mel_path = mel_dir / f"{utt_id}.npy"
        if not wav_path.exists() or not mel_path.exists():
            failed += 1
            continue

        ph_ids   = np.load(ph_dir / f"{utt_id}.npy")
        n_frames = np.load(mel_path, mmap_mode="r").shape[0]

        durations = None
        aligner_ids = to_aligner_ids(ph_ids)
        if aligner_ids is not None:
            try:
                durations = align_utterance(
                    wav_path, aligner_ids, n_frames, model, device, blank_id)
            except Exception:
                durations = None

        if durations is None:
            # uniform fallback — keeps the pipeline unblocked for the rare
            # utterance the CTC path can't handle
            n_ph = len(ph_ids)
            base = n_frames // n_ph
            rem  = n_frames % n_ph
            durations = [base + (1 if i < rem else 0) for i in range(n_ph)]
            fallback += 1

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
