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

<sil> tokens are re-derived from acoustic energy in the inter-phone gaps
(head, mid-utterance, and tail), not from a TextGrid. The previous
TextGrid + exact-espeak-count gate (regenerate_phonemes.py) silently
dropped any utterance where espeak's phoneme count didn't match the
TextGrid's speech-interval count exactly — on LJSpeech that kept only
~6% of the data, and training on that starved, overfit set is what
produced the monotonically-rising val loss this rewrite is fixing.
Any <sil> a prior TextGrid pass already wrote into phoneme/*.npy is
stripped and re-derived here instead of trusted.

Reads phoneme ids and energy straight from data/processed/{phoneme,energy}/*.npy
(no phonemizer/espeak needed at alignment time) and overwrites both the
phoneme and duration files for each utterance it processes.

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

MIN_SIL_W2V     = 6      # gap must be >= 6 emission frames (120ms) to become <sil>
SIL_ENERGY_FRAC = 0.05   # and mean RMS over the gap < 5% of the utterance's max RMS


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


def align_utterance(wav_path, ph_ids, aligner_ids, energy, n_mel_frames,
                     model, device, blank_id, sil_id):
    """Force-align ph_ids (no <sil>) against the wav2vec2 CTC emissions, then
    re-derive <sil> tokens from acoustic energy in the inter-phone gaps —
    instead of trusting a TextGrid's silence tier, which is what the old
    exact-count gate did and which dropped ~94% of utterances that didn't
    match espeak's phoneme count exactly.

    A gap (head, inter-phone, or tail) becomes <sil> only if it's both long
    enough (MIN_SIL_W2V emission frames) and quiet enough (mean RMS under
    SIL_ENERGY_FRAC of the utterance's peak) — this keeps peaky-CTC blank
    spans from being mislabelled as pauses.

    Returns (phoneme_ids_with_sil, durations) or None.
    """
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

    n_emission = emissions.size(1)
    energy_max = energy.max() if len(energy) else 0.0

    def gap_is_silent(w2v_start, w2v_end):
        if w2v_end - w2v_start < MIN_SIL_W2V or energy_max <= 0:
            return False
        a = int(w2v_start * W2V_FRAME_SEC / MEL_FRAME_SEC)
        b = max(a + 1, int(w2v_end * W2V_FRAME_SEC / MEL_FRAME_SEC))
        seg = energy[a:b]
        if len(seg) == 0:
            return False
        return seg.mean() < SIL_ENERGY_FRAC * energy_max

    # out_ids and frame_bounds grow in lockstep: frame_bounds always has
    # exactly len(out_ids) + 1 entries, i.e. frame_bounds[k] is the start
    # boundary of out_ids[k] and frame_bounds[k+1] is its end.
    out_ids, frame_bounds = [], [0]

    if gap_is_silent(0, spans[0].start):
        out_ids.append(sil_id)
        frame_bounds.append(spans[0].start)

    for i, (pid, span) in enumerate(zip(ph_ids, spans)):
        out_ids.append(int(pid))
        gap_start = span.end + 1
        gap_end   = spans[i + 1].start if i + 1 < len(spans) else n_emission
        if gap_start < gap_end and gap_is_silent(gap_start, gap_end):
            frame_bounds.append(gap_start)   # close this phone at the gap
            out_ids.append(sil_id)
            frame_bounds.append(gap_end)     # <sil> spans the rest of the gap
        else:
            frame_bounds.append(gap_end)     # phone absorbs the (non-silent) gap

    frame_bounds[-1] = n_emission

    boundaries_sec = [b * W2V_FRAME_SEC for b in frame_bounds]
    durations = distribute_to_frames(boundaries_sec, n_mel_frames)
    if durations is None or len(durations) != len(out_ids):
        return None
    return np.array(out_ids, dtype=np.int32), durations


def run_alignment(data_root, processed_dir, overwrite=False, device=None):
    data_root     = Path(data_root)
    processed_dir = Path(processed_dir)
    wav_dir       = data_root / "wavs"
    dur_dir       = processed_dir / "duration"
    mel_dir       = processed_dir / "mel"
    ph_dir        = processed_dir / "phoneme"
    energy_dir    = processed_dir / "energy"
    dur_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not ph_dir.exists():
        raise RuntimeError("Run preprocess.py first — phoneme files not found.")

    with open(processed_dir / "phoneme_vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    sil_id = vocab["<sil>"]

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
        # re-derive <sil> from acoustic energy below, not from whatever a
        # prior TextGrid-based pass may have already baked into this file
        ph_ids   = ph_ids[ph_ids != sil_id]
        n_frames = np.load(mel_path, mmap_mode="r").shape[0]
        energy_path = energy_dir / f"{utt_id}.npy"
        energy = np.load(energy_path) if energy_path.exists() else np.zeros(0, dtype=np.float32)

        if len(ph_ids) == 0:
            failed += 1
            continue

        result = None
        aligner_ids = to_aligner_ids(ph_ids)
        if aligner_ids is not None:
            try:
                result = align_utterance(
                    wav_path, ph_ids, aligner_ids, energy, n_frames,
                    model, device, blank_id, sil_id)
            except Exception:
                result = None

        if result is None:
            # uniform fallback — keeps the pipeline unblocked for the rare
            # utterance the CTC path can't handle; no <sil> in this case
            n_ph = len(ph_ids)
            base = n_frames // n_ph
            rem  = n_frames % n_ph
            durations = [base + (1 if i < rem else 0) for i in range(n_ph)]
            out_ids = ph_ids.astype(np.int32)
            fallback += 1
        else:
            out_ids, durations = result

        np.save(ph_dir / f"{utt_id}.npy", out_ids)
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
