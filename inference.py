"""
Inference: text → wav

Requires:
  - Trained FastSpeech2 checkpoint
  - Trained HiFi-GAN generator checkpoint

Usage:
    # a checkpoint produced by train_hifigan.py (from-scratch or fine-tuned from
    # pretrained weights)
    python inference.py --text "Hello, this is a test." \
                        --fs2-ckpt checkpoints/fastspeech2/best.pt \
                        --hifi-ckpt checkpoints/hifigan/g_best.pt \
                        --output output.wav

    # the RAW, untouched official checkpoint (generator_v1 / universal_v1)
    python inference.py --text "Hello, this is a test." \
                        --fs2-ckpt checkpoints/fastspeech2/best.pt \
                        --hifi-ckpt generator_v1 --hifi-config config.json \
                        --raw-official-hifigan --output output.wav

    # with a standalone PostNet from train_postnet.py applied to the mel
    python inference.py --text "..." --fs2-ckpt ... --hifi-ckpt ... \
                        --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt

Optional control knobs (all default 1.0):
    --speed   0.8   # slower speech
    --pitch   1.2   # higher pitch
    --energy  0.9   # quieter
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

from config import audio as acfg, model as mcfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from model.decoder import PostNet
from vocoder.generator import Generator, config_from_hcfg

# split on sentence terminators, and on clause punctuation within a sentence
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BREAK = re.compile(r"[,;:]")

# LJSpeech phoneme-sequence lengths: p50 70, p90 98, p99 113, max 132. Anything
# past ~100 is territory the duration predictor and decoder attention barely saw,
# and the tail of the chunk degrades. Cap at the p90.
MAX_CHUNK_PHONEMES = 100

# crossfade applied when joining chunks, in seconds
CROSSFADE_SEC = 0.005


def _resolve_asset(filename):
    """assets/ (checked into git, ships with the repo) first, falling back to
    data/processed/ (produced locally by preprocess.py) — so inference works
    on a fresh clone with just the checkpoints, no dataset required."""
    asset_path = paths.assets_dir / filename
    if asset_path.exists():
        return asset_path
    processed_path = paths.processed_dir / filename
    if processed_path.exists():
        return processed_path
    raise FileNotFoundError(
        f"{filename} not found in {paths.assets_dir} or {paths.processed_dir}. "
        "Run preprocess.py first."
    )


def load_phoneme_vocab():
    with open(_resolve_asset("phoneme_vocab.json"), encoding="utf-8") as f:
        return json.load(f)


def load_stats():
    with open(_resolve_asset("stats.json"), encoding="utf-8") as f:
        return json.load(f)


def denorm_mel(mel):
    """Undo preprocess.py's [-1,1] normalisation -> natural-log mel scale.

    Every vocoder path needs this. train_hifigan.py's WavMelDataset denormalises
    before training, so a fine-tuned generator expects natural-log mel just like
    the official pretrained one does.
    """
    return (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min


def pack_chunks(units, max_len=MAX_CHUNK_PHONEMES):
    """units: list of id-lists, one per word, some ending in <sil>.

    Greedily packs units into chunks of at most max_len ids. Because a unit is a
    word, a chunk can break mid-clause when a clause is too long to split at a
    pause — packing whole clauses only would let an unpunctuated run-on produce
    a single oversized chunk regardless of max_len.
    """
    chunks, cur = [], []
    for unit in units:
        if cur and len(cur) + len(unit) > max_len:
            chunks.append(cur)
            cur = []
        cur += unit
    if cur:
        chunks.append(cur)
    return chunks


def text_to_phonemes(text, vocab):
    """text -> list of phoneme-id chunks.

    <sil> is inserted at clause punctuation and at the end of every sentence,
    matching the acoustically-derived <sil> tokens the model was trained on.
    Without this, punctuation has no effect on the output at all.
    """
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
    # must match preprocess.py exactly: phone-level tokens, '|' marks word ends
    sep = Separator(phone=" ", word="| ", syllable="")

    sil_id = vocab["<sil>"]
    pad_id = vocab.get("<pad>", 0)

    chunks = []
    for sentence in (s.strip() for s in SENTENCE_END.split(text)):
        if not sentence:
            continue
        clauses = [c.strip() for c in CLAUSE_BREAK.split(sentence) if c.strip()]
        if not clauses:
            continue
        ph_strs = backend.phonemize(clauses, separator=sep)

        units = []
        for ph_str in ph_strs:
            for word in ph_str.split("|"):
                ids = [vocab.get(p, pad_id) for p in word.split() if p.strip()]
                if ids:
                    units.append(ids)
            if units:   # <sil> at every clause break and at the sentence end
                units[-1] = units[-1] + [sil_id]
        chunks.extend(pack_chunks(units))

    return chunks


def join_wavs(wavs, sample_rate):
    """Concatenate chunk waveforms with a short crossfade, so a chunk boundary
    that fell mid-clause does not produce an audible click."""
    if not wavs:
        return np.zeros(0, dtype=np.float32)
    if len(wavs) == 1:
        return wavs[0]

    xf = int(CROSSFADE_SEC * sample_rate)
    out = wavs[0]
    for nxt in wavs[1:]:
        n = min(xf, len(out), len(nxt))
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out = np.concatenate([out[:-n],
                                  out[-n:] * (1 - ramp) + nxt[:n] * ramp,
                                  nxt[n:]])
        else:
            out = np.concatenate([out, nxt])
    return out


def load_postnet(ckpt_path, device):
    net = PostNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["postnet"] if "postnet" in state else state)
    net.eval()
    print(f"PostNet: {ckpt_path} (step {state.get('step', '?')})")
    return net


def load_vocoder(hifi_ckpt, device, raw_official=False, hifi_config=None):
    if raw_official:
        if not hifi_config:
            raise ValueError("--hifi-config is required with --raw-official-hifigan")
        from vocoder.generator import load_pretrained_generator
        return load_pretrained_generator(hifi_ckpt, hifi_config, device)

    hifi = Generator(config_from_hcfg(hcfg)).to(device)
    state = torch.load(hifi_ckpt, map_location=device)
    hifi.load_state_dict(state["generator"])
    hifi.eval()
    hifi.remove_weight_norm()   # fuse weight norm for faster inference
    return hifi


def mel_to_wav(mel, generator, device):
    """mel: (T, 80) numpy on the natural-log scale → wav: (T_wav,) numpy"""
    mel_t = torch.from_numpy(mel.T).float().unsqueeze(0).to(device)
    with torch.no_grad():
        wav = generator(mel_t)
    return wav.squeeze().cpu().numpy()


def infer(text, fs2_ckpt, hifi_ckpt, output_path,
          speed=1.0, pitch=1.0, energy=1.0,
          raw_official_hifigan=False, hifi_config=None, postnet_ckpt=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = load_phoneme_vocab()
    chunks = text_to_phonemes(text, vocab)
    if not chunks:
        raise ValueError("Phonemization produced empty sequence.")
    print(f"{len(chunks)} chunk(s), {sum(len(c) for c in chunks)} phonemes total")
    print(f"chunk lengths: {[len(c) for c in chunks]}")

    fs2 = FastSpeech2().to(device)
    ckpt = torch.load(fs2_ckpt, map_location=device)
    fs2.variance_adaptor.set_stats(**load_stats())
    load_fs2_state(fs2, ckpt["model"])
    fs2.eval()
    print(f"FastSpeech2: {fs2_ckpt} (step {ckpt.get('step', '?')})")

    postnet = load_postnet(postnet_ckpt, device) if postnet_ckpt else None
    hifi = load_vocoder(hifi_ckpt, device, raw_official_hifigan, hifi_config)

    wavs = []
    for phonemes in chunks:
        ph_tensor = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)
        ph_lens = torch.tensor([len(phonemes)], dtype=torch.long).to(device)

        with torch.no_grad():
            mel_pred, _, _, _, mel_lens = fs2(
                ph_tensor, ph_lens,
                duration_scale=1.0 / speed,   # slower speed = more frames per phoneme
                pitch_scale=pitch,
                energy_scale=energy,
            )
            if postnet is not None:
                mel_pred = mel_pred + postnet(mel_pred)

        mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy()   # (T, 80), normalised
        wavs.append(mel_to_wav(denorm_mel(mel), hifi, device))

    wav = join_wavs(wavs, acfg.sample_rate)
    sf.write(output_path, wav, acfg.sample_rate)
    print(f"Saved {output_path}  ({len(wav) / acfg.sample_rate:.2f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--fs2-ckpt", required=True)
    ap.add_argument("--hifi-ckpt", required=True)
    ap.add_argument("--hifi-config", default=None)
    ap.add_argument("--raw-official-hifigan", action="store_true")
    ap.add_argument("--postnet-ckpt", default=None,
                    help="Standalone PostNet from train_postnet.py")
    ap.add_argument("--output", default="output.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, default=1.0)
    ap.add_argument("--energy", type=float, default=1.0)
    args = ap.parse_args()

    infer(args.text, args.fs2_ckpt, args.hifi_ckpt, args.output,
          speed=args.speed, pitch=args.pitch, energy=args.energy,
          raw_official_hifigan=args.raw_official_hifigan,
          hifi_config=args.hifi_config, postnet_ckpt=args.postnet_ckpt)