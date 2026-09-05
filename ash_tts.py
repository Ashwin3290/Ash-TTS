"""
Ash-TTS as a single model.

The pipeline is three stages — FastSpeech2, PostNet, HiFi-GAN generator — that
run strictly in sequence with no branching, so for distribution they can live in
one checkpoint behind one class. This is a bundle, not a weight merge: the
stages have different input and output spaces and their parameters are never
combined, only stored and called together.

Bundling matters for release because the three stages are coupled by more than
order. The vocoder is fine-tuned on the mel distribution this acoustic model and
this PostNet produce, and the phoneme vocabulary must be the exact one whose
index order the embedding table was trained on. Shipping them separately invites
mismatched pairings that load without error and sound wrong.

Build the bundle:
    python ash_tts.py bundle \
        --fs2-ckpt checkpoints/fastspeech2/best_sil.pt \
        --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt \
        --hifi-ckpt checkpoints/hifigan/g_best_new.pt \
        --out checkpoints/release/ash_tts.pt

Use it:
    python ash_tts.py speak --model checkpoints/release/ash_tts.pt \
        --text "Hello, world." --output out.wav

    from ash_tts import AshTTS
    tts = AshTTS.from_pretrained("checkpoints/release/ash_tts.pt")
    wav = tts.synthesize("Hello, world.")
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from config import audio as acfg, model as mcfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from model.decoder import PostNet
from vocoder.generator import Generator, config_from_hcfg
from inference import (text_to_phonemes, denorm_mel, join_wavs,
                       MAX_CHUNK_PHONEMES)

BUNDLE_VERSION = 1


def safe_load(path, map_location="cpu"):
    """torch.load with weights_only=True, which becomes the default in a future
    release. Our checkpoints hold only tensors, dicts and primitives, so this
    succeeds; the fallback covers checkpoints pickled with anything else."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


class AshTTS(nn.Module):
    """Text to waveform. Holds the three stages and the assets they need."""

    def __init__(self, vocab, stats, fs2=None, postnet=None, hifi=None):
        super().__init__()
        self.vocab = vocab
        self.stats = stats
        self.fs2 = fs2 if fs2 is not None else FastSpeech2()
        self.postnet = postnet if postnet is not None else PostNet()
        self.hifi = hifi if hifi is not None else Generator(config_from_hcfg(hcfg))
        self.fs2.variance_adaptor.set_stats(**stats)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, bundle_path, device=None, fuse_weight_norm=True):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = safe_load(bundle_path)

        version = bundle.get("bundle_version")
        if version != BUNDLE_VERSION:
            raise RuntimeError(f"{bundle_path}: bundle version {version}, "
                               f"expected {BUNDLE_VERSION}")

        model = cls(bundle["phoneme_vocab"], bundle["stats"])
        load_fs2_state(model.fs2, bundle["fastspeech2"])
        model.postnet.load_state_dict(bundle["postnet"])
        model.hifi.load_state_dict(bundle["hifigan_generator"])

        model.to(device).eval()
        if fuse_weight_norm:
            # folds weight norm into the conv weights; inference only, and it
            # cannot be undone, so skip it if the generator will be trained again
            model.hifi.remove_weight_norm()
        return model

    @torch.no_grad()
    def synthesize(self, text, speed=1.0, pitch=1.0, energy=1.0, use_postnet=True):
        """text -> waveform, float32 numpy at config.audio.sample_rate.

        Long input is split into chunks of at most MAX_CHUNK_PHONEMES phonemes,
        preferring <sil> positions, and the chunks are crossfaded back together.
        """
        device = self.device
        chunks = text_to_phonemes(text, self.vocab)
        if not chunks:
            raise ValueError("Phonemization produced an empty sequence.")

        wavs = []
        for phonemes in chunks:
            ph = torch.tensor(phonemes, dtype=torch.long, device=device).unsqueeze(0)
            ph_lens = torch.tensor([len(phonemes)], dtype=torch.long, device=device)

            mel_pred, _, _, _, mel_lens = self.fs2(
                ph, ph_lens,
                duration_scale=1.0 / speed,
                pitch_scale=pitch,
                energy_scale=energy,
            )
            if use_postnet:
                mel_pred = mel_pred + self.postnet(mel_pred)

            mel = mel_pred[0, :mel_lens[0].item()]
            # the generator was trained on natural-log mel, not the [-1,1]
            # scale FastSpeech2 predicts
            mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min
            wav = self.hifi(mel.T.unsqueeze(0)).squeeze().float().cpu().numpy()
            wavs.append(wav)

        return join_wavs(wavs, acfg.sample_rate)

    def save_wav(self, text, path, **kwargs):
        wav = self.synthesize(text, **kwargs)
        sf.write(str(path), wav, acfg.sample_rate)
        return len(wav) / acfg.sample_rate


def _weights_only(ckpt, key):
    """Pull the weight dict out of a training checkpoint, dropping optimizer
    state — it is useless for inference and larger than the weights."""
    inner = ckpt.get(key)
    if inner is None:
        raise RuntimeError(f"checkpoint has no '{key}' (keys: {list(ckpt)})")
    return inner, ckpt.get("step")


def cmd_bundle(args):
    fs2_ckpt = safe_load(args.fs2_ckpt)
    post_ckpt = safe_load(args.postnet_ckpt)
    hifi_ckpt = safe_load(args.hifi_ckpt)

    fs2_sd, fs2_step = _weights_only(fs2_ckpt, "model")
    post_sd, post_step = _weights_only(post_ckpt, "postnet")
    hifi_sd, hifi_step = _weights_only(hifi_ckpt, "generator")

    # the in-model PostNet is gone; strip its keys so old checkpoints bundle cleanly
    fs2_sd = {k: v for k, v in fs2_sd.items() if not k.startswith("postnet.")}

    vocab_path = Path(args.vocab)
    stats_path = Path(args.stats)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    # the embedding table must be able to index every phoneme in the vocab, or
    # inference silently produces wrong phones
    emb_key = next(k for k in fs2_sd
                   if k.endswith("embedding.weight")
                   and "pitch" not in k and "energy" not in k)
    n_rows = fs2_sd[emb_key].shape[0]
    if len(vocab) > n_rows:
        raise RuntimeError(f"vocab has {len(vocab)} entries but the embedding "
                           f"table has {n_rows} rows")
    norms = fs2_sd[emb_key].norm(dim=1)
    baseline = norms[len(vocab):].median().item() if n_rows > len(vocab) else 0.0
    trained = int((norms[:len(vocab)] > 1.3 * baseline).sum()) if baseline else len(vocab)
    print(f"vocab {len(vocab)} entries, embedding {n_rows} rows, "
          f"{trained} rows above the untrained baseline ({baseline:.2f})")

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "fastspeech2": fs2_sd,
        "postnet": post_sd,
        "hifigan_generator": hifi_sd,
        "phoneme_vocab": vocab,
        "stats": stats,
        "config": {
            "sample_rate": acfg.sample_rate,
            "n_fft": acfg.n_fft,
            "hop_length": acfg.hop_length,
            "win_length": acfg.win_length,
            "n_mels": acfg.n_mels,
            "fmin": acfg.fmin,
            "fmax": acfg.fmax,
            "mel_min": acfg.mel_min,
            "mel_max": acfg.mel_max,
            "d_model": mcfg.d_model,
            "n_phonemes": mcfg.n_phonemes,
            "max_chunk_phonemes": MAX_CHUNK_PHONEMES,
        },
        "provenance": {
            "fastspeech2": {"file": str(args.fs2_ckpt), "step": fs2_step},
            "postnet": {"file": str(args.postnet_ckpt), "step": post_step},
            "hifigan_generator": {"file": str(args.hifi_ckpt), "step": hifi_step},
            "vocab": str(vocab_path),
            "stats": str(stats_path),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    for name, info in bundle["provenance"].items():
        if isinstance(info, dict):
            print(f"  {name:<20} {info['file']}  step {info['step']}")

    if not args.no_verify:
        print("\nVerifying by loading the bundle and synthesizing:")
        tts = AshTTS.from_pretrained(out)
        n = sum(p.numel() for p in tts.parameters())
        print(f"  total parameters: {n:,}")
        dur = tts.save_wav("The quick brown fox jumps over the lazy dog, and the "
                           "north wind blew cold.", args.verify_wav)
        print(f"  wrote {args.verify_wav} ({dur:.2f}s) — listen before publishing")


def cmd_speak(args):
    tts = AshTTS.from_pretrained(args.model)
    dur = tts.save_wav(args.text, args.output, speed=args.speed,
                       pitch=args.pitch, energy=args.energy,
                       use_postnet=not args.no_postnet)
    print(f"Wrote {args.output} ({dur:.2f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bundle", help="Combine the three stages into one checkpoint")
    b.add_argument("--fs2-ckpt", required=True)
    b.add_argument("--postnet-ckpt", required=True)
    b.add_argument("--hifi-ckpt", required=True)
    b.add_argument("--vocab", default=str(paths.assets_dir / "phoneme_vocab.json"))
    b.add_argument("--stats", default=str(paths.assets_dir / "stats.json"))
    b.add_argument("--out", default="checkpoints/release/ash_tts.pt")
    b.add_argument("--verify-wav", default="bundle_check.wav")
    b.add_argument("--no-verify", action="store_true")

    s = sub.add_parser("speak", help="Synthesize from a bundle")
    s.add_argument("--model", default="checkpoints/release/ash_tts.pt")
    s.add_argument("--text", required=True)
    s.add_argument("--output", default="output.wav")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--pitch", type=float, default=1.0)
    s.add_argument("--energy", type=float, default=1.0)
    s.add_argument("--no-postnet", action="store_true")

    args = ap.parse_args()
    {"bundle": cmd_bundle, "speak": cmd_speak}[args.cmd](args)