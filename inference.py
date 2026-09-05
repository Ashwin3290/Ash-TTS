"""
Inference: text → wav

Requires:
  - Trained FastSpeech2 checkpoint
  - Trained HiFi-GAN generator checkpoint

Usage:
    # a checkpoint produced by train_hifigan.py (from-scratch or fine-tuned from
    # pretrained weights) — trained on our normalised [-1,1] mel, so no denorm
    python inference.py --text "Hello, this is a test." \
                        --fs2-ckpt checkpoints/fastspeech2/step_300000.pt \
                        --hifi-ckpt checkpoints/hifigan/g_step_500000.pt \
                        --output output.wav

    # the RAW, untouched official checkpoint (generator_v1 / universal_v1),
    # never fine-tuned on our data — still on its own natural-log mel scale,
    # so this path denormalises before vocoding
    python inference.py --text "Hello, this is a test." \
                        --fs2-ckpt checkpoints/fastspeech2/step_300000.pt \
                        --hifi-ckpt generator_v1 --hifi-config config.json \
                        --raw-official-hifigan --output output.wav

Optional control knobs (all default 1.0):
    --speed   0.8   # slower speech
    --pitch   1.2   # higher pitch
    --energy  0.9   # quieter
"""

import argparse
import json
import re
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from config import audio as acfg, model as mcfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from vocoder.generator import Generator, config_from_hcfg

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
CLAUSE_BREAK = re.compile(r'[,;:]')


def load_phoneme_vocab():
    vocab_path = paths.processed_dir / "phoneme_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Phoneme vocab not found at {vocab_path}. "
            "Run preprocess.py first."
        )
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def chunk_by_sil(ids, sil, max_len=150):
    """Greedily split a phoneme-id list into chunks <= max_len, cutting only
    at <sil> tokens. Each chunk keeps its trailing <sil>."""
    if len(ids) <= max_len:
        return [ids]
    chunks, cur = [], []
    clauses, buf = [], []
    for i in ids:
        buf.append(i)
        if i == sil:
            clauses.append(buf); buf = []
    if buf:
        clauses.append(buf + [sil])
    for cl in clauses:
        if cur and len(cur) + len(cl) > max_len:
            chunks.append(cur); cur = []
        cur += cl
    if cur:
        chunks.append(cur)
    return chunks


def text_to_phonemes(text, vocab):
    """Returns a list of phoneme-id lists, one per sentence.
    <sil> is inserted at commas/semicolons/colons and at each sentence end,
    matching the acoustic <sil> the model saw in training."""
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
    sep = Separator(phone=' ', word='| ', syllable='')
    sil = vocab["<sil>"]
    pad = vocab.get("<pad>", 0)

    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    out = []
    for sent in sentences:
        clauses = [c.strip() for c in CLAUSE_BREAK.split(sent) if c.strip()]
        ph_strs = backend.phonemize(clauses, separator=sep)
        ids = []
        for i, ph_str in enumerate(ph_strs):
            ids += [vocab.get(p, pad) for p in ph_str.replace('|', '').split() if p.strip()]
            if i < len(ph_strs) - 1:
                ids.append(sil)
        ids.append(sil)
        if ids:
            out.extend(chunk_by_sil(ids, sil))
    return out


def mel_to_wav(mel, generator, device):
    """mel: (T, 80) numpy → wav: (T_wav,) numpy"""
    mel_t = torch.from_numpy(mel.T).float().unsqueeze(0).to(device)  # (1, 80, T)
    with torch.no_grad():
        wav = generator(mel_t)  # (1, 1, T_wav)
    return wav.squeeze().cpu().numpy()


def infer(text, fs2_ckpt, hifi_ckpt, output_path,
          speed=1.0, pitch=1.0, energy=1.0,
          raw_official_hifigan=False, hifi_config=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = load_phoneme_vocab()
    sentences = text_to_phonemes(text, vocab)
    if not sentences:
        raise ValueError("Phonemization produced empty sequence.")
    print(f"{len(sentences)} sentence(s), {sum(len(s) for s in sentences)} phonemes")

    # load FastSpeech2
    fs2 = FastSpeech2().to(device)
    ckpt = torch.load(fs2_ckpt, map_location=device)
    load_fs2_state(fs2, ckpt["model"])
    fs2.eval()

    # load HiFi-GAN generator
    if raw_official_hifigan:
        if not hifi_config:
            raise ValueError("--hifi-config is required with --raw-official-hifigan")
        from vocoder.generator import load_pretrained_generator
        hifi = load_pretrained_generator(hifi_ckpt, hifi_config, device)
    else:
        hifi = Generator(config_from_hcfg(hcfg)).to(device)
        hifi_ckpt_data = torch.load(hifi_ckpt, map_location=device)
        hifi.load_state_dict(hifi_ckpt_data["generator"])
        hifi.eval()
        hifi.remove_weight_norm()   # fuse weight norm for faster inference

    # run FastSpeech2 + HiFi-GAN, one sentence at a time
    wavs = []
    for phonemes in sentences:
        ph_tensor = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)  # (1, L)
        ph_lens   = torch.tensor([len(phonemes)], dtype=torch.long).to(device)

        with torch.no_grad():
            _, mel_pred, _, _, _, mel_lens = fs2(   # mel_pred = PostNet-refined mel_after
                ph_tensor, ph_lens,
                duration_scale=1.0 / speed,   # slower speed = more frames per phoneme
                pitch_scale=pitch,
                energy_scale=energy,
            )

        mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy()  # (T, 80)

        if raw_official_hifigan:
            # the untouched official checkpoint expects raw natural-log mel, not
            # our [-1,1] normalised training scale — denormalise before vocoding
            mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min
        # else: any checkpoint from train_hifigan.py (from-scratch or fine-tuned)
        # was trained on our normalised mels, so no denorm needed here.

        wavs.append(mel_to_wav(mel, hifi, device))

    wav = np.concatenate(wavs)
    print(f"Total: {len(wav) / acfg.sample_rate:.2f}s")

    # save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, acfg.sample_rate)
    print(f"Saved: {output_path}  ({len(wav) / acfg.sample_rate:.2f}s at {acfg.sample_rate}Hz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",      required=True)
    parser.add_argument("--fs2-ckpt",  required=True)
    parser.add_argument("--hifi-ckpt", required=True)
    parser.add_argument("--hifi-config", default=None,
                        help="Required with --raw-official-hifigan (official config.json)")
    parser.add_argument("--raw-official-hifigan", action="store_true",
                        help="Load the untouched official jik876/hifi-gan checkpoint "
                             "(not fine-tuned) instead of a train_hifigan.py checkpoint")
    parser.add_argument("--output",    default="output.wav")
    parser.add_argument("--speed",     type=float, default=1.0)
    parser.add_argument("--pitch",     type=float, default=1.0)
    parser.add_argument("--energy",    type=float, default=1.0)
    args = parser.parse_args()

    infer(
        text=args.text,
        fs2_ckpt=args.fs2_ckpt,
        hifi_ckpt=args.hifi_ckpt,
        output_path=args.output,
        speed=args.speed,
        pitch=args.pitch,
        energy=args.energy,
        raw_official_hifigan=args.raw_official_hifigan,
        hifi_config=args.hifi_config,
    )
