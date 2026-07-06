"""
Inference: text → wav

Requires:
  - Trained FastSpeech2 checkpoint
  - Trained HiFi-GAN generator checkpoint

Usage:
    python inference.py --text "Hello, this is a test." \
                        --fs2-ckpt checkpoints/fastspeech2/step_300000.pt \
                        --hifi-ckpt checkpoints/hifigan/g_step_500000.pt \
                        --output output.wav

Optional control knobs (all default 1.0):
    --speed   0.8   # slower speech
    --pitch   1.2   # higher pitch
    --energy  0.9   # quieter
"""

import argparse
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from config import audio as acfg, model as mcfg, paths
from model.fastspeech2 import FastSpeech2
from vocoder.generator  import Generator


def load_phoneme_vocab():
    vocab_path = paths.processed_dir / "phoneme_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Phoneme vocab not found at {vocab_path}. "
            "Run preprocess.py first."
        )
    with open(vocab_path) as f:
        return json.load(f)


def text_to_phonemes(text, vocab):
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
        ph_str = backend.phonemize([text], separator=None)[0]
        phonemes = [vocab.get(p, vocab.get("<pad>", 0)) for p in ph_str.split()]
    except ImportError:
        print("WARNING: phonemizer not available, falling back to character-level")
        phonemes = [vocab.get(c, vocab.get("<pad>", 0)) for c in text]
    return phonemes


def mel_to_wav(mel, generator, device):
    """mel: (T, 80) numpy → wav: (T_wav,) numpy"""
    mel_t = torch.from_numpy(mel.T).float().unsqueeze(0).to(device)  # (1, 80, T)
    with torch.no_grad():
        wav = generator(mel_t)  # (1, 1, T_wav)
    return wav.squeeze().cpu().numpy()


def infer(text, fs2_ckpt, hifi_ckpt, output_path,
          speed=1.0, pitch=1.0, energy=1.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = load_phoneme_vocab()
    phonemes = text_to_phonemes(text, vocab)
    if not phonemes:
        raise ValueError("Phonemization produced empty sequence.")
    print(f"Phoneme count: {len(phonemes)}")

    # load FastSpeech2
    fs2 = FastSpeech2().to(device)
    ckpt = torch.load(fs2_ckpt, map_location=device)
    fs2.load_state_dict(ckpt["model"])
    fs2.eval()

    # load HiFi-GAN generator
    hifi = Generator().to(device)
    hifi_ckpt_data = torch.load(hifi_ckpt, map_location=device)
    hifi.load_state_dict(hifi_ckpt_data["generator"])
    hifi.eval()
    hifi.remove_weight_norm()   # fuse weight norm for faster inference

    # run FastSpeech2
    ph_tensor  = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)  # (1, L)
    ph_lens    = torch.tensor([len(phonemes)], dtype=torch.long).to(device)

    with torch.no_grad():
        mel_pred, _, _, _, mel_lens = fs2(
            ph_tensor, ph_lens,
            duration_scale=1.0 / speed,   # slower speed = more frames per phoneme
            pitch_scale=pitch,
            energy_scale=energy,
        )

    mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy()  # (T, 80)
    print(f"Mel frames: {mel.shape[0]}  ({mel.shape[0] * acfg.hop_length / acfg.sample_rate:.2f}s)")

    # denormalize mel from [-1, 1] back to log scale
    mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min

    # run HiFi-GAN
    wav = mel_to_wav(mel, hifi, device)

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
    )
