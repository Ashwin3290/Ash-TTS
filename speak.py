"""
Text in, speech out — the actual use case.

Usage:
    python speak.py "Hello there, this is my own text to speech model."
    python speak.py "..." --output hello.wav --speed 1.2
    python speak.py                    # interactive: type sentences, get wavs

Uses the best checkpoints (checkpoints/fastspeech2/best.pt +
checkpoints/hifigan/g_best.pt), auto-downloading them from HuggingFace on
first run if they aren't present locally.

Control knobs (all default 1.0):
    --speed   1.2   # faster speech
    --pitch   1.1   # higher pitch
    --energy  0.9   # quieter
"""

import argparse
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

from config import audio as acfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from vocoder.generator import Generator, config_from_hcfg
from inference import text_to_phonemes, load_phoneme_vocab

FS2_CKPT  = Path("checkpoints/fastspeech2/best.pt")
HIFI_CKPT = Path("checkpoints/hifigan/g_best.pt")
FS2_REPO  = "Ashwin-C9/tts-fastspeech2-ckpt"
HIFI_REPO = "Ashwin-C9/tts-hifigan-ckpt"


def ensure_checkpoints():
    for local, repo, fname in [(FS2_CKPT, FS2_REPO, "best.pt"),
                                (HIFI_CKPT, HIFI_REPO, "g_best.pt")]:
        if not local.exists():
            print(f"Downloading {fname} from {repo}...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo, fname, repo_type="model")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(Path(downloaded).read_bytes())


def load_models(device):
    with open(paths.processed_dir / "stats.json") as f:
        stats = json.load(f)
    fs2 = FastSpeech2().to(device)
    fs2.variance_adaptor.set_stats(**stats)
    ckpt = torch.load(FS2_CKPT, map_location=device)
    load_fs2_state(fs2, ckpt["model"])
    fs2.eval()
    print(f"FastSpeech2: step {ckpt.get('step', '?')}")

    hifi = Generator(config_from_hcfg(hcfg)).to(device)
    hifi_ckpt = torch.load(HIFI_CKPT, map_location=device)
    hifi.load_state_dict(hifi_ckpt["generator"])
    hifi.eval()
    hifi.remove_weight_norm()
    print(f"HiFi-GAN:    step {hifi_ckpt.get('step', '?')}")
    return fs2, hifi


def synthesize(text, fs2, hifi, vocab, device, speed=1.0, pitch=1.0, energy=1.0):
    phonemes = text_to_phonemes(text, vocab)
    if not phonemes:
        raise ValueError("Phonemization produced an empty sequence.")

    ph = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)
    ph_lens = torch.tensor([len(phonemes)], dtype=torch.long).to(device)

    with torch.no_grad():
        _, mel_pred, _, _, _, mel_lens = fs2(   # mel_pred = PostNet-refined mel_after
            ph, ph_lens,
            duration_scale=1.0 / speed,
            pitch_scale=pitch,
            energy_scale=energy,
        )
        mel = mel_pred[:, :mel_lens[0].item()]           # (1, T, 80) normalised
        # -> natural-log scale, the convention the vocoder is trained on
        mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min
        wav = hifi(mel.transpose(1, 2)).squeeze().cpu().numpy()
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=None,
                        help="Sentence to speak. Omit for interactive mode.")
    parser.add_argument("--output", default=None,
                        help="Output wav path (default: speech.wav / speech_N.wav)")
    parser.add_argument("--speed",  type=float, default=1.0)
    parser.add_argument("--pitch",  type=float, default=1.0)
    parser.add_argument("--energy", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_checkpoints()
    vocab = load_phoneme_vocab()
    fs2, hifi = load_models(device)

    def speak_to_file(text, out_path):
        wav = synthesize(text, fs2, hifi, vocab, device,
                         speed=args.speed, pitch=args.pitch, energy=args.energy)
        sf.write(str(out_path), wav, acfg.sample_rate)
        print(f"  -> {out_path}  ({len(wav) / acfg.sample_rate:.2f}s)")

    if args.text:
        speak_to_file(args.text, args.output or "speech.wav")
        return

    # interactive mode
    print("\nType a sentence and press Enter (empty line to quit).")
    n = 1
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        out = args.output or f"speech_{n}.wav"
        try:
            speak_to_file(text, out)
        except Exception as e:
            print(f"  error: {e}")
            continue
        n += 1


if __name__ == "__main__":
    main()
