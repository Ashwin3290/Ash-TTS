"""
Sanity-check a FastSpeech2 checkpoint end to end: pick a validation utterance
(or take custom text), run real inference (not teacher-forced), plot predicted
vs ground truth mel, and vocode to a listenable wav.

Usage:
    python test_inference.py                              # random val utterance, HF best.pt
    python test_inference.py --utt-id LJ001-0001
    python test_inference.py --text "Custom sentence to synthesize."
    python test_inference.py --fs2-ckpt checkpoints/fastspeech2/best_sil.pt \
                             --hifi-ckpt checkpoints/hifigan/g_sil2_25k.pt
    python test_inference.py --text "..." --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt

With no --fs2-ckpt the latest best.pt is pulled from HuggingFace.
With no --hifi-ckpt the raw official pretrained generator_v1 is used.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download

from config import audio as acfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from inference import (text_to_phonemes, load_phoneme_vocab, load_stats, denorm_mel,
                       load_postnet, load_vocoder, mel_to_wav)

HF_CKPT_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"
HIFIGAN_DIR = Path("GAN model")   # generator_v1 + config.json


def pull_best_checkpoint():
    ckpt_dir = paths.fastspeech_ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pulling best.pt from {HF_CKPT_REPO}...")
    # force_download: never trust the local HF cache here — a stale cached
    # best.pt once made a barely-trained model look like a broken one
    downloaded = hf_hub_download(HF_CKPT_REPO, "best.pt", repo_type="model",
                                 force_download=True)
    dest = ckpt_dir / "best.pt"
    dest.write_bytes(Path(downloaded).read_bytes())
    print(f"Saved to {dest}")
    return dest


def pick_utterance(utt_id=None):
    with open(paths.processed_dir / "val_manifest.json") as f:
        manifest = json.load(f)
    if utt_id is None:
        return random.choice(manifest)
    item = next((m for m in manifest if m["id"] == utt_id), None)
    if item is None:
        raise ValueError(f"{utt_id} not found in val_manifest.json")
    return item


def official_generator_paths():
    for base in (HIFIGAN_DIR, Path("pretrained_hifigan")):
        gen, cfg = base / "generator_v1", base / "config.json"
        if gen.exists() and cfg.exists():
            return str(gen), str(cfg)
    raise FileNotFoundError(
        "generator_v1 + config.json not found in 'GAN model/' or 'pretrained_hifigan/'"
    )


def main(utt_id, text, output_dir, fs2_ckpt=None, hifi_ckpt=None, postnet_ckpt=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(fs2_ckpt) if fs2_ckpt else pull_best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"FastSpeech2: {ckpt_path} (step {ckpt.get('step', '?')})")

    model = FastSpeech2().to(device)
    model.variance_adaptor.set_stats(**load_stats())
    load_fs2_state(model, ckpt["model"])
    model.eval()

    gt_mel = None
    gt_wav_path = None
    if text is None:
        item = pick_utterance(utt_id)
        utt_id = item["id"]
        text = item["text"]
        print(f"Utterance: {utt_id}")
        print(f"Text: {text}")
        gt_mel = np.load(paths.processed_dir / "mel" / f"{utt_id}.npy")
        gt_wav_path = paths.data_root / "wavs" / f"{utt_id}.wav"
    else:
        print(f"Custom text: {text}")

    vocab = load_phoneme_vocab()
    chunks = text_to_phonemes(text, vocab)
    if not chunks:
        raise ValueError("Phonemization produced empty sequence.")
    print(f"{len(chunks)} chunk(s), {sum(len(c) for c in chunks)} phonemes total")

    postnet = load_postnet(postnet_ckpt, device) if postnet_ckpt else None

    mels = []
    for phonemes in chunks:
        ph_tensor = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)
        ph_lens = torch.tensor([len(phonemes)], dtype=torch.long).to(device)
        with torch.no_grad():
            mel_pred, _, _, _, mel_lens = model(ph_tensor, ph_lens)
            if postnet is not None:
                mel_pred = mel_pred + postnet(mel_pred)
        mels.append(mel_pred[0, :mel_lens[0].item()].cpu().numpy())

    pred_mel = np.concatenate(mels, axis=0)   # (T, 80), normalised
    print(f"Predicted mel frames: {pred_mel.shape[0]} "
          f"({pred_mel.shape[0] * acfg.hop_length / acfg.sample_rate:.2f}s)")

    # --- plot predicted vs ground truth (if we have one) ---
    if gt_mel is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].imshow(gt_mel.T, aspect="auto", origin="lower", interpolation="none")
        axes[0].set_title(f"Ground truth mel — {utt_id} ({gt_mel.shape[0]} frames)")
        axes[1].imshow(pred_mel.T, aspect="auto", origin="lower", interpolation="none")
        axes[1].set_title(f"Predicted mel — step {ckpt.get('step', '?')} "
                          f"({pred_mel.shape[0]} frames)")
        for ax in axes:
            ax.set_ylabel("mel bin")
        axes[1].set_xlabel("frame")
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(pred_mel.T, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(f"Predicted mel — step {ckpt.get('step', '?')}")
        ax.set_xlabel("frame")
        ax.set_ylabel("mel bin")
    fig.tight_layout()
    plot_path = output_dir / "mel_comparison.png"
    fig.savefig(plot_path, dpi=120)
    print(f"Saved plot: {plot_path}")

    # --- vocode ---
    # both paths take the raw natural-log mel: train_hifigan.py's WavMelDataset
    # denormalises before training, so a fine-tuned generator expects the same
    # scale as the official pretrained one
    if hifi_ckpt:
        hifi = load_vocoder(hifi_ckpt, device)
        print(f"Vocoder: fine-tuned checkpoint {hifi_ckpt}")
    else:
        gen_path, cfg_path = official_generator_paths()
        hifi = load_vocoder(gen_path, device, raw_official=True, hifi_config=cfg_path)
        print("Vocoder: raw official generator_v1")

    wav = mel_to_wav(denorm_mel(pred_mel), hifi, device)
    pred_wav_path = output_dir / "predicted.wav"
    sf.write(str(pred_wav_path), wav, acfg.sample_rate)
    print(f"Saved predicted audio: {pred_wav_path} ({len(wav) / acfg.sample_rate:.2f}s)")

    if gt_wav_path is not None and gt_wav_path.exists():
        gt_wav, sr = sf.read(str(gt_wav_path))
        if sr != acfg.sample_rate:
            gt_wav = librosa.resample(gt_wav, orig_sr=sr, target_sr=acfg.sample_rate)
        gt_wav_out = output_dir / "ground_truth.wav"
        sf.write(str(gt_wav_out), gt_wav, acfg.sample_rate)
        print(f"Saved ground truth audio: {gt_wav_out} "
              f"({len(gt_wav) / acfg.sample_rate:.2f}s)")

    print(f"\nDone. Check {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt-id", default=None, help="Specific val_manifest utterance id")
    parser.add_argument("--text", default=None, help="Custom text instead of a val utterance")
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument("--fs2-ckpt", default=None,
                        help="Local FastSpeech2 checkpoint. Default: pull best.pt from HF.")
    parser.add_argument("--hifi-ckpt", default=None,
                        help="Fine-tuned train_hifigan.py generator. Default: official generator_v1.")
    parser.add_argument("--postnet-ckpt", default=None,
                        help="Standalone PostNet from train_postnet.py")
    args = parser.parse_args()
    main(utt_id=args.utt_id, text=args.text, output_dir=args.output_dir,
         fs2_ckpt=args.fs2_ckpt, hifi_ckpt=args.hifi_ckpt,
         postnet_ckpt=args.postnet_ckpt)
