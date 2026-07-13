"""
Pull the latest FastSpeech2 checkpoint from HuggingFace and sanity-check it
end to end: pick a validation utterance, run real text-to-speech inference
(not teacher-forced with ground-truth durations), plot predicted vs ground
truth mel, and vocode the prediction to a listenable wav.

Since HiFi-GAN fine-tuning hasn't started yet, this uses the raw pretrained
official HiFi-GAN checkpoint (GAN model/generator_v1) as the vocoder — good
enough to judge whether FastSpeech2's mel predictions are already legible.

Usage:
    python test_inference.py                      # random validation utterance
    python test_inference.py --utt-id LJ001-0001   # a specific one
    python test_inference.py --text "Custom sentence to synthesize."
"""

import argparse
import json
import os
import sys
import random
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
import soundfile as sf
import librosa
from pathlib import Path
from huggingface_hub import hf_hub_download

from config import audio as acfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from vocoder.generator import load_pretrained_generator
from inference import text_to_phonemes, load_phoneme_vocab

HF_CKPT_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"
HIFIGAN_DIR = Path("GAN model")  # generator_v1 + config.json, already local


def pull_best_checkpoint():
    # only best.pt is published to HF now — latest.pt stays local-only on
    # whatever instance is training, since it's just a same-session resume
    # point and best.pt is the one actually worth pulling down to test
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
        item = random.choice(manifest)
    else:
        item = next((m for m in manifest if m["id"] == utt_id), None)
        if item is None:
            raise ValueError(f"{utt_id} not found in val_manifest.json")
    return item


def denorm_mel(mel):
    """Undo preprocess.py's [-1,1] normalisation -> natural-log mel scale."""
    return (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min


def main(utt_id, text, output_dir, hifi_ckpt=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = pull_best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Checkpoint step: {ckpt['step']}")

    with open(paths.processed_dir / "stats.json") as f:
        stats = json.load(f)
    model = FastSpeech2().to(device)
    model.variance_adaptor.set_stats(**stats)
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
        gt_mel = np.load(paths.processed_dir / "mel" / f"{utt_id}.npy")  # (T, 80), normalised
        gt_wav_path = paths.data_root / "wavs" / f"{utt_id}.wav"
    else:
        print(f"Custom text: {text}")

    vocab = load_phoneme_vocab()
    phonemes = text_to_phonemes(text, vocab)
    if not phonemes:
        raise ValueError("Phonemization produced empty sequence.")

    ph_tensor = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)
    ph_lens = torch.tensor([len(phonemes)], dtype=torch.long).to(device)

    with torch.no_grad():
        _, mel_pred, _, _, _, mel_lens = model(ph_tensor, ph_lens)  # PostNet-refined mel_after
    pred_mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy()  # (T, 80), normalised
    print(f"Predicted mel frames: {pred_mel.shape[0]} "
          f"({pred_mel.shape[0] * acfg.hop_length / acfg.sample_rate:.2f}s)")

    # --- plot predicted vs ground truth (if we have one) ---
    if gt_mel is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        axes[0].imshow(gt_mel.T, aspect="auto", origin="lower", interpolation="none")
        axes[0].set_title(f"Ground truth mel — {utt_id} ({gt_mel.shape[0]} frames)")
        axes[1].imshow(pred_mel.T, aspect="auto", origin="lower", interpolation="none")
        axes[1].set_title(f"Predicted mel — step {ckpt['step']} ({pred_mel.shape[0]} frames)")
        for ax in axes:
            ax.set_ylabel("mel bin")
        axes[1].set_xlabel("frame")
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(pred_mel.T, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(f"Predicted mel — step {ckpt['step']}")
        ax.set_xlabel("frame"); ax.set_ylabel("mel bin")
    fig.tight_layout()
    plot_path = output_dir / "mel_comparison.png"
    fig.savefig(plot_path, dpi=120)
    print(f"Saved plot: {plot_path}")

    # --- vocode the predicted mel ---
    # both paths expect the raw natural-log mel scale now — train_hifigan.py's
    # WavMelDataset denormalises before training (see its comment for why
    # feeding the [-1,1] preprocessing scale to either a pretrained or
    # from-scratch generator produced unintelligible/noise-like output)
    if hifi_ckpt:
        from vocoder.generator import Generator, config_from_hcfg
        hifi = Generator(config_from_hcfg(hcfg)).to(device)
        hifi_state = torch.load(hifi_ckpt, map_location=device)
        hifi.load_state_dict(hifi_state["generator"])
        hifi.eval()
        hifi.remove_weight_norm()
        print(f"Vocoder: fine-tuned checkpoint {hifi_ckpt} (step {hifi_state.get('step', '?')})")
    else:
        try:
            hifi = load_pretrained_generator(
                str(HIFIGAN_DIR / "generator_v1"), str(HIFIGAN_DIR / "config.json"), device
            )
        except:
            hifi = load_pretrained_generator(
                str("pretrained_hifigan" / "generator_v1"), str('pretrained_hifigan' / "config.json"), device
            ) 

        print("Vocoder: raw official generator_v1")
    mel_for_vocoder = denorm_mel(pred_mel)
    mel_t = torch.from_numpy(mel_for_vocoder.T).float().unsqueeze(0).to(device)
    with torch.no_grad():
        wav = hifi(mel_t).squeeze().cpu().numpy()

    pred_wav_path = output_dir / "predicted.wav"
    sf.write(str(pred_wav_path), wav, acfg.sample_rate)
    print(f"Saved predicted audio: {pred_wav_path} ({len(wav) / acfg.sample_rate:.2f}s)")

    if gt_wav_path is not None and gt_wav_path.exists():
        gt_wav, sr = sf.read(str(gt_wav_path))
        if sr != acfg.sample_rate:
            gt_wav = librosa.resample(gt_wav, orig_sr=sr, target_sr=acfg.sample_rate)
        gt_wav_out = output_dir / "ground_truth.wav"
        sf.write(str(gt_wav_out), gt_wav, acfg.sample_rate)
        print(f"Saved ground truth audio: {gt_wav_out} ({len(gt_wav) / acfg.sample_rate:.2f}s)")

    print(f"\nDone. Check {output_dir}/ for mel_comparison.png, predicted.wav"
          + (", ground_truth.wav" if gt_wav_path else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt-id", default=None, help="Specific val_manifest utterance id")
    parser.add_argument("--text", default=None, help="Custom text instead of a val utterance")
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument("--hifi-ckpt", default=None,
                        help="Path to a fine-tuned train_hifigan.py checkpoint "
                             "(e.g. checkpoints/hifigan/g_latest.pt). Default: "
                             "raw official pretrained generator_v1.")
    args = parser.parse_args()
    main(utt_id=args.utt_id, text=args.text, output_dir=args.output_dir,
         hifi_ckpt=args.hifi_ckpt)
