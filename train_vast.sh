#!/usr/bin/env bash
# Vast.ai unified training pipeline:
#   1. Download model checkpoint from HuggingFace
#   2. Download dataset (phoneme/duration/manifests) from HuggingFace
#   3. FastSpeech2 training (~6 hrs)
#   4. HiFi-GAN fine-tune (~4-6 hrs)
#
# Checkpoints: keeps only best.pt + latest.pt per model, deletes step_*.pt to save quota
# Total: ~10-12 hrs, ~$2 on RTX 4080S
#
# Usage:
#   export HF_TOKEN="hf_xxxxx"
#   bash train_vast.sh
#
# Safe to re-run after crash: marker files in .vast_training/ track completion,
# both trainers auto-resume from their own checkpoints.

set -euo pipefail
cd "$(dirname "$0")"

# HF repos (also check download_vast.py if changing these)
export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

MARKERS=.vast_training
mkdir -p "$MARKERS" logs

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN environment variable first"
  echo "  export HF_TOKEN='hf_xxxxx'"
  exit 1
fi

echo "=== Stage 0: dependencies ==="
pip install -q -r requirements.txt

if [ ! -f "$MARKERS/download_done" ]; then
  echo "=== Stage 1: download model & data from HuggingFace ==="
  python download_vast.py
  touch "$MARKERS/download_done"
else
  echo "=== Stage 1: already done, skipping ==="
fi

if [ ! -f "$MARKERS/fastspeech2_done" ]; then
  echo ""
  echo "=== Stage 2: FastSpeech2 training (~6 hrs) ==="
  python train_fastspeech.py --resume | tee logs/fastspeech2.log
  touch "$MARKERS/fastspeech2_done"
else
  echo "=== Stage 2: already done, skipping ==="
fi

if [ ! -f "$MARKERS/hifigan_done" ]; then
  echo ""
  echo "=== Stage 3: HiFi-GAN fine-tune (~4-6 hrs) ==="
  # HiFi-GAN fine-tunes on mel spectrograms from best.pt
  # --resume auto-resumes from g_latest.pt/d_latest.pt if they exist
  python train_hifigan.py --resume | tee logs/hifigan.log
  touch "$MARKERS/hifigan_done"
else
  echo "=== Stage 3: already done, skipping ==="
fi

echo ""
echo "=== Training pipeline complete ==="
echo "Deliverables:"
echo "  FastSpeech2: checkpoints/fastspeech2/best.pt"
echo "  HiFi-GAN: checkpoints/hifigan/g_best.pt"
echo ""
echo "(both backed up to HuggingFace whenever they improved)"
echo "✓ Total time: ~10-12 hrs | Cost: ~\$2 on RTX 4080S"
