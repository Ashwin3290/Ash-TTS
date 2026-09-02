#!/usr/bin/env bash
# Vast.ai training pipeline (based on overnight.sh):
#   1. Download model & data from HuggingFace
#   2. FastSpeech2 training (~6 hrs)
#   3. Predicted mel generation from best.pt
#   4. HiFi-GAN fine-tune on predicted mels (~4-6 hrs)
#
# Safe to re-run after crash: marker files in .vast_training/ track completion

set -euo pipefail
cd "$(dirname "$0")"

export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

MARKERS=.vast_training
mkdir -p "$MARKERS" logs

echo "=== Stage 0: dependencies ==="
pip install -q -r requirements.txt

if [ ! -f "$MARKERS/download_done" ]; then
  echo "=== Stage 1: download model & data from HuggingFace ==="
  python download_vast.py
  touch "$MARKERS/download_done"
else
  echo "=== Stage 1: already done, skipping ==="
fi

if [ ! -f "$MARKERS/fs2_done" ]; then
  echo "=== Stage 2: FastSpeech2 training (~6 hrs) ==="
  python train_fastspeech.py --resume | tee logs/fastspeech2.log
  touch "$MARKERS/fs2_done"
else
  echo "=== Stage 2: already done, skipping ==="
fi

if [ ! -f "$MARKERS/mels_done" ]; then
  echo "=== Stage 3: predicted mels from best.pt ==="
  rm -rf data/processed/mel_pred
  python generate_mels.py
  touch "$MARKERS/mels_done"
else
  echo "=== Stage 3: already done, skipping ==="
fi

if [ ! -f "$MARKERS/hifigan_done" ]; then
  echo "=== Stage 4: HiFi-GAN fine-tune on predicted mels (~4-6 hrs) ==="
  python train_hifigan.py --resume | tee logs/hifigan.log
  touch "$MARKERS/hifigan_done"
else
  echo "=== Stage 4: already done, skipping ==="
fi

echo ""
echo "=== Training pipeline complete ==="
echo "Deliverables: checkpoints/fastspeech2/best.pt + checkpoints/hifigan/g_best.pt"
echo "(both backed up to HuggingFace whenever they improved)"
