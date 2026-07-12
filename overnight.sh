#!/usr/bin/env bash
# Overnight pipeline (supersedes cloud_train.sh for this run):
#   1. phone-level re-alignment of all durations   (~30-40 min)
#   2. FastSpeech2 retrain from scratch, 250k steps (~15h)
#   3. predicted-mel generation from best.pt        (~2 min)
#   4. HiFi-GAN fine-tune on predicted mels, 25k    (~1.7h)
#
# Safe to re-run after a crash: finished stages are skipped via marker files
# in .overnight/, and both trainers auto-resume from their own checkpoints.
# To force a full fresh pipeline: rm -rf .overnight
#
# Run it so it survives SSH disconnect:
#   nohup bash overnight.sh > logs/overnight.log 2>&1 &
#   tail -f logs/overnight.log
set -euo pipefail
cd "$(dirname "$0")"

# off-machine backup of best checkpoints while you sleep
export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

MARKERS=.overnight
mkdir -p "$MARKERS" logs

echo "=== Stage 0: dependencies ==="
pip install -q -r requirements.txt

if [ ! -f "$MARKERS/align_done" ]; then
  echo "=== Stage 1: phone-level re-alignment (overwrites all durations) ==="
  python align.py --overwrite
  touch "$MARKERS/align_done"
else
  echo "=== Stage 1: already done, skipping ==="
fi

if [ ! -f "$MARKERS/fs2_done" ]; then
  echo "=== Stage 2: FastSpeech2 retrain, 250k steps ==="
  if [ ! -f "$MARKERS/fs2_started" ]; then
    # clean start required: new durations invalidate all old checkpoints.
    # guarded by fs2_started so a re-run resumes instead of wiping progress.
    rm -f checkpoints/fastspeech2/*.pt
    touch "$MARKERS/fs2_started"
  fi
  python train_fastspeech.py --resume
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
  echo "=== Stage 4: HiFi-GAN fine-tune on predicted mels, 25k steps ==="
  if [ ! -f "$MARKERS/hifigan_started" ]; then
    rm -f checkpoints/hifigan/*.pt
    touch "$MARKERS/hifigan_started"
  fi
  # --resume auto takes precedence on re-runs; --init-g only seeds a fresh start
  python train_hifigan.py --resume --init-g pretrained_hifigan/generator_v1 --mel-dir mel_pred
  touch "$MARKERS/hifigan_done"
else
  echo "=== Stage 4: already done, skipping ==="
fi

echo ""
echo "=== Overnight pipeline complete ==="
echo "Deliverables: checkpoints/fastspeech2/best.pt + checkpoints/hifigan/g_best.pt"
echo "(both also backed up to HuggingFace whenever they improved)"
