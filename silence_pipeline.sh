#!/usr/bin/env bash
# Silence-token pipeline:
#   1. FastSpeech2 warm-start from pre-<sil> best.pt on the new <sil> data
#   2. predicted mels from the new best.pt
#   3. HiFi-GAN fine-tune on the new predicted mels
#
# Run:
#   nohup bash silence_pipeline.sh > logs/silence.log 2>&1 &
#   tail -f logs/silence.log
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

FS2_STEPS="${FS2_STEPS:-60000}"
INIT_BASE=checkpoints/fastspeech2/best_presil.pt
MARKERS=.silence_training
mkdir -p "$MARKERS" logs

# preserve the pre-<sil> checkpoint under a name the trainer will never overwrite
if [ ! -f "$INIT_BASE" ]; then
  cp checkpoints/fastspeech2/best.pt "$INIT_BASE"
fi

if [ ! -f "$MARKERS/fs2_done" ]; then
  echo "=== Stage 1: FastSpeech2 warm-start on <sil> data, $FS2_STEPS steps ==="
  if [ ! -f "$MARKERS/fs2_started" ]; then
    # drop the overfit latest/step checkpoints from the previous run so --init
    # starts clean; best_presil.pt is the only thing we keep
    find checkpoints/fastspeech2 -name '*.pt' ! -name 'best_presil.pt' -delete
    touch "$MARKERS/fs2_started"
    python train_fastspeech.py --init "$INIT_BASE" --max-steps "$FS2_STEPS"
  else
    python train_fastspeech.py --resume --max-steps "$FS2_STEPS"
  fi
  touch "$MARKERS/fs2_done"
else
  echo "=== Stage 1: already done, skipping ==="
fi

if [ ! -f "$MARKERS/mels_done" ]; then
  echo "=== Stage 2: predicted mels from new best.pt ==="
  rm -rf data/processed/mel_pred
  python generate_mels.py
  touch "$MARKERS/mels_done"
else
  echo "=== Stage 2: already done, skipping ==="
fi

if [ ! -f "$MARKERS/hifigan_done" ]; then
  echo "=== Stage 3: HiFi-GAN fine-tune on new predicted mels ==="
  if [ ! -f "$MARKERS/hifigan_started" ]; then
    # the <sil> model changes the predicted-mel distribution (real pauses now),
    # so re-tune from the known-good generator on the new mel_pred set
    cp checkpoints/hifigan/g_best.pt checkpoints/hifigan/g_presil.pt
    find checkpoints/hifigan -name '*.pt' ! -name 'g_presil.pt' -delete
    touch "$MARKERS/hifigan_started"
  fi
  python train_hifigan.py --resume --init-g checkpoints/hifigan/g_presil.pt --mel-dir mel_pred
  touch "$MARKERS/hifigan_done"
else
  echo "=== Stage 3: already done, skipping ==="
fi

echo ""
echo "=== Silence pipeline complete ==="
echo "Deliverables: checkpoints/fastspeech2/best.pt + checkpoints/hifigan/g_best.pt"
echo "Pre-<sil> baselines kept as best_presil.pt / g_presil.pt"