#!/usr/bin/env bash
# Internal helper invoked by cloud_train.sh — not meant to be run directly.
# Runs FastSpeech2 to completion, then HiFi-GAN fine-tuning to completion.
# Each stage is independently resumable (--resume), so re-running this file
# after a crash just continues from wherever training left off; a finished
# stage exits its training loop immediately (start_step == max_steps) and
# falls through to the next stage at negligible cost.
set -euo pipefail

echo "=== Stage 1: FastSpeech2 ==="
python train_fastspeech.py --resume

echo ""
echo "=== FastSpeech2 complete. Stage 2: HiFi-GAN fine-tune ==="
python train_hifigan.py --resume --init-g pretrained_hifigan/generator_v1

echo ""
echo "=== Pipeline complete. ==="
