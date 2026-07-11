#!/usr/bin/env bash
# Runs FastSpeech2 to completion, then HiFi-GAN fine-tuning to completion.
# Runs in the foreground — output stays on screen, nothing detached.
#
# Both stages are independently resumable (--resume), so re-running this
# script after an interruption just continues from wherever it left off.
set -euo pipefail

echo "=== Stage 1: FastSpeech2 ==="
python train_fastspeech.py --resume

echo ""
echo "=== FastSpeech2 complete. Stage 2: HiFi-GAN fine-tune ==="
python train_hifigan.py --resume --init-g pretrained_hifigan/generator_v1
