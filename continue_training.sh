#!/usr/bin/env bash
# Continuation run (~6.5h):
#   1. FastSpeech2: resume from latest.pt, train the next 250k steps
#      (250k -> 500k ceiling, already set in config.py)
#   2. refresh predicted mels from the improved best.pt
#   3. HiFi-GAN: resume from g_latest.pt/d_latest.pt, 25k -> 50k
#
# Idempotent: re-running after a crash just resumes each stage — a finished
# trainer exits immediately (start step == max_steps) and falls through.
#
# Run it so it survives SSH disconnect:
#   nohup bash continue_training.sh > logs/continue.log 2>&1 &
#   tail -f logs/continue.log
set -euo pipefail
cd "$(dirname "$0")"

# off-machine backup of best checkpoints whenever they improve
export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

mkdir -p logs

echo "=== Stage 1: FastSpeech2 resume -> 500k ==="
python train_fastspeech.py --resume

echo ""
echo "=== Stage 2: refresh predicted mels from the improved best.pt ==="
rm -rf data/processed/mel_pred
python generate_mels.py

echo ""
echo "=== Stage 3: HiFi-GAN resume -> 50k ==="
# --init-g only seeds if g_latest.pt is somehow missing; --resume wins otherwise
python train_hifigan.py --resume --init-g pretrained_hifigan/generator_v1 --mel-dir mel_pred

echo ""
echo "=== Done. checkpoints/fastspeech2/best.pt + checkpoints/hifigan/g_best.pt ==="
echo "(both pushed to HuggingFace whenever they improved during the run)"
