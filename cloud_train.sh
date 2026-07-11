#!/usr/bin/env bash
# One-shot setup + training entrypoint for a fresh cloud GPU instance (Vast.ai etc).
#
# Usage (on the instance):
#   export HF_TOKEN=hf_xxx          # write-access token, from huggingface.co/settings/tokens
#   bash cloud_train.sh
#
# What it does:
#   1. Clones/pulls the repo, installs deps
#   2. Logs into HuggingFace, downloads the processed dataset + pretrained
#      HiFi-GAN seed checkpoint if not already present
#   3. Starts TensorBoard in the background (Vast.ai auto-detects/exposes the
#      port), covering both FastSpeech2 and HiFi-GAN logs under logs/
#   4. Runs the full pipeline detached via nohup (_cloud_pipeline.sh):
#      FastSpeech2 to completion, then HiFi-GAN fine-tuning to completion —
#      survives SSH disconnect, auto-resumes from checkpoints on every re-run
#
# Re-running this script is safe/idempotent: it skips steps already done and
# just resumes wherever the pipeline left off.
set -euo pipefail

REPO_URL="https://github.com/Ashwin3290/Ash-TTS.git"
REPO_DIR="Ash-TTS"
HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"
PRETRAINED_HIFIGAN_REPO="Ashwin-C9/tts-hifigan-ckpt"
TB_PORT="${TB_PORT:-6006}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: export HF_TOKEN first (a write-access token from huggingface.co/settings/tokens)."
  exit 1
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "==> Repo already present, pulling latest..."
  git -C "$REPO_DIR" pull
else
  echo "==> Cloning $REPO_URL..."
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> Installing dependencies..."
pip install -q -r requirements.txt

if [ ! -f "data/processed/train_manifest.json" ]; then
  echo "==> Downloading processed dataset..."
  python download_processed.py
else
  echo "==> Processed dataset already present, skipping download."
fi

if [ ! -f "pretrained_hifigan/generator_v1" ]; then
  echo "==> Downloading pretrained HiFi-GAN seed checkpoint..."
  mkdir -p pretrained_hifigan
  huggingface-cli download "$PRETRAINED_HIFIGAN_REPO" generator_v1 config.json \
      --local-dir pretrained_hifigan
else
  echo "==> Pretrained HiFi-GAN seed already present, skipping download."
fi

mkdir -p logs
export HF_CKPT_REPO
export HF_HIFIGAN_CKPT_REPO

if ! pgrep -f "tensorboard.*--logdir logs" > /dev/null 2>&1; then
  echo "==> Starting TensorBoard on port $TB_PORT..."
  nohup tensorboard --logdir logs --host 0.0.0.0 --port "$TB_PORT" \
      > logs/tensorboard.log 2>&1 &
  disown
else
  echo "==> TensorBoard already running."
fi

if pgrep -f "python train_fastspeech.py" > /dev/null 2>&1 || pgrep -f "python train_hifigan.py" > /dev/null 2>&1; then
  echo "==> Pipeline already running in the background — nothing to do."
else
  echo "==> Starting training pipeline in the background (FastSpeech2 -> HiFi-GAN fine-tune)..."
  nohup bash _cloud_pipeline.sh > logs/pipeline.log 2>&1 &
  PIPELINE_PID=$!
  disown
  echo "    PID: $PIPELINE_PID"
fi

echo ""
echo "==> Done. Pipeline runs detached — safe to close this SSH session."
echo "    Live log:    tail -f $(pwd)/logs/pipeline.log"
echo "    TensorBoard: http://<instance-ip>:$TB_PORT  (or via Vast.ai's port-forward UI)"
echo "    Re-run this script any time to check status / resume after a crash."
