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

export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"
export HF_DATA_REPO="${HF_DATA_REPO:-Ashwin-C9/tts-data-silence-fix}"

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

  # Download model checkpoint
  echo "Downloading model from $HF_CKPT_REPO..."
  mkdir -p checkpoints/fastspeech2
  python -c "
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HF_TOKEN')
hf_hub_download(
    repo_id='$HF_CKPT_REPO',
    filename='best.pt',
    local_dir='checkpoints/fastspeech2',
    token=token,
)
# Rename best.pt to latest.pt for auto-resume
import shutil
best_path = 'checkpoints/fastspeech2/best.pt'
latest_path = 'checkpoints/fastspeech2/latest.pt'
if os.path.exists(best_path):
    shutil.move(best_path, latest_path)
print('✓ Model checkpoint ready for resume')
"

  # Download manifests
  echo "Downloading manifests from $HF_DATA_REPO..."
  mkdir -p data/processed
  for manifest in train_manifest.json val_manifest.json; do
    python -c "
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HF_TOKEN')
try:
    hf_hub_download(
        repo_id='$HF_DATA_REPO',
        filename='$manifest',
        local_dir='data/processed',
        repo_type='dataset',
        token=token,
    )
    print('✓ $manifest downloaded')
except Exception as e:
    print(f'⚠️ $manifest: {e}')
"
  done

  # Download phoneme and duration files
  echo "Downloading phoneme files from $HF_DATA_REPO..."
  python -c "
import subprocess
import os

token = os.environ.get('HF_TOKEN')
subprocess.run(
    ['huggingface-cli', 'download', '$HF_DATA_REPO',
     '--include', 'phoneme/*', '--local-dir', 'data/processed', '--repo-type', 'dataset'],
    env={**os.environ, 'HF_TOKEN': token},
    check=False,
)
"

  echo "Downloading duration files from $HF_DATA_REPO..."
  python -c "
import subprocess
import os

token = os.environ.get('HF_TOKEN')
subprocess.run(
    ['huggingface-cli', 'download', '$HF_DATA_REPO',
     '--include', 'duration/*', '--local-dir', 'data/processed', '--repo-type', 'dataset'],
    env={**os.environ, 'HF_TOKEN': token},
    check=False,
)
"

  touch "$MARKERS/download_done"
  echo "✓ Download stage complete"
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
