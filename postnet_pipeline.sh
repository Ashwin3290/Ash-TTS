#!/usr/bin/env bash
# PostNet warm-start pipeline (alignment + preprocessing already done):
#   1. FastSpeech2 warm-start from the existing best.pt: backbone frozen while
#      the fresh PostNet trains, metric-gated unfreeze, then low-LR fine-tune
#      of everything (150k steps total)
#   2. predicted-mel generation from the new best.pt        (~2 min)
#   3. HiFi-GAN fine-tune on the new predicted mels, 50k
#
# Safe to re-run after a crash: finished stages are skipped via marker files
# in .postnet/, and both trainers auto-resume from their own checkpoints.
# To force a full fresh pipeline: rm -rf .postnet
#
# Run it so it survives SSH disconnect:
#   nohup bash postnet_pipeline.sh > logs/postnet_pipeline.log 2>&1 &
#   tail -f logs/postnet_pipeline.log
set -euo pipefail
cd "$(dirname "$0")"

# off-machine backup of best checkpoints while you sleep
export HF_CKPT_REPO="${HF_CKPT_REPO:-Ashwin-C9/tts-fastspeech2-ckpt}"
export HF_HIFIGAN_CKPT_REPO="${HF_HIFIGAN_CKPT_REPO:-Ashwin-C9/tts-hifigan-ckpt}"

MARKERS=.postnet
FS2_STEPS=150000       # warm start restarts the step counter at 0
INIT_BASE=checkpoints/fastspeech2/init_base.pt

mkdir -p "$MARKERS" logs checkpoints/fastspeech2 checkpoints/hifigan

echo "=== Stage 0: dependencies ==="
pip install -q -r requirements.txt

# Preserve the 245k-run best.pt as the warm-start base BEFORE any cleanup —
# best.pt itself gets overwritten by the new run's best tracking. Downloads
# from HF if there's no local copy (fresh machine).
if [ ! -f "$INIT_BASE" ]; then
  echo "=== Stage 0b: securing warm-start base checkpoint ==="
  if [ ! -f checkpoints/fastspeech2/best.pt ]; then
    python - <<'EOF'
from pathlib import Path
from huggingface_hub import hf_hub_download
p = hf_hub_download("Ashwin-C9/tts-fastspeech2-ckpt", "best.pt", repo_type="model")
dst = Path("checkpoints/fastspeech2/best.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(Path(p).read_bytes())
print(f"Downloaded best.pt -> {dst}")
EOF
  fi
  cp checkpoints/fastspeech2/best.pt "$INIT_BASE"
  echo "Saved warm-start base: $INIT_BASE"
fi

if [ ! -f "$MARKERS/fs2_done" ]; then
  echo "=== Stage 1: FastSpeech2 PostNet warm-start, ${FS2_STEPS} steps ==="
  if [ ! -f "$MARKERS/fs2_started" ]; then
    # clean slate for the new run's checkpoints (init_base.pt is kept) —
    # guarded by fs2_started so a re-run resumes instead of wiping progress
    find checkpoints/fastspeech2 -name '*.pt' ! -name 'init_base.pt' -delete
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
    # the PostNet changes the mel distribution, so the vocoder re-tunes from
    # the official pretrained generator on the NEW mel_pred set
    rm -f checkpoints/hifigan/*.pt
    touch "$MARKERS/hifigan_started"
  fi
  # --resume auto takes precedence on re-runs; --init-g only seeds a fresh start
  python train_hifigan.py --resume --init-g pretrained_hifigan/generator_v1 --mel-dir mel_pred
  touch "$MARKERS/hifigan_done"
else
  echo "=== Stage 3: already done, skipping ==="
fi

echo ""
echo "=== PostNet pipeline complete ==="
echo "Deliverables: checkpoints/fastspeech2/best.pt + checkpoints/hifigan/g_best.pt"
echo "(both also backed up to HuggingFace whenever they improved)"
echo "Quick check: python test_inference.py   # plot + audio for a val utterance"
