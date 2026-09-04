# Training on Vast.ai

## Quick Start

### Step 1: Push Data & Model (Local)

```bash
export HF_TOKEN="hf_xxxxx"  # Your HuggingFace token
python push_to_hf.py
```

This uploads:
- Regenerated phoneme/duration files (940 with silence tokens)
- Current best.pt checkpoint
- Train/val manifests

To `Ashwin-C9/tts-fastspeech2-ckpt` and `Ashwin-C9/tts-data-silence-fix`

### Step 2: Rent Vast.ai GPU

- Pick an instance with 24GB+ VRAM (RTX 3090, A100, etc.)
- Launch with PyTorch image

### Step 3: Clone & Train (On Vast.ai VM)

```bash
# Clone repo
git clone https://github.com/Ashwin3290/Ash-TTS.git
cd Ash-TTS
git checkout v2

# Install dependencies
pip install -r requirements.txt

# Set token
export HF_TOKEN="hf_xxxxx"

# Train (pulls model + data from HF, resumes training)
python train_on_vm.py
```

**That's it.** The script will:
1. Download `best.pt` from HuggingFace
2. Download all phoneme/duration files
3. Resume training with `--resume`

### Monitoring

Training logs go to:
- `logs/` - TensorBoard events
- Console output shows loss every 100 steps

Kill training anytime with Ctrl+C. Latest checkpoint saved to `checkpoints/fastspeech2/latest.pt`

---

## What's New in Training

- **Silence tokens**: 940 utterances now have `<sil>` tokens at proper positions
- **Silence-weighted loss**: Silence frames (low energy) get 2x loss weight
- **Expected**: Model learns to generate clear silence boundaries
- **Duration**: ~50-100k steps to converge (fine-tuning from best.pt)

## Checkpoints Saved

- `best.pt` - Best validation loss (auto-pushed to HF)
- `latest.pt` - Most recent checkpoint (local only)
- `step_*.pt` - Periodic snapshots (oldest pruned to save disk)

## Troubleshooting

**Data download hangs:**
- Vast.ai sometimes has slow network. Manual download:
  ```bash
  huggingface-cli download Ashwin-C9/tts-data-silence-fix --include "phoneme/*" "duration/*" --local-dir data/processed
  ```

**OOM errors:**
- Reduce batch_size in `config.py` (TrainFastSpeechConfig.batch_size)
- Currently 16, try 8 or 4

**Model not found:**
- Verify HF_TOKEN is set and has access to repos
- Check repos exist: `Ashwin-C9/tts-fastspeech2-ckpt`, `Ashwin-C9/tts-data-silence-fix`
