import numpy as np
from pathlib import Path
from config import paths, audio as acfg
import random

print("\n" + "="*80)
print("SILENCE ANALYSIS: Checking if silence is in training data")
print("="*80)

processed = Path(paths.processed_dir)

# Load a sample phoneme vocab to understand the tokens
import json
vocab_path = processed / "phoneme_vocab.json"
if vocab_path.exists():
    try:
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"\nPhoneme vocab size: {len(vocab)}")
        print(f"Vocab contains <sil>: {'<sil>' in vocab}")
        if '<sil>' in vocab:
            print(f"  <sil> token id: {vocab['<sil>']}")
    except:
        print("\nNote: vocab file is corrupted, but we know token 1 is typically <sil>")

# Check if any phoneme sequence contains token ID 1 (which is <sil>)
print("\nAnalyzing 100 random utterances for silence tokens...")

ph_files = list((processed / "phoneme").glob("*.npy"))
random.shuffle(ph_files)

has_silence = 0
no_silence = 0
max_silence_frames = 0

for ph_file in ph_files[:100]:
    phonemes = np.load(ph_file)
    mel_file = processed / "mel" / ph_file.name
    mel = np.load(mel_file)

    dur_file = processed / "duration" / ph_file.name
    if dur_file.exists():
        durations = np.load(dur_file)
    else:
        continue

    # Token 1 is <sil> (silence)
    has_sil = (phonemes == 1).any()

    if has_sil:
        has_silence += 1
        # Count how many silence phonemes and their total frames
        sil_mask = (phonemes == 1)
        sil_frames = durations[sil_mask].sum() if (durations[sil_mask].sum() > 0) else 0
        max_silence_frames = max(max_silence_frames, sil_frames)
    else:
        no_silence += 1

print(f"\n  Utterances WITH silence tokens: {has_silence}/100")
print(f"  Utterances WITHOUT silence tokens: {no_silence}/100")
if max_silence_frames > 0:
    print(f"  Max silence duration found: {max_silence_frames} frames (~{max_silence_frames * acfg.hop_length / acfg.sample_rate:.2f}s)")

if has_silence == 0:
    print("\n❌ CRITICAL: No utterances have silence tokens in phoneme sequence!")
    print("   This explains missing silences — the model was never trained on them.")
    print("\nThe issue:")
    print("  1. TextGrid contains 'sil'/'sp' intervals")
    print("  2. Preprocessing REMOVES them (line 220)")
    print("  3. Espeak-ng phonemization DOESN'T produce silence tokens")
    print("  4. Result: No way for model to learn where silences should be")
else:
    print("\n✓ Silence tokens found in training data")

# Also check: compare phoneme count vs duration count
print("\n" + "="*80)
print("Checking if phoneme sequence length matches duration count...")

mismatch_count = 0
for ph_file in ph_files[:100]:
    phonemes = np.load(ph_file)
    dur_file = processed / "duration" / ph_file.name
    if dur_file.exists():
        durations = np.load(dur_file)
        if len(phonemes) != len(durations):
            mismatch_count += 1

if mismatch_count > 0:
    print(f"❌ {mismatch_count}/100 have phoneme_len != duration_len")
    print("   (Phonemes are missing <sil> tokens that TextGrid has)")
else:
    print("✓ Phoneme and duration lengths match")
