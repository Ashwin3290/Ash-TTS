import numpy as np
from pathlib import Path
from config import paths

print("\n" + "="*80)
print("ALIGNMENT ANALYSIS: Duration file integrity")
print("="*80)

processed = Path(paths.processed_dir)
dur_files = sorted(processed.glob("duration/*.npy"))

if not dur_files:
    print("ERROR: No duration files found in", processed / "duration")
    exit(1)

print(f"Found {len(dur_files)} duration files\n")

alignment_issues = {
    "zero_duration": 0,
    "mismatch_total": 0,
    "missing_mel": 0,
}

sample_mismatches = []

for i, dur_file in enumerate(dur_files[:50]):  # Check first 50
    mel_file = processed / "mel" / dur_file.name
    ph_file = processed / "phoneme" / dur_file.name

    if not mel_file.exists() or not ph_file.exists():
        alignment_issues["missing_mel"] += 1
        continue

    durations = np.load(dur_file)
    mel = np.load(mel_file)
    phonemes = np.load(ph_file)

    dur_sum = durations.sum()
    mel_frames = mel.shape[0]

    # Check for alignment issues
    if dur_sum != mel_frames:
        alignment_issues["mismatch_total"] += 1
        if len(sample_mismatches) < 5:
            sample_mismatches.append({
                "id": dur_file.stem,
                "phonemes": len(phonemes),
                "mel_frames": mel_frames,
                "dur_sum": dur_sum,
                "diff": mel_frames - dur_sum,
            })

    zero_dur = (durations == 0).sum()
    if zero_dur > 0:
        alignment_issues["zero_duration"] += 1
        if len([s for s in sample_mismatches if "zero" not in s.get("type", "")]) < 3:
            sample_mismatches.append({
                "id": dur_file.stem,
                "type": "zero_duration",
                "zero_count": zero_dur,
                "total_phonemes": len(phonemes),
            })

print("RESULTS:")
print(f"  Total utterances checked: {len(dur_files[:50])}")
print(f"  ❌ Duration sum ≠ mel frames: {alignment_issues['mismatch_total']}")
print(f"  ❌ Phonemes with 0 duration: {alignment_issues['zero_duration']}")
print(f"  ❌ Missing files: {alignment_issues['missing_mel']}")

if sample_mismatches:
    print("\nSample mismatches:")
    for m in sample_mismatches:
        if "type" in m and m["type"] == "zero_duration":
            print(f"  {m['id']}: {m['zero_count']}/{m['total_phonemes']} zero-duration phonemes")
        else:
            print(f"  {m['id']}: mel={m['mel_frames']} frames, dur_sum={m['dur_sum']} (diff={m['diff']})")

print("\nDuration statistics (first 100 files):")
all_durations = []
for dur_file in dur_files[:100]:
    durations = np.load(dur_file)
    all_durations.extend(durations)

all_durations = np.array(all_durations)
print(f"  Mean phoneme duration: {np.mean(all_durations):.2f} frames")
print(f"  Median: {np.median(all_durations):.2f}")
print(f"  Min: {np.min(all_durations)}, Max: {np.max(all_durations)}")
print(f"  Phonemes with 1 frame: {(all_durations == 1).sum()}/{len(all_durations)} ({100*(all_durations==1).sum()/len(all_durations):.1f}%)")
print(f"  Phonemes with 0 frames: {(all_durations == 0).sum()}/{len(all_durations)} ({100*(all_durations==0).sum()/len(all_durations):.1f}%)")

if alignment_issues["mismatch_total"] > 0 or alignment_issues["zero_duration"] > 0:
    print("\n❌ ALIGNMENT ISSUES DETECTED - CTC forced alignment may be corrupted")
else:
    print("\n✓ Alignment looks good")
