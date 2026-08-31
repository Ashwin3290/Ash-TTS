import torch
import numpy as np
from pathlib import Path
from config import paths, train as tcfg, audio as acfg, model as mcfg
from data.dataset import LJSpeechDataset, collate_fn
from torch.utils.data import DataLoader
from model.fastspeech2 import FastSpeech2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fs2 = FastSpeech2().to(device)
    ckpt_path = Path("checkpoints/fastspeech2/best.pt")
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    fs2.load_state_dict(ckpt["model"], strict=False)
    fs2.eval()

    # Load validation data without multiprocessing
    val_ds = LJSpeechDataset(paths.processed_dir / "val_manifest.json", paths.processed_dir)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("\n" + "="*80)
    print("DURATION ANALYSIS: Predicted vs Ground Truth")
    print("="*80)

    dur_pred_ratios = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10:
                break

            ph = batch["phonemes"].to(device)
            ph_lens = batch["ph_lens"].to(device)
            dur_gt = batch["durations"].cpu().numpy()

            mel_before, _, log_dur_pred, _, _, _ = fs2(ph, ph_lens, durations_gt=batch["durations"].to(device))

            dur_pred = torch.exp(log_dur_pred) - 1  # reverse the log(dur+1) transform
            dur_pred = dur_pred.cpu().numpy()

            for j in range(len(ph_lens)):
                L = ph_lens[j].item()
                if L == 0:
                    continue

                gt_sum = dur_gt[j, :L].sum()
                pred_sum = dur_pred[j, :L].sum()
                ratio = pred_sum / (gt_sum + 1e-6)
                dur_pred_ratios.append(ratio)

                print(f"\nBatch {i}, Utt {j}:")
                print(f"  Phonemes: {L}")
                print(f"  GT duration sum: {gt_sum}")
                print(f"  Pred duration sum: {pred_sum:.1f}")
                print(f"  Ratio (pred/gt): {ratio:.3f}x")
                print(f"  First 10 durations - GT: {dur_gt[j, :min(10, L)]}")
                print(f"  First 10 durations - Pred: {dur_pred[j, :min(10, L)].astype(int)}")

                # Check for near-zero predictions
                zero_pred = (dur_pred[j, :L] < 0.5).sum()
                if zero_pred > 0:
                    print(f"  ⚠️  {zero_pred}/{L} phonemes predicted <0.5 frames!")

    if dur_pred_ratios:
        print("\n" + "="*80)
        print("SUMMARY:")
        print(f"  Mean ratio (pred/gt): {np.mean(dur_pred_ratios):.3f}x")
        print(f"  Median ratio: {np.median(dur_pred_ratios):.3f}x")
        print(f"  Min: {np.min(dur_pred_ratios):.3f}x, Max: {np.max(dur_pred_ratios):.3f}x")
        if np.mean(dur_pred_ratios) > 1.1:
            print(f"  ❌ DURATION OVERPREDICTION DETECTED (predicting {(np.mean(dur_pred_ratios)-1)*100:.1f}% extra frames)")
        elif np.mean(dur_pred_ratios) < 0.9:
            print(f"  ❌ DURATION UNDERPREDICTION DETECTED (predicting {(1-np.mean(dur_pred_ratios))*100:.1f}% fewer frames)")
        else:
            print(f"  ✓ Durations look balanced")

if __name__ == "__main__":
    main()
