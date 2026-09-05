"""
Train the PostNet on its own, decoupled from FastSpeech2.

Inputs are cached pairs produced by generate_mels.py:
    data/processed/mel_pred/<id>.npy   FastSpeech2 output, teacher-forced durations
    data/processed/mel/<id>.npy        ground truth
Both are on the [-1, 1] normalised scale and frame-aligned, so this is a plain
supervised residual regression: minimise L1(mel_pred + postnet(mel_pred), mel_gt).

FastSpeech2 is never loaded, so a run is minutes rather than hours. The saved
checkpoint holds only the postnet.* weights and can either be applied at
inference time (inference.py --postnet-ckpt) or folded into a FastSpeech2
checkpoint later.

Usage:
    python train_postnet.py --steps 6000
    python train_postnet.py --steps 6000 --out checkpoints/fastspeech2/postnet_only.pt

Decision rule after the run: the printed gain is the val L1 improvement over
doing nothing. Under 1% means the PostNet is not worth carrying.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import paths
from model.decoder import PostNet


class MelPairDataset(Dataset):
    def __init__(self, utt_ids):
        self.utt_ids = utt_ids

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, i):
        utt_id = self.utt_ids[i]
        pred = np.load(paths.processed_dir / "mel_pred" / f"{utt_id}.npy")
        gt = np.load(paths.processed_dir / "mel" / f"{utt_id}.npy")
        n = min(len(pred), len(gt))
        return torch.from_numpy(pred[:n]).float(), torch.from_numpy(gt[:n]).float()


def collate(batch):
    max_t = max(p.size(0) for p, _ in batch)
    n_mels = batch[0][0].size(1)
    pred = torch.zeros(len(batch), max_t, n_mels)
    gt = torch.zeros(len(batch), max_t, n_mels)
    mask = torch.zeros(len(batch), max_t, 1)
    for i, (p, g) in enumerate(batch):
        pred[i, :p.size(0)] = p
        gt[i, :g.size(0)] = g
        mask[i, :p.size(0)] = 1.0
    return pred, gt, mask


def masked_l1(pred, target, mask):
    """Per-element L1 over valid frames."""
    return ((pred - target).abs() * mask).sum() / (mask.sum() * pred.size(-1))


def manifest_ids(name):
    with open(paths.processed_dir / name) as f:
        manifest = json.load(f)
    pred_dir = paths.processed_dir / "mel_pred"
    return [m["id"] for m in manifest if (pred_dir / f"{m['id']}.npy").exists()]


@torch.no_grad()
def evaluate(net, loader, device):
    """Returns (l1_without_postnet, l1_with_postnet)."""
    net.eval()
    before = after = 0.0
    n = 0
    for pred, gt, mask in loader:
        pred, gt, mask = pred.to(device), gt.to(device), mask.to(device)
        before += masked_l1(pred, gt, mask).item()
        after += masked_l1(pred + net(pred), gt, mask).item()
        n += 1
    net.train()
    return before / n, after / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/fastspeech2/postnet_only.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pred_dir = paths.processed_dir / "mel_pred"
    if not pred_dir.exists():
        raise RuntimeError(f"{pred_dir} not found. Run generate_mels.py first.")

    train_ids = manifest_ids("train_manifest.json")
    val_ids = manifest_ids("val_manifest.json")
    if not train_ids or not val_ids:
        raise RuntimeError("No (mel_pred, mel) pairs found for the manifests.")
    print(f"Train pairs: {len(train_ids)}  Val pairs: {len(val_ids)}")

    train_loader = DataLoader(MelPairDataset(train_ids), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(MelPairDataset(val_ids), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    net = PostNet().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"PostNet parameters: {n_params:,}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base_before, base_after = evaluate(net, val_loader, device)
    print(f"[step 0] val L1/elem — without PostNet: {base_before:.5f}  "
          f"with zero-init PostNet: {base_after:.5f}  (these must match)")
    best_after = base_after

    step = 0
    it = iter(train_loader)
    while step < args.steps:
        try:
            pred, gt, mask = next(it)
        except StopIteration:
            it = iter(train_loader)
            pred, gt, mask = next(it)

        pred, gt, mask = pred.to(device), gt.to(device), mask.to(device)
        loss = masked_l1(pred + net(pred), gt, mask)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.eval_every == 0:
            before, after = evaluate(net, val_loader, device)
            gain = (before - after) / before * 100
            print(f"[step {step}] train {loss.item():.5f}  "
                  f"val before {before:.5f}  after {after:.5f}  gain {gain:+.2f}%")
            if after < best_after:
                best_after = after
                torch.save({"postnet": net.state_dict(), "step": step,
                            "val_after": after, "val_before": before}, args.out)
                print(f"  saved {args.out}")

    total_gain = (base_before - best_after) / base_before * 100
    print(f"\nBest val L1/elem: {best_after:.5f} vs {base_before:.5f} without PostNet "
          f"({total_gain:+.2f}%)")
    print("Under 1%: not worth keeping. Over 2%: enable it. In between: listen.")


if __name__ == "__main__":
    main()
