"""
Recover the phoneme vocab best.pt was trained with and remap its embedding
table onto the current vocab.

  python recover_vocab.py build  --out old_vocab.json
      Rebuild the vocab from metadata.csv with the LOCAL espeak (run on the
      machine that did the original preprocessing). Text only, no audio.

  python recover_vocab.py check  --ckpt best.pt --vocab old_vocab.json
      Verify a candidate vocab against the checkpoint: common phones must sit
      on trained-looking rows, rows past the vocab must look untrained.

  python recover_vocab.py remap  --ckpt best.pt --old old_vocab.json \
                                 --new data/processed/phoneme_vocab.json --out best_remapped.pt
      Copy embedding rows by phoneme string from old ids to new ids.
      Add --reset-embedding to skip the old vocab entirely and re-init the
      whole table (fallback when the old vocab cannot be recovered).
"""

import os
import sys
import json
import argparse
import torch

if sys.platform == "win32":
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY",
                          r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

COMMON = ["θ", "ᵻ", "s", "t", "n", "ð", "ˈɪ", "ə"]


def embed_key(sd):
    keys = [k for k in sd if "embed" in k and k.endswith("weight")]
    if len(keys) != 1:
        raise RuntimeError(f"expected one embedding weight, found {keys}")
    return keys[0]


def load_sd(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt, ckpt.get("model", ckpt)


def cmd_build(args):
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from config import paths

    metadata = paths.data_root / "metadata.csv"
    texts = [l.strip().split("|")[2] for l in open(metadata, encoding="utf-8") if l.strip()]
    print(f"{len(texts)} lines from {metadata}")

    backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
    sep = Separator(phone=" ", word="| ", syllable="")
    phones = set()
    for i in range(0, len(texts), 256):
        for seq in backend.phonemize(texts[i:i + 256], separator=sep):
            phones.update(p for p in seq.replace("|", "").split() if p.strip())

    vocab = {"<pad>": 0, "<sil>": 1}
    for p in sorted(phones):
        vocab.setdefault(p, len(vocab))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(vocab)} entries")


def cmd_check(args):
    _, sd = load_sd(args.ckpt)
    norms = sd[embed_key(sd)].norm(dim=1)
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    n = len(vocab)
    beyond = norms[n:]
    if len(beyond) == 0:
        raise RuntimeError("vocab fills the whole table; cannot estimate untrained baseline")
    base = beyond.median().item()
    thresh = 1.3 * base
    trained = (norms[:n] > thresh)
    print(f"vocab size {n}  untrained baseline {base:.2f}  threshold {thresh:.2f}")
    print(f"trained rows inside vocab: {int(trained.sum())}/{n}")
    print(f"max trained id: {int(trained.nonzero().max())}")
    ok = True
    for p in COMMON:
        if p not in vocab:
            print(f"  {p!r}: not in vocab"); ok = False; continue
        v = norms[vocab[p]].item()
        flag = "ok" if v > thresh else "UNTRAINED"
        if v <= thresh:
            ok = False
        print(f"  {p!r} id {vocab[p]:3d} norm {v:.2f}  {flag}")
    untrained_inside = [k for k, i in vocab.items() if i > 1 and norms[i] <= thresh]
    print(f"untrained rows inside vocab (expect rare phones only): {untrained_inside}")
    print("VERDICT:", "vocab matches checkpoint" if ok else "vocab does NOT match checkpoint")


def cmd_remap(args):
    ckpt, sd = load_sd(args.ckpt)
    key = embed_key(sd)
    W = sd[key]
    new = json.load(open(args.new, encoding="utf-8"))

    if args.reset_embedding:
        W_new = torch.empty_like(W)
        torch.nn.init.normal_(W_new, mean=0.0, std=W.shape[1] ** -0.5)
        W_new[new["<pad>"]] = 0
        print("embedding table re-initialised; all phones learn from scratch")
    else:
        old = json.load(open(args.old, encoding="utf-8"))
        W_new = W.clone()
        hit = 0
        for p, nid in new.items():
            if p in old:
                W_new[nid] = W[old[p]]
                hit += 1
        missing = [p for p in new if p not in old]
        print(f"remapped {hit}/{len(new)} rows; left at init: {missing}")

    sd[key] = W_new
    if "model" in ckpt:
        ckpt["model"] = sd
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", default="old_vocab.json")

    c = sub.add_parser("check")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--vocab", required=True)

    r = sub.add_parser("remap")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--new", required=True)
    r.add_argument("--old", default=None)
    r.add_argument("--out", default="best_remapped.pt")
    r.add_argument("--reset-embedding", action="store_true")

    args = ap.parse_args()
    if args.cmd == "remap" and not args.reset_embedding and not args.old:
        ap.error("remap needs --old, or --reset-embedding")
    {"build": cmd_build, "check": cmd_check, "remap": cmd_remap}[args.cmd](args)