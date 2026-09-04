import torch, json
sd = torch.load("checkpoints/fastspeech2/best.pt", map_location="cpu")["model"]
key = [k for k in sd if "embed" in k and k.endswith("weight")][0]
norms = sd[key].norm(dim=1)
vocab = json.load(open("data/processed/phoneme_vocab.json"))
inv = {v: k for k, v in vocab.items()}
for i in range(0, 125):
    print(i, repr(inv.get(i, "-")), f"{norms[i].item():.3f}")