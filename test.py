import torch, json
sd = torch.load("checkpoints/fastspeech2/best.pt", map_location="cpu")["model"]
key = [k for k in sd if "embed" in k and k.endswith("weight")][0]
print(key)
norms = sd[key].norm(dim=1)
active = (norms > 0.5 * norms.max()).nonzero().flatten().tolist()
print("rows with trained-looking norms:", len(active), "max id:", max(active))
vocab = json.load(open("data/processed/phoneme_vocab.json"))
print("vocab size:", len(vocab))