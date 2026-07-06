import torch
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from config import audio as acfg, paths, model as mcfg
from model.fastspeech2 import FastSpeech2
import sys
import os
# Windows espeak path (needed for phonemizer in this script)
if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vocab
vocab = json.load(open(paths.processed_dir / "phoneme_vocab.json"))

# load model
model = FastSpeech2().to(device)
ckpt = torch.load("checkpoints/fastspeech2/latest.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

stats = json.load(open(paths.processed_dir / "stats.json"))
model.variance_adaptor.set_stats(**stats)

print(f"Loaded checkpoint at step {ckpt['step']}")

# phonemize
text = "The quick brown fox jumps over the lazy dog."
backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True)
sep = Separator(phone=' ', word='| ', syllable='')
ph_str = backend.phonemize([text], separator=sep)[0]
phones = [p for p in ph_str.replace('|', '').split() if p.strip()]
ids = [vocab.get(p, 0) for p in phones]
print(f"Phonemes: {phones}")

ph_tensor = torch.tensor(ids).long().unsqueeze(0).to(device)
ph_lens   = torch.tensor([len(ids)]).to(device)

with torch.no_grad():
    mel_pred, _, _, _, mel_lens = model(ph_tensor, ph_lens)

mel = mel_pred[0, :mel_lens[0].item()].cpu().numpy()  # (T, 80)
print(f"Mel shape: {mel.shape} — {mel.shape[0] * acfg.hop_length / acfg.sample_rate:.2f}s")

# denormalize
mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min
mel_linear = np.exp(mel).T  # (80, T) — librosa expects (n_mels, T)

# Griffin-Lim
wav = librosa.feature.inverse.mel_to_audio(
    mel_linear,
    sr=acfg.sample_rate,
    n_fft=acfg.n_fft,
    hop_length=acfg.hop_length,
    win_length=acfg.win_length,
    n_iter=60
)

sf.write("test_output.wav", wav, acfg.sample_rate)
print("Saved: test_output.wav")



import matplotlib.pyplot as plt
import numpy as np

# plot predicted mel
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(mel.T, aspect='auto', origin='lower')
plt.title('FastSpeech2 predicted mel')
plt.colorbar()

# plot a ground truth mel for comparison
gt_mel = np.load('data/processed/mel/LJ001-0001.npy')
plt.subplot(1, 2, 2)
plt.imshow(gt_mel.T, aspect='auto', origin='lower')
plt.title('Ground truth mel')
plt.colorbar()

plt.tight_layout()
plt.savefig('mel_comparison.png')
print('Saved mel_comparison.png')