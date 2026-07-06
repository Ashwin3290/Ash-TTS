# debug_failures.py
import os, sys
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

import torch, torchaudio, numpy as np
from torchaudio.pipelines import MMS_FA as BUNDLE
from align import text_to_char_tokens, get_phonemizer, BLANK_IDX
from config import paths
import json

device  = torch.device("cuda")
model   = BUNDLE.get_model().to(device)
model.eval()

text_map = {}
for line in open("data/LJSpeech-1.1/metadata.csv", encoding="utf-8"):
    parts = line.strip().split("|")
    if len(parts) >= 3:
        text_map[parts[0]] = parts[2]

# test 20 failing utterances
from pathlib import Path
durs = list(Path('data/processed/duration').glob('*.npy'))
failures = [f.stem for f in durs if len(set(np.load(f).tolist())) <= 2][:20]

errors = {}
for utt_id in failures:
    text = text_map.get(utt_id, "")
    wav_path = f"data/LJSpeech-1.1/wavs/{utt_id}.wav"

    token_ids, word_of_token, words = text_to_char_tokens(text)
    if not token_ids:
        errors['no_tokens'] = errors.get('no_tokens', 0) + 1
        continue

    waveform, sr = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, BUNDLE.sample_rate)
    waveform = waveform.mean(dim=0, keepdim=True).to(device)
    with torch.inference_mode():
        emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)

    try:
        token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emissions, token_ids_t.unsqueeze(0), blank=BLANK_IDX
        )
        spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])
        if len(spans) != len(token_ids):
            errors['span_mismatch'] = errors.get('span_mismatch', 0) + 1
            print(f"{utt_id}: spans={len(spans)} tokens={len(token_ids)} frames={emissions.shape[1]}")
        else:
            errors['length_mismatch'] = errors.get('length_mismatch', 0) + 1
            print(f"{utt_id}: aligned ok but phone count mismatch")
    except Exception as e:
        errors[str(e)[:50]] = errors.get(str(e)[:50], 0) + 1
        print(f"{utt_id}: EXCEPTION {e}")

print(f"\nError summary: {errors}")