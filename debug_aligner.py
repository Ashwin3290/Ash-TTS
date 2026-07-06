import os, sys
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

import torch, torchaudio, numpy as np
from torchaudio.pipelines import MMS_FA as BUNDLE
from align import text_to_char_tokens, phonemize_words, get_phonemizer
from config import audio as acfg

ALIGN_SR  = BUNDLE.sample_rate
LABELS    = BUNDLE.get_labels()
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
BLANK_IDX = LABEL2IDX.get('|', 0)
W2V_FRAME_SEC = 320 / ALIGN_SR
MEL_FRAME_SEC = acfg.hop_length / acfg.sample_rate

device  = torch.device("cuda")
model   = BUNDLE.get_model().to(device)
model.eval()
backend = get_phonemizer()

text     = "Printing, in the only sense with which we are at present concerned,"
wav_path = "data/LJSpeech-1.1/wavs/LJ001-0001.wav"

token_ids, word_of_token, words = text_to_char_tokens(text)
print(f"Words: {words}")
print(f"Token ids ({len(token_ids)}): {token_ids[:20]}")

waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, ALIGN_SR)
waveform = waveform.mean(dim=0, keepdim=True).to(device)
with torch.inference_mode():
    emissions, _ = model(waveform)
emissions = torch.log_softmax(emissions, dim=-1)

token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
aligned_tokens, scores = torchaudio.functional.forced_align(
    emissions, token_ids_t.unsqueeze(0), blank=BLANK_IDX
)
spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])
print(f"Spans count: {len(spans)}")
print(f"First 10 spans: {spans[:10]}")

# FIXED: use span_idx not span.token
n_words    = len(words)
word_start = [None] * n_words
word_end   = [None] * n_words

for span_idx, span in enumerate(spans):
    if span_idx >= len(word_of_token):
        continue
    w_idx = word_of_token[span_idx]
    if word_start[w_idx] is None:
        word_start[w_idx] = span.start
    word_end[w_idx] = span.end

# fill missing
for i in range(n_words):
    if word_start[i] is None:
        prev_end = word_end[i-1] if i > 0 and word_end[i-1] is not None else 0
        word_start[i] = prev_end
        word_end[i]   = prev_end + 1

print(f"\nWord spans:")
for i, w in enumerate(words):
    print(f"  {w}: {word_start[i]} -> {word_end[i]}")

# phonemize words and build durations
word_phones = phonemize_words(words, backend)
durations = []
for w_idx, w_phones in enumerate(word_phones):
    start_sec   = word_start[w_idx] * W2V_FRAME_SEC
    end_sec     = word_end[w_idx]   * W2V_FRAME_SEC
    word_frames = max(len(w_phones), round((end_sec - start_sec) / MEL_FRAME_SEC))
    n_ph = len(w_phones)
    base = word_frames // n_ph
    rem  = word_frames % n_ph
    for j in range(n_ph):
        durations.append(base + (1 if j < rem else 0))

print(f"\nDurations ({len(durations)}): {durations[:20]}")
print(f"Unique values: {len(set(durations))}")
print(f"Total frames: {sum(durations)}")