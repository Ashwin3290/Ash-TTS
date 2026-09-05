# Evaluation

100 LJSpeech validation utterances. PostNet off. Device: cuda:0.

| Metric | Value |
|---|---|
| WER (Whisper base.en) | 14.5% |
| CER | 8.2% |
| WER on real audio (floor) | 4.9% |
| MCD (DTW) | 7.54 dB |
| Real-time factor | 0.076 |

WER is Whisper's transcription of synthesized audio against the input text, so it reflects Whisper's own error rate as well as the model's; compare against the floor row where present. MCD is computed against the real recording of the same sentence with DTW alignment, so it also penalises legitimate differences in phrasing and speaking rate.
