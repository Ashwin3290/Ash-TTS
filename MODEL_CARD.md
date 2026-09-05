---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - fastspeech2
  - hifi-gan
  - speech-synthesis
  - ljspeech
datasets:
  - lj_speech
---

# Ash-TTS

A from-scratch FastSpeech2 + HiFi-GAN text-to-speech system for English, trained on LJSpeech-1.1. Single speaker, 22.05 kHz, 59.5 M parameters end to end.

Two things distinguish it from a standard reimplementation: phoneme durations come from a pretrained wav2vec2 CTC forced aligner instead of the Montreal Forced Aligner, and pause tokens are derived from the acoustic signal instead of from punctuation.

Code: [github.com/Ashwin3290/Ash-TTS](https://github.com/Ashwin3290/Ash-TTS)

## Samples

**A short introduction.**

> Hello. This is a text to speech model I trained from scratch.

<audio controls src="https://huggingface.co/Ashwin-C9/Ash-TTS/resolve/main/samples/short.wav"></audio>

**Nine clause breaks. Each comma and semicolon produces a real pause, because the model has a silence token it learned from the audio.**

> When the printer arrived, late, tired, and out of paper, he found the shop locked, the lights off, and the owner gone; nobody, it seemed, had expected him.

<audio controls src="https://huggingface.co/Ashwin-C9/Ash-TTS/resolve/main/samples/pauses.wav"></audio>

**Three short sentences in a row, with the pause at each full stop.**

> Is that really what happened? I find it hard to believe. Are you certain?

<audio controls src="https://huggingface.co/Ashwin-C9/Ash-TTS/resolve/main/samples/questions.wav"></audio>

**Plain narrative prose, no unusual vocabulary.**

> She set the letter down without opening it. Outside, the rain had stopped, and the street was quiet again. For a long moment she did not move.

<audio controls src="https://huggingface.co/Ashwin-C9/Ash-TTS/resolve/main/samples/narrative.wav"></audio>


## Usage

Everything ships in one file, `ash_tts.pt`: all three stages, the phoneme vocabulary, and the normalisation statistics. They are bundled together because they are coupled — the vocoder is fine-tuned on the mel distribution this specific acoustic model and PostNet produce, and the vocabulary must be the exact one whose index order the embedding table was trained on. Mismatched pairings load without error and sound wrong.

```bash
git clone https://github.com/Ashwin3290/Ash-TTS && cd Ash-TTS
pip install -r requirements.txt
apt-get install espeak-ng            # phonemizer shells out to it
```

```python
from huggingface_hub import hf_hub_download
from ash_tts import AshTTS

tts = AshTTS.from_pretrained(hf_hub_download("Ashwin-C9/Ash-TTS", "ash_tts.pt"))
tts.save_wav("When the printer arrived, late and out of paper, "
             "he found the shop locked.", "out.wav")
```

Or from the command line:

```bash
python ash_tts.py speak --model ash_tts.pt --text "Hello there." --output out.wav
```

Use `AshTTS`, not the stages directly. It handles `<sil>` insertion at punctuation, chunking of long input, and mel denormalisation before the vocoder. Omitting any of the three degrades or breaks the output.

Controls: `speed`, `pitch`, `energy`, all default 1.0. `use_postnet=False` disables the refiner, which is worse (see below) but useful for comparison.

## How it works

```
text
  ↓  espeak-ng phonemization; <sil> inserted at clause and sentence breaks
phoneme ids (111-token IPA inventory)
  ↓  FastSpeech2 — 4-layer FFT encoder, variance adaptor, 4-layer FFT decoder
mel, 80 bins, normalised to [-1, 1]
  ↓  PostNet — 5 Conv1d layers, residual
refined mel
  ↓  denormalise to natural-log scale
  ↓  HiFi-GAN V1 generator
waveform, 22.05 kHz
```

| Stage | Parameters |
|---|---|
| FastSpeech2 | ~41.3 M |
| PostNet | 4.35 M |
| HiFi-GAN V1 generator | ~13.9 M |
| **Total** | **59.5 M** |

### Durations without MFA

FastSpeech2 needs a duration for every phoneme. The usual source is the Montreal Forced Aligner, which means installing Kaldi, training an acoustic model on the corpus, and reconciling MFA's ARPAbet dictionary with whatever phone set the frontend produces.

This uses `facebook/wav2vec2-lv-60-espeak-cv-ft` with torchaudio's `forced_align` instead. The phoneme sequence already comes from espeak-ng and that model emits espeak-compatible labels, so no cross-dictionary mapping is needed. Frame boundaries are accumulated rather than rounded per interval, so durations sum exactly to the mel frame count. All 12,910 utterances aligned successfully with no fallbacks.

### Pause tokens from the audio

CTC emissions are peaky: the aligner marks a phone on a handful of frames and leaves the surrounding frames blank. Attributing those blank gaps to the preceding phone stretches phones across real pauses and leaves the model with no token meaning "silence here." Speech generated that way runs flat and evenly paced, with no phrasing.

Here a gap becomes `<sil>` only when it is both long enough — at least 6 emission frames, about 120 ms — and quiet enough, mean RMS below 5% of the utterance maximum. The energy gate is necessary rather than decorative: stop closures for /p/, /t/ and /k/ are genuinely silent and clear the duration threshold on their own.

On LJSpeech this yields 1.65 `<sil>` tokens per utterance, present in 81% of utterances, median duration 28 frames (~325 ms). Because they are derived from the audio, they land where the speaker actually paused rather than wherever a comma happens to be.

### PostNet as a separate stage

The PostNet is trained on its own against cached (predicted mel, ground-truth mel) pairs, with FastSpeech2 frozen and out of the graph. The pairs use ground-truth durations for frame alignment but *predicted* pitch and energy, so the refiner learns to correct variance-predictor error as well as regression blur — the same conditions it faces at inference.

Validation mel L1 per element: 0.2824 without, 0.2392 with. A 15.3% reduction.

### Chunking long input

LJSpeech clips are capped at 10 seconds, giving phoneme-sequence lengths with a median of 70, p90 of 98, and a maximum of 132. Past roughly 100 phonemes the duration predictor and decoder attention are extrapolating and the tail of the output degrades.

Input is packed into chunks of at most 100 phonemes, split at `<sil>` where a pause is available and at word boundaries when a clause is too long to split at a pause, then vocoded separately and joined with a 5 ms crossfade. Splitting only at clause boundaries is not enough: one long unpunctuated clause would produce a single oversized chunk regardless of the cap.

## Evaluation

100 held-out LJSpeech validation utterances. WER and CER are Whisper `base.en` transcriptions of synthesized audio scored against the input text; MCD is mel-cepstral distortion against the real recording with DTW alignment, order 24, c0 dropped. RTF measured on an RTX 3050.

| Metric | PostNet on | PostNet off |
|---|---|---|
| WER | 8.8% | 14.5% |
| CER | 5.1% | 8.2% |
| MCD (DTW) | 7.02 dB | 7.54 dB |
| Real-time factor | 0.080 | 0.076 |
| WER on the real recordings (floor) | 4.9% | 4.9% |

Read these against the floor. Whisper makes its own errors, so the same ASR on the original LJSpeech audio scores 4.9%; the model therefore adds about 3.9 points with the PostNet and 9.6 without, meaning the refiner removes roughly 60% of the intelligibility gap. That comparison is the point of the table — the absolute numbers are less informative than the deltas.

The absolute WER is also pessimistic for a scoring reason. Manual review of the worst-scoring utterances found most were not synthesis failures: Whisper writes digits where LJSpeech's transcripts spell numbers out, so a sentence rendered perfectly can score 0.375 because "three thousand, nine hundred eighty-five" was transcribed "3,985". Proper nouns account for most of the rest (Mackintosh/McIntosh, Neild/Nield). The distribution is healthier than the mean suggests: median WER 4.9%, 44% of utterances transcribed exactly, 5% above 0.3.

No MOS study has been run, so there is no listener-rated naturalness figure. Judge that from the samples.

## Training

| | |
|---|---|
| Data | LJSpeech-1.1; 12,910 utterances after a 10 s length filter; 12,264 train / 646 val |
| Frontend | espeak-ng via phonemizer, IPA with stress marks, 111 tokens |
| Durations | wav2vec2 CTC forced alignment; 12,910 real alignments, 0 fallbacks |
| Acoustic model | trained to convergence, then warm-started for 60k steps on the `<sil>` data; best validation mel L1 0.0989 at step 28k |
| PostNet | 39.5k steps, AdamW, lr 2e-4 |
| Vocoder | fine-tuned from the official LJ_V1 checkpoint for 34k steps on predicted mels, with the official universal discriminators as initialisation |
| Hardware | consumer cloud GPUs (RTX 4060 Ti / 5070 Ti class); RTX 3050 locally |

One detail worth repeating for anyone fine-tuning HiFi-GAN: initialise the discriminators as well as the generator. Randomly initialised discriminators against a pretrained generator drive adversarial loss up steadily and produce a shrill, metallic artifact.

## Limitations

- Single speaker, English only, in LJSpeech's audiobook reading style. Quality drops outside that domain.
- One `<sil>` token, so pause *length* is inferred from context and cannot be set directly.
- Rare and out-of-distribution words are stretched and can slur; reduplications like "higgledy-piggledy" that appear once or twice in the corpus are a visible failure case.
- Text beyond ~100 phonemes is chunked. A chunk boundary that has to fall mid-clause is crossfaded but may still be faintly audible.
- No MFA baseline was trained under matched conditions, so this makes no claim that CTC alignment beats MFA — only that it is sufficient here and much easier to set up.
- The vocoder is fine-tuned on this acoustic model's mel distribution and will not transfer cleanly to another one.

## Related work

Replacing MFA with a CTC aligner for non-autoregressive TTS is established practice, not a new idea: charsiu (Zhu et al., ICASSP 2022), the EveryVoice toolkit, and DailyTalk all do it. This is an independent implementation. The energy-gated pause handling is an engineering response to peaky CTC alignment rather than a novel technique.

## Citation

```bibtex
@software{ash_tts,
  author = {Ashwin C},
  title  = {Ash-TTS: FastSpeech2 and HiFi-GAN with CTC forced alignment and acoustic pause tokens},
  year   = {2026},
  url    = {https://github.com/Ashwin3290/Ash-TTS}
}
```

## Acknowledgements

FastSpeech2 (Ren et al., 2021); HiFi-GAN (Kong et al., 2020), implementation and LJ_V1 / universal weights from [jik876/hifi-gan](https://github.com/jik876/hifi-gan); the wav2vec2 espeak phone recognizer (Xu et al., 2021) via torchaudio's forced alignment API; espeak-ng and phonemizer; LJSpeech-1.1 by Keith Ito.