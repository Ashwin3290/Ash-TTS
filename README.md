# Ash-TTS

A from-scratch text-to-speech pipeline: FastSpeech2 acoustic model, standalone PostNet refiner, HiFi-GAN vocoder. Trained on LJSpeech-1.1. English, single speaker, 22.05 kHz, 59.5 M parameters end to end.

Phoneme durations come from a pretrained wav2vec2 CTC forced aligner rather than the Montreal Forced Aligner, and pause tokens are derived from the acoustic signal rather than from punctuation. Both are explained below.

**Weights and audio samples:** [huggingface.co/Ashwin-C9/Ash-TTS](https://huggingface.co/Ashwin-C9/Ash-TTS)

| | WER | CER | MCD (DTW) | RTF |
|---|---|---|---|---|
| PostNet on | 8.8% | 5.1% | 7.02 dB | 0.080 |
| PostNet off | 14.5% | 8.2% | 7.54 dB | 0.076 |
| Real recordings (floor) | 4.9% | — | — | — |

100 held-out utterances, Whisper `base.en`, RTX 3050. The floor row is the same ASR on the original LJSpeech audio, so the model adds about 3.9 points of transcription error with the PostNet and 9.6 without.

## Quick start

```bash
git clone https://github.com/Ashwin3290/Ash-TTS && cd Ash-TTS
pip install -r requirements.txt
sudo apt-get install espeak-ng      # phonemizer shells out to it
```

On Windows, install espeak-ng from its releases page; the scripts look for the DLL at the default install path.

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

Controls: `speed`, `pitch`, `energy`, all default 1.0.

`ash_tts.pt` is a single bundle holding all three stages plus the phoneme vocabulary and normalisation statistics. They ship together because they are coupled: the vocoder is fine-tuned on the mel distribution this acoustic model and PostNet produce, and the vocabulary must be the exact one whose index order the embedding table was trained on. Mismatched pairings load without error and sound wrong.

To run the stages separately from raw checkpoints, use `inference.py`, which takes `--fs2-ckpt`, `--hifi-ckpt` and `--postnet-ckpt`.

## Architecture

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

| Stage | Parameters | Source |
|---|---|---|
| FastSpeech2 | ~41.3 M | trained here |
| PostNet | 4.35 M | trained here |
| HiFi-GAN V1 generator | ~13.9 M | fine-tuned from jik876/hifi-gan LJ_V1 |

Audio config: 22050 Hz, 1024-point FFT, 256 hop, 80 mel bins. Model: `d_model` 256, 2 attention heads, `d_ff` 1024.

### Durations without MFA

FastSpeech2 needs a duration for every phoneme. The usual source is the Montreal Forced Aligner, which means installing Kaldi, training an acoustic model on the corpus, and reconciling MFA's ARPAbet dictionary with whatever phone set the frontend produces.

`align.py` uses `facebook/wav2vec2-lv-60-espeak-cv-ft` with torchaudio's `forced_align` instead. The phoneme sequence already comes from espeak-ng and that model emits espeak-compatible labels, so there is no cross-dictionary mapping to get wrong. Frame boundaries are accumulated rather than rounded per interval, so durations sum exactly to the mel frame count — a per-interval rounding scheme drifts by a few frames per utterance, which puts a floor under mel loss that training cannot get past.

All 12,910 utterances aligned successfully, with no fallbacks.

This is not a new idea. charsiu (Zhu et al., ICASSP 2022), the EveryVoice toolkit, and DailyTalk all replace MFA with a CTC aligner; this is an independent implementation.

### Pause tokens from the audio

CTC emissions are peaky: the aligner marks a phone on a handful of frames and leaves the surrounding frames blank. Attributing those blank gaps to the preceding phone stretches phones across real pauses and leaves the model with no token that means "silence here." Speech generated that way runs flat and evenly paced, with no phrasing.

Instead, an inter-phone gap becomes a `<sil>` token when it is both long enough — at least 6 emission frames, about 120 ms — and quiet enough, mean RMS below 5% of the utterance maximum. The energy gate earns its place: stop closures for /p/, /t/ and /k/ are genuinely silent and clear the duration threshold on their own, so a duration-only rule inserts pauses inside words.

On LJSpeech this yields 1.65 `<sil>` tokens per utterance, present in 81% of utterances, median 28 frames (~325 ms). Because they come from the audio, they land where the speaker actually paused rather than wherever a comma happens to be.

At inference, `inference.py` inserts `<sil>` at commas, semicolons, colons and sentence ends. Without that step punctuation has no effect on the output at all — the model learned what `<sil>` sounds like and would simply never receive one.

### PostNet as a separate stage

The PostNet is not part of `FastSpeech2`. It is trained on its own by `train_postnet.py` against cached (predicted mel, ground-truth mel) pairs from `generate_mels.py`, with the acoustic model frozen and out of the graph, so a run takes minutes rather than hours.

It is also a fairer setting than joint training. `generate_mels.py` uses ground-truth durations for frame alignment but *predicted* pitch and energy, so the refiner learns to correct variance-predictor error as well as regression blur — the same conditions it faces at inference.

Validation mel L1 per element: 0.2824 without, 0.2392 with. A 15.3% reduction, and it shows up perceptually — the WER table above is the same model with and without it.

Because the vocoder is fine-tuned on FastSpeech2's mel distribution, adding the PostNet shifts that distribution and the vocoder needs re-tuning on the sharpened mels. `generate_mels.py --postnet-ckpt` produces them.

### Chunking long input

LJSpeech clips are capped at 10 seconds, giving phoneme-sequence lengths with a median of 70, p90 of 98, and a maximum of 132. Past roughly 100 phonemes the duration predictor and decoder attention are extrapolating and the tail of the output degrades.

Input is packed into chunks of at most 100 phonemes, split at `<sil>` where a pause is available and at word boundaries when a clause is too long to split at a pause, then vocoded separately and joined with a 5 ms crossfade. Splitting only at clause boundaries is not enough: a single long unpunctuated clause would produce one oversized chunk regardless of the cap.

## Training

```bash
python download_data.py                        # LJSpeech-1.1, ~2.6 GB
python preprocess.py --workers 8               # mel, f0, energy, phoneme ids, vocab
python align.py --overwrite                    # CTC durations and <sil> tokens

python train_fastspeech.py                     # acoustic model
python generate_mels.py                        # predicted mels for the next two stages
python train_postnet.py --steps 30000          # residual refiner
python generate_mels.py --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt
python train_hifigan.py --init-g pretrained_hifigan/generator_v1 \
                        --init-d pretrained_hifigan/do_v1 \
                        --mel-dir mel_pred     # vocoder fine-tune
```

`generator_v1` and the universal `do_*` discriminator checkpoint come from the pretrained models folder linked in the [jik876/hifi-gan](https://github.com/jik876/hifi-gan) README. Put them in `pretrained_hifigan/`.

Then bundle the three stages for release:

```bash
python ash_tts.py bundle \
    --fs2-ckpt checkpoints/fastspeech2/best.pt \
    --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt \
    --hifi-ckpt checkpoints/hifigan/g_best.pt
```

Things worth knowing before you spend GPU hours on this:

**Set thread limits before preprocessing.** `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. On a machine reporting a high core count, each worker otherwise spawns a full OpenBLAS thread pool and the process hits its pid limit partway through.

**Don't select the vocoder checkpoint by `val/mel_l1`.** That metric compares vocoded output against the *ground-truth* mel, so it is dominated by FastSpeech2's error and barely moves regardless of vocoder quality. Use it to confirm nothing diverged; select by listening.

**The phoneme vocabulary is load-bearing.** It maps IPA strings to embedding row indices, and it is built from whatever espeak-ng emits. A different espeak-ng version produces a different phone set, which shifts every index after the first difference — the checkpoint then loads without error and produces unintelligible speech. `assets/phoneme_vocab.json` is the vocabulary these weights were trained with; keep it.

## Repository layout

```
config.py                   all hyperparameters, single import point
preprocess.py               audio features, phoneme ids, vocab, manifests
align.py                    CTC forced alignment, durations and <sil> extraction
data/dataset.py             batching and padding
model/                      encoder, variance adaptor, decoder, PostNet
vocoder/                    HiFi-GAN generator, MPD/MSD discriminators, losses

train_fastspeech.py         acoustic model training
train_postnet.py            standalone PostNet training
train_hifigan.py            vocoder training and fine-tuning
generate_mels.py            predicted mels for vocoder fine-tuning

ash_tts.py                  the release bundle: one class, one checkpoint
inference.py                text → wav from separate checkpoints
test_inference.py           single-utterance check with mel plots

run_eval.py                 WER, CER, MCD and RTF over the validation split
make_samples.py             demo clips for the model card
publish_hf.py               upload the bundle and card to the Hub

assets/                     phoneme vocabulary and normalisation stats
eval_results/               the numbers behind the table above
examples/                   applications built on the model
download_data.py            LJSpeech
download_processed.py       cached features from the Hub, to skip preprocessing
download_latest_model.py    training checkpoints from the Hub, for resuming
push_to_hf.py               upload training checkpoints
```

`run_eval.py` needs `openai-whisper`, `jiwer`, `pyworld` and `pysptk`, which are not in `requirements.txt` since nothing else uses them.

## Evaluation notes

Per-utterance results are in [`eval_results/results.json`](eval_results/results.json), with the aggregate in [`eval_results/summary.md`](eval_results/summary.md). The summary file covers the PostNet-on condition; the PostNet-off column in the table above came from a separate run with `--no-postnet`.

The 8.8% WER is pessimistic for a scoring reason worth stating. Manual review of the worst-scoring utterances found most were not synthesis failures: Whisper writes digits where LJSpeech's transcripts spell numbers out, so a sentence rendered perfectly scores 0.375 because "three thousand, nine hundred eighty-five" was transcribed as "3,985". Proper nouns account for most of the rest. The distribution is healthier than the mean suggests — median WER 4.9%, 44% of utterances transcribed exactly, only 5% above 0.3.

No MOS study has been run, so there is no listener-rated naturalness figure.

## Limitations

Single speaker, English only, LJSpeech's audiobook reading style; quality drops outside that domain. One `<sil>` token, so pause *length* is inferred from context and cannot be set directly. Rare words get stretched and can slur — reduplications like "higgledy-piggledy" that appear once or twice in the corpus are a visible failure case. No MFA baseline was trained under matched conditions, so this makes no claim that CTC alignment beats MFA, only that it is sufficient here and much easier to set up.

## Acknowledgements

FastSpeech2 (Ren et al., 2021); HiFi-GAN (Kong et al., 2020), with the implementation and LJ_V1 / universal weights from [jik876/hifi-gan](https://github.com/jik876/hifi-gan); the wav2vec2 espeak phone recognizer (Xu et al., 2021) via torchaudio's forced alignment API; espeak-ng and phonemizer; LJSpeech-1.1 by Keith Ito.

## License

MIT, for both the code and the weights. LJSpeech-1.1 is public domain. The HiFi-GAN generator is fine-tuned from jik876/hifi-gan, which is MIT-licensed.