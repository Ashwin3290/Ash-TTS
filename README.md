# Ash-TTS

A from-scratch implementation of a non-autoregressive text-to-speech pipeline: FastSpeech2 acoustic model, standalone PostNet refiner, HiFi-GAN vocoder. Trained on LJSpeech-1.1.

Phoneme durations come from a pretrained wav2vec2 CTC forced aligner rather than the Montreal Forced Aligner, and pause tokens are derived from the acoustic signal rather than from punctuation. Both choices are described below.

**Model weights:** [Ashwin-C9/Ash-TTS](https://huggingface.co/Ashwin-C9/Ash-TTS)
**Demo:** TODO — link the Space once it is up

## Samples

TODO — embed or link 4–6 clips, including one multi-sentence paragraph.

## Quick start

```bash
pip install -r requirements.txt
apt-get install espeak-ng          # or the Windows installer; phonemizer needs it

# fetch weights + assets from the Hub into checkpoints/ and assets/
python download_latest_model.py

python inference.py \
    --text "When the printer arrived, late and out of paper, he found the shop locked." \
    --fs2-ckpt checkpoints/fastspeech2/best.pt \
    --hifi-ckpt checkpoints/hifigan/g_best.pt \
    --postnet-ckpt checkpoints/fastspeech2/postnet_only.pt \
    --output output.wav
```

Inference needs `assets/phoneme_vocab.json` and `assets/stats.json`. They ship with the weights; a full preprocessing run is only required for training.

Control knobs: `--speed`, `--pitch`, `--energy`, all defaulting to 1.0.

## Architecture

```
text
  ↓  espeak-ng phonemization, <sil> inserted at clause and sentence breaks
phoneme ids
  ↓  FastSpeech2: 4-layer FFT encoder → variance adaptor → 4-layer FFT decoder
mel (80 bins, normalised to [-1, 1])
  ↓  PostNet: 5 Conv1d layers, residual, trained separately
refined mel
  ↓  denormalise to natural-log scale
  ↓  HiFi-GAN V1 generator
waveform, 22.05 kHz
```

| Component | Parameters | Source |
|---|---|---|
| FastSpeech2 | TODO | trained here |
| PostNet | 4.35 M | trained here |
| HiFi-GAN V1 generator | 13.9 M | fine-tuned from jik876/hifi-gan LJ_V1 |

Key config: 22050 Hz, 1024-point FFT, 256 hop, 80 mel bins, `d_model` 256, 2 attention heads, 111-entry IPA phoneme vocabulary.

### Duration alignment without MFA

FastSpeech2 needs a phoneme duration for every token. The usual source is the Montreal Forced Aligner, which means installing Kaldi, training an acoustic model on the corpus, and reconciling MFA's ARPAbet dictionary with whatever phone set the frontend produces.

`align.py` uses `facebook/wav2vec2-lv-60-espeak-cv-ft` with torchaudio's `forced_align` instead. The phoneme sequence already comes from espeak-ng, and that model emits espeak-compatible phone labels, so no dictionary mapping is needed. Frame boundaries are accumulated and distributed so durations sum exactly to the mel frame count, which a per-interval rounding scheme does not guarantee.

Related work does the same thing: charsiu (Zhu et al., ICASSP 2022), the EveryVoice toolkit, and DailyTalk all replace MFA with a CTC aligner. This is an independent implementation, not a new method.

### Pause tokens from the audio

CTC emissions are peaky: the aligner marks a phone on a handful of frames and leaves the surrounding frames blank. Attributing those blank gaps to the preceding phone stretches phones across real pauses, and the model then has no token that means "silence here." Speech generated that way runs flat and even, with no phrasing.

Instead, an inter-phone gap becomes a `<sil>` token when it is both long enough (at least 6 emission frames, ~120 ms) and quiet enough (mean RMS below 5% of the utterance maximum). The energy gate matters: stop closures for /p/, /t/, /k/ are genuinely silent and can clear the duration threshold on their own.

On LJSpeech this yields 1.65 `<sil>` tokens per utterance, present in 81% of utterances, with a median duration of 28 frames (~325 ms). Because the tokens are derived from the audio, they land where the speaker actually paused rather than wherever a comma happens to be.

At inference `inference.py` inserts `<sil>` at commas, semicolons, colons, and sentence ends. Skipping that step means punctuation has no effect on the output at all — the model has learned what `<sil>` sounds like and simply never receives one.

### PostNet as a separate stage

The PostNet is trained on its own by `train_postnet.py`, on cached `(predicted mel, ground-truth mel)` pairs from `generate_mels.py`. FastSpeech2 is not in the graph, so a run takes minutes.

This is also a fairer setting than joint training. `generate_mels.py` uses ground-truth durations for frame alignment but *predicted* pitch and energy, so the PostNet learns to correct the variance predictors' error as well as the regression blur — the same conditions it faces at inference.

Measured on the LJSpeech validation split: mel L1 per element falls from 0.2824 to 0.2392, a 15.3% improvement.

Because the vocoder is fine-tuned on FastSpeech2's mel distribution, adding the PostNet shifts that distribution and the vocoder must be re-tuned on the sharpened mels. `generate_mels.py --postnet-ckpt` handles that.

### Chunking long input

LJSpeech clips are capped at 10 seconds, giving phoneme-sequence lengths with a median of 70, p90 of 98, and a maximum of 132. Beyond about 100 phonemes the duration predictor and decoder attention are extrapolating, and the tail of the output degrades.

`inference.py` packs the phoneme stream into chunks of at most 100, splitting at `<sil>` where a pause is available and at word boundaries when a clause is too long to split at a pause. Chunks are vocoded separately and joined with a 5 ms crossfade. Splitting only at clause boundaries is not sufficient: one long unpunctuated clause would produce a single oversized chunk regardless of the cap.

## Training

```bash
python download_data.py                        # LJSpeech-1.1, ~2.6 GB
python preprocess.py --workers 8               # mel, f0, energy, phoneme ids
python align.py --overwrite                    # CTC durations + <sil> tokens

python train_fastspeech.py                     # acoustic model
python generate_mels.py                        # predicted mels for the next two stages
python train_postnet.py --steps 30000          # residual refiner
python train_hifigan.py --init-g pretrained_hifigan/generator_v1 \
                        --init-d pretrained_hifigan/do_v1 \
                        --mel-dir mel_pred     # vocoder fine-tune
```

Set `OMP_NUM_THREADS=1` and friends before `preprocess.py` on machines reporting a high core count; each worker otherwise spawns a full OpenBLAS thread pool and exhausts the process limit.

The HiFi-GAN stage fine-tunes the official LJ_V1 checkpoint rather than training from scratch. From-scratch HiFi-GAN needs on the order of 500k steps and is not a good use of rented GPU time when a matched pretrained checkpoint exists. Initialise the discriminators too: a from-scratch discriminator against a pretrained generator drives adversarial loss up steadily and produces a shrill, metallic artifact.

`val/mel_l1` in the HiFi-GAN log compares vocoded output against the *ground-truth* mel, so it is dominated by FastSpeech2's error and stays flat regardless of vocoder quality. Use it to confirm nothing diverged, not to select a checkpoint. Select by listening.

## Repository layout

```
config.py                 all hyperparameters, single import point
preprocess.py             audio features, phoneme ids, vocab, manifests
align.py                  CTC forced alignment, duration + <sil> extraction
model/                    encoder, variance adaptor, decoder, PostNet
vocoder/                  HiFi-GAN generator, MPD/MSD discriminators, losses
data/dataset.py           batching and padding
train_fastspeech.py       acoustic model training
train_postnet.py          standalone PostNet training
train_hifigan.py          vocoder training / fine-tuning
generate_mels.py          predicted mels for vocoder fine-tuning
inference.py              text → wav
test_inference.py         single-utterance check with mel plots
assets/                   phoneme vocab and normalisation stats
```

## Limitations

Single speaker, English only, LJSpeech's reading-audiobook style. Output quality drops on text far outside that domain. The `<sil>` inventory is a single token, so pause *length* is whatever the duration predictor infers from context and cannot be controlled directly. There is no MFA baseline trained under matched conditions, so the CTC aligner is not claimed to be better than MFA — only sufficient, and considerably easier to set up.

## Acknowledgements

FastSpeech2 (Ren et al., 2021), HiFi-GAN (Kong et al., 2020) with the official implementation and LJ_V1 weights from [jik876/hifi-gan](https://github.com/jik876/hifi-gan), the wav2vec2 espeak phone recognizer (Xu et al., 2021) via torchaudio's forced alignment API, espeak-ng and phonemizer, and the LJSpeech dataset by Keith Ito.

## License

Code: MIT. Weights: MIT. LJSpeech-1.1 is public domain. The HiFi-GAN generator is fine-tuned from jik876/hifi-gan, which is MIT-licensed.