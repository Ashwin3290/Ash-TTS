from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256        # samples between frames: sample_rate/hop_length = ~86 frames/sec
    win_length: int = 1024
    n_mels: int = 80             # mel filterbank bands — standard for TTS, matches HiFi-GAN expectation
    fmin: float = 0.0
    fmax: float = 8000.0
    max_wav_length: int = 10     # seconds — clips longer than this are dropped during preprocessing
    # mel normalisation bounds (log scale) — clips values to this range before normalising to [-1, 1]
    mel_min: float = -11.5129    # log(1e-5)
    mel_max: float = 2.0


@dataclass
class ModelConfig:
    # shared transformer params
    d_model: int = 256           # hidden dimension throughout the model
    n_heads: int = 2             # attention heads — low to save VRAM
    d_ff: int = 1024             # feed-forward inner dimension (4 * d_model)
    dropout: float = 0.2

    # encoder (phoneme → hidden)
    encoder_layers: int = 4
    encoder_kernel: int = 9      # conv kernel in FFT block

    # decoder (hidden → mel frames)
    decoder_layers: int = 4
    decoder_kernel: int = 9

    # variance adaptor
    variance_kernel: int = 3
    variance_channels: int = 256

    # pitch / energy: these are quantised into bins for prediction as a classification problem
    n_pitch_bins: int = 256
    n_energy_bins: int = 256

    # postnet (Tacotron2-style residual refiner on top of the decoder output —
    # sharpens the over-smoothed mels that a plain L2/L1 regression decoder produces)
    postnet_layers: int = 5
    postnet_channels: int = 512
    postnet_kernel: int = 5
    postnet_dropout: float = 0.5

    # phoneme vocab — filled in by dataset, set conservatively here
    n_phonemes: int = 150        # espeak-ng produces ~70 distinct phonemes for English, pad for safety

    # positional encoding max length (in frames — 10s * 86fps = 860, round up)
    max_seq_len: int = 2048


@dataclass
class TrainFastSpeechConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    # raised from 250k after the phone-level-duration run finished with its
    # best checkpoint at step 245k — i.e. still improving at the ceiling
    # (unlike the old uniform-duration runs, which bottomed out ~100k before
    # theirs). One more 250k block; best.pt tracking captures the optimum
    # wherever val loss actually bottoms out within it.
    max_steps: int = 700_000
    grad_clip: float = 0.5
    fp16: bool = True
    # decoupled weight decay (AdamW) — val loss climbed steadily past ~step 90k
    # while train loss kept falling (overfitting). Earlier attempts used plain
    # Adam + L2 penalty applied to every parameter including biases/norms,
    # which is known to regularise poorly (see Loshchilov & Hutter, the paper
    # that introduced AdamW) and pushed val loss up further instead of down.
    # 0.01 is AdamW's own default and a standard starting point.
    weight_decay: float = 0.01

    # loss weights
    mel_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    pitch_loss_weight: float = 1.0
    energy_loss_weight: float = 1.0

    log_every: int = 100
    save_every: int = 5000
    val_every: int = 1000

    # PostNet warm-start (train_fastspeech.py --init): backbone stays frozen while
    # the fresh PostNet trains, and unfreezes only once val mel_after L1 beats the
    # frozen backbone's own mel_before L1 by the threshold below (metric-gated,
    # not step-gated). max_freeze_steps is a safety cap in case the gate never
    # triggers. After unfreezing, fine-tune everything at a flat low LR with a
    # short linear re-warmup so the converged backbone isn't blown up by fresh
    # optimizer state.
    unfreeze_threshold: float = 0.98   # unfreeze when val_after <= baseline * this
    max_freeze_steps: int = 30_000
    finetune_lr: float = 1e-4
    finetune_warmup: int = 2000


@dataclass
class HiFiGANConfig:
    # generator
    upsample_rates: list = field(default_factory=lambda: [8, 8, 2, 2])      # product must = hop_length (256)
    upsample_kernel_sizes: list = field(default_factory=lambda: [16, 16, 4, 4])
    # must be 512 to match jik876/hifi-gan's official generator_v1 checkpoint —
    # we're fine-tuning from it, so architecture width can't be shrunk for VRAM
    upsample_initial_channels: int = 512
    resblock_kernel_sizes: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list = field(default_factory=lambda: [[1,3,5], [1,3,5], [1,3,5]])

    # discriminator
    # MPD (Multi-Period Discriminator) periods
    mpd_periods: list = field(default_factory=lambda: [2, 3, 5, 7, 11])

    # training
    batch_size: int = 16
    learning_rate: float = 2e-4
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    # raised from 25k after the fine-tune finished with its best at step 22k
    # (still improving at the ceiling); 50k also matches the official repo's
    # fine-tuning budget more closely
    max_steps: int = 50_000
    fp16: bool = True

    # audio segment for training — HiFi-GAN trains on fixed-length chunks, not full utterances
    segment_length: int = 8192  # samples (~0.37s at 22050Hz)

    log_every: int = 100
    save_every: int = 5000
    val_every: int = 1000   # validation mel-L1 pass; drives g_best/d_best tracking


@dataclass
class PathConfig:
    data_root: Path = Path("data/LJSpeech-1.1")
    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("checkpoints")
    fastspeech_ckpt_dir: Path = Path("checkpoints/fastspeech2")
    hifigan_ckpt_dir: Path = Path("checkpoints/hifigan")
    log_dir: Path = Path("logs")

    def make_dirs(self):
        for p in [self.processed_dir, self.fastspeech_ckpt_dir, self.hifigan_ckpt_dir, self.log_dir]:
            p.mkdir(parents=True, exist_ok=True)


# single import point for the whole project
audio  = AudioConfig()
model  = ModelConfig()
train  = TrainFastSpeechConfig()
hifigan = HiFiGANConfig()
paths  = PathConfig()