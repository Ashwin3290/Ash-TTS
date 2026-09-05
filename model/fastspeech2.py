"""
FastSpeech2: full model assembly.

Forward pass:
  phonemes → Encoder → VarianceAdaptor → Decoder → mel predictions

The model predicts mel spectrograms in a single parallel forward pass.
No recurrence, no autoregressive decoding.
"""

import torch
import torch.nn as nn

from model.encoder  import Encoder
from model.variance import VarianceAdaptor
from model.decoder  import Decoder
from config import model as mcfg, audio as acfg


class FastSpeech2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder          = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder          = Decoder()

    def forward(self, phonemes, ph_lens,
                durations_gt=None, f0_gt=None, energy_gt=None,
                duration_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        """
        Training:   pass all _gt tensors
        Inference:  pass only phonemes + ph_lens, use *_scale knobs

        Returns:
          mel:          (B, T, n_mels) — decoder output
          log_dur_pred: (B, L)
          pitch_pred:   (B, T)
          energy_pred:  (B, T)
          mel_lens:     (B,)
        """
        # 1. encode phoneme sequence
        enc_out, _ = self.encoder(phonemes, ph_lens)   # (B, L, d_model)

        # 2. variance adaptor: predict durations, expand, predict pitch+energy
        x, log_dur_pred, pitch_pred, energy_pred, mel_lens = self.variance_adaptor(
            enc_out, ph_lens,
            durations_gt=durations_gt,
            f0_gt=f0_gt,
            energy_gt=energy_gt,
            duration_scale=duration_scale,
            pitch_scale=pitch_scale,
            energy_scale=energy_scale,
        )

        # 3. decode to mel
        mel = self.decoder(x, mel_lens)          # (B, T, n_mels)

        return mel, log_dur_pred, pitch_pred, energy_pred, mel_lens

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_fs2_state(model, state):
    """
    Load a FastSpeech2 state dict, dropping any postnet.* keys from
    checkpoints saved before the PostNet became a separate training stage
    (see train_postnet.py) — the joint in-model PostNet no longer exists.
    Any other mismatch is a real error.
    """
    state = {k: v for k, v in state.items() if not k.startswith("postnet.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model


def masked_mse(pred, target, lens, dim=-1):
    """
    MSE loss that ignores padding positions.
    pred, target: (B, T, ...)
    lens:         (B,) — valid lengths
    """
    B, T = pred.shape[:2]
    mask = torch.arange(T, device=pred.device).unsqueeze(0) < lens.unsqueeze(1)  # (B, T)
    if pred.dim() == 3:
        mask = mask.unsqueeze(-1)  # (B, T, 1) for mel
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum()


def masked_l1(pred, target, lens):
    """
    L1 loss that ignores padding positions. Same normalisation as masked_mse
    (divides by valid FRAME count, so the value is summed over mel channels —
    divide by n_mels for a per-element number).
    """
    B, T = pred.shape[:2]
    mask = torch.arange(T, device=pred.device).unsqueeze(0) < lens.unsqueeze(1)  # (B, T)
    if pred.dim() == 3:
        mask = mask.unsqueeze(-1)
    loss = (pred - target).abs() * mask
    return loss.sum() / mask.sum()


def pitch_loss(pitch_pred, f0_gt, lens):
    """
    MSE on voiced AND valid frames only. f0_gt uses 0 as the unvoiced marker
    (see dataset.py), so unvoiced frames must be excluded from the target —
    the old formulation masked the *prediction* but compared against the
    unmasked target and averaged over padding, which corrupted the objective.
    """
    B, T = pitch_pred.shape
    len_mask = torch.arange(T, device=pitch_pred.device).unsqueeze(0) < lens.unsqueeze(1)
    voiced = (f0_gt != 0).float() * len_mask
    loss = ((pitch_pred - f0_gt) ** 2) * voiced
    return loss.sum() / voiced.sum().clamp(min=1)


def energy_loss(energy_pred, energy_gt, lens):
    """MSE over valid frames only (padding zeros previously diluted the loss)."""
    B, T = energy_pred.shape
    mask = torch.arange(T, device=energy_pred.device).unsqueeze(0) < lens.unsqueeze(1)
    loss = ((energy_pred - energy_gt) ** 2) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_l1_silence_weighted(pred, target, lens, energy_gt=None, silence_weight=2.0):
    """
    L1 loss with higher weight on silence frames.
    Silence frames (low energy) get higher gradient contribution to prevent
    their loss from being diluted in the batch.

    energy_gt: (B, T) normalized energy; silence has energy near -1
    silence_weight: multiplier for silence frame loss (e.g., 2.0 = 2x weight)
    """
    B, T = pred.shape[:2]
    mask = torch.arange(T, device=pred.device).unsqueeze(0) < lens.unsqueeze(1)
    if pred.dim() == 3:
        mask = mask.unsqueeze(-1)

    # Reweight silence frames: energy < -0.5 is typically silence
    frame_weights = torch.ones_like(mask, dtype=torch.float32)
    if energy_gt is not None:
        is_silence = (energy_gt < -0.5).unsqueeze(-1) if energy_gt.dim() == 2 else (energy_gt < -0.5)
        frame_weights = torch.where(is_silence & mask, torch.tensor(silence_weight, device=frame_weights.device), frame_weights)

    loss = (pred - target).abs() * mask * frame_weights
    return loss.sum() / (mask * frame_weights).sum().clamp(min=1)


def duration_loss(log_dur_pred, durations_gt, ph_lens):
    """
    L2 loss on log-durations. We predict log(duration+1) to prevent negative values.
    durations_gt are clipped at 1 minimum (a phoneme must span at least 1 frame).
    """
    target = torch.log((durations_gt.float() + 1).clamp(min=1))
    B, L = log_dur_pred.shape
    mask = torch.arange(L, device=log_dur_pred.device).unsqueeze(0) < ph_lens.unsqueeze(1)
    loss = ((log_dur_pred - target) ** 2) * mask
    return loss.sum() / mask.sum()
