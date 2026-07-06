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
          mel_pred:     (B, T, n_mels)
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
        mel_pred = self.decoder(x, mel_lens)            # (B, T, n_mels)

        return mel_pred, log_dur_pred, pitch_pred, energy_pred, mel_lens

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
