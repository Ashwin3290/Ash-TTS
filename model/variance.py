"""
Variance Adaptor — the core of what makes FastSpeech2 expressive.

Takes the encoder hidden states (phoneme-level) and:
  1. Duration predictor  → predicts how many mel frames each phoneme spans
  2. Length regulator    → repeats each phoneme hidden state by its duration,
                           expanding (B, L, d) → (B, T, d)
  3. Pitch predictor     → predicts frame-level F0 (after expansion)
  4. Energy predictor    → predicts frame-level energy (after expansion)

Pitch and energy are predicted as classification over quantised bins,
not as raw regression. This is more stable to train and allows embedding
lookup to inject the prediction back into the hidden states.

At inference time you can scale duration/pitch/energy to control
speaking rate, pitch level, and loudness independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import model as mcfg, audio as acfg


class VariancePredictor(nn.Module):
    """
    Shared architecture for duration, pitch, and energy predictors.
    Two conv layers + linear output. Predicts one scalar per frame/phoneme.
    """
    def __init__(self, d_model, channels, kernel_size, dropout, out_dim=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.LayerNorm(channels),   # LN after transpose — applied per-channel
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(channels, out_dim)

    def forward(self, x):
        # x: (B, T, d_model)
        out = x.transpose(1, 2)        # (B, d_model, T) for Conv1d

        # ensure sequence is at least as long as the conv kernel
        min_len = self.layers[0].kernel_size[0]
        pad_amt = max(0, min_len - out.size(2))
        if pad_amt > 0:
            out = F.pad(out, (0, pad_amt))

        # apply conv layers — LayerNorm needs (B, T, C) so we transpose around it
        out = self.layers[0](out)       # Conv1d
        out = self.layers[1](out)       # ReLU
        out = self.layers[2](out.transpose(1,2)).transpose(1,2)  # LN
        out = self.layers[3](out)       # Dropout
        out = self.layers[4](out)       # Conv1d
        out = self.layers[5](out)       # ReLU
        out = self.layers[6](out.transpose(1,2)).transpose(1,2)  # LN
        out = self.layers[7](out)       # Dropout
        out = out.transpose(1, 2)       # back to (B, T, C)

        # trim back to original length if we padded
        if pad_amt > 0:
            out = out[:, :x.size(1), :]

        return self.linear(out).squeeze(-1)  # (B, T)


class LengthRegulator(nn.Module):
    """
    Expands phoneme-level hidden states to frame-level using predicted durations.

    Example:
      phoneme hidden: [h_a, h_b, h_c]
      durations:      [  2,   3,   1]
      output:         [h_a, h_a, h_b, h_b, h_b, h_c]

    During training: uses ground-truth durations (teacher forcing)
    During inference: uses predicted durations (rounded to int)
    """
    def forward(self, x, durations, max_len=None):
        """
        x:         (B, L, d_model)
        durations: (B, L) int — frames per phoneme
        Returns:   (B, T, d_model) where T = sum of durations
        """
        outputs = []
        for i in range(x.size(0)):
            repeated = torch.repeat_interleave(x[i], durations[i], dim=0)  # (T_i, d)
            outputs.append(repeated)

        # pad to max length in batch (or given max_len)
        if max_len is None:
            max_len = max(o.size(0) for o in outputs)
        d_model = x.size(-1)
        out = torch.zeros(len(outputs), max_len, d_model, device=x.device, dtype=x.dtype)
        for i, o in enumerate(outputs):
            T = min(o.size(0), max_len)
            out[i, :T] = o[:T]
        return out


class VarianceAdaptor(nn.Module):
    def __init__(self, pitch_stats=None, energy_stats=None):
        super().__init__()

        d  = mcfg.d_model
        ch = mcfg.variance_channels
        k  = mcfg.variance_kernel
        dr = mcfg.dropout

        self.duration_predictor = VariancePredictor(d, ch, k, dr, out_dim=1)
        self.pitch_predictor    = VariancePredictor(d, ch, k, dr, out_dim=1)
        self.energy_predictor   = VariancePredictor(d, ch, k, dr, out_dim=1)

        self.length_regulator = LengthRegulator()

        # pitch and energy embedding: quantise continuous value → bin id → embedding
        # adds the predicted value back into the hidden representation
        self.pitch_embedding  = nn.Embedding(mcfg.n_pitch_bins,  d)
        self.energy_embedding = nn.Embedding(mcfg.n_energy_bins, d)

        # quantisation bins computed from dataset stats — set via set_stats()
        self.register_buffer("pitch_bins",  torch.linspace(-3, 3, mcfg.n_pitch_bins  - 1))
        self.register_buffer("energy_bins", torch.linspace(-3, 3, mcfg.n_energy_bins - 1))

    def set_stats(self, f0_mean, f0_std, f0_min, f0_max, energy_mean, energy_std, energy_min, energy_max):
        """Call after loading dataset stats to set data-driven quantisation bins."""
        # pitch bins over normalised range (already z-scored in dataset.py)
        pitch_bins  = torch.linspace(-3, 3, mcfg.n_pitch_bins  - 1)
        energy_bins = torch.linspace(-3, 3, mcfg.n_energy_bins - 1)
        self.pitch_bins.copy_(pitch_bins)
        self.energy_bins.copy_(energy_bins)

    def quantise(self, values, bins):
        """Bucket continuous values into bin ids for embedding lookup."""
        return torch.bucketize(values, bins)

    def forward(self, x, ph_lens,
                durations_gt=None, f0_gt=None, energy_gt=None,
                duration_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        """
        x:            (B, L, d_model) — encoder output
        ph_lens:      (B,) — phoneme sequence lengths
        durations_gt: (B, L) — ground truth durations (None at inference)
        f0_gt:        (B, T) — ground truth F0 (None at inference)
        energy_gt:    (B, T) — ground truth energy (None at inference)
        *_scale:      float  — inference-time control knobs

        Returns:
          x_expanded:       (B, T, d_model)
          log_dur_pred:     (B, L) — log duration predictions (for loss)
          pitch_pred:       (B, T)
          energy_pred:      (B, T)
          mel_lens:         (B,) — frame lengths after expansion
        """
        # --- Duration ---
        log_dur_pred = self.duration_predictor(x)   # (B, L) — predict in log domain (more stable)

        if durations_gt is not None:
            # training: use ground truth durations to expand
            durations = durations_gt
            mel_lens  = durations.sum(dim=1)
        else:
            # inference: convert predicted log-durations to frame counts
            durations = (log_dur_pred.exp() * duration_scale).round().long().clamp(min=0)
            durations = durations * (torch.arange(durations.size(1), device=x.device).unsqueeze(0)
                                     < ph_lens.unsqueeze(1))  # zero out padding
            mel_lens = durations.sum(dim=1)

        x_expanded = self.length_regulator(x, durations)  # (B, T, d)

        # --- Pitch ---
        T_exp = x_expanded.size(1)
        pitch_pred = self.pitch_predictor(x_expanded)       # (B, T)
        if f0_gt is not None:
            pitch_target = f0_gt[:, :T_exp]
            # f0_gt may be shorter than x_expanded — pad if needed
            if pitch_target.size(1) < T_exp:
                pitch_target = F.pad(pitch_target, (0, T_exp - pitch_target.size(1)))
            pitch_bins = self.quantise(pitch_target, self.pitch_bins)
        else:
            pitch_bins = self.quantise(pitch_pred * pitch_scale, self.pitch_bins)
        x_expanded = x_expanded + self.pitch_embedding(pitch_bins)

        # --- Energy ---
        energy_pred = self.energy_predictor(x_expanded)     # (B, T)
        if energy_gt is not None:
            energy_target = energy_gt[:, :T_exp]
            if energy_target.size(1) < T_exp:
                energy_target = F.pad(energy_target, (0, T_exp - energy_target.size(1)))
            energy_bins = self.quantise(energy_target, self.energy_bins)
        else:
            energy_bins = self.quantise(energy_pred * energy_scale, self.energy_bins)
        x_expanded = x_expanded + self.energy_embedding(energy_bins)
        return x_expanded, log_dur_pred, pitch_pred, energy_pred, mel_lens