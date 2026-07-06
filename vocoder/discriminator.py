"""
HiFi-GAN Discriminators.

Two discriminator families working together:

1. MPD (Multi-Period Discriminator)
   Reshapes the waveform into 2D (time × period) and applies 2D convolutions.
   Different periods [2, 3, 5, 7, 11] capture different quasi-periodic structures.
   Primes are used to avoid aliasing between periods.
   This is what enforces fine-grained pitch and periodicity structure.

2. MSD (Multi-Scale Discriminator)
   Operates on the raw waveform at 3 scales: full resolution, 2× downsampled, 4× downsampled.
   Average pooling for downsampling. Captures broad spectral structure.

Both return lists of:
  - final logit (real/fake score)
  - intermediate feature maps (used for feature matching loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from config import hifigan as hcfg

LRELU_SLOPE = 0.1


class PeriodDiscriminator(nn.Module):
    """Single period discriminator. Reshapes 1D audio into 2D grid of shape (T//p, p)."""
    def __init__(self, period):
        super().__init__()
        self.period = period
        norm = weight_norm

        # 5 strided Conv2D layers — each halves T dimension
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,   32,  (5, 1), (3, 1), padding=(2, 0))),
            norm(nn.Conv2d(32,  128, (5, 1), (3, 1), padding=(2, 0))),
            norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm(nn.Conv2d(512, 1024,(5, 1), (3, 1), padding=(2, 0))),
            norm(nn.Conv2d(1024,1024,(5, 1), 1,      padding=(2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        # x: (B, 1, T)
        B, C, T = x.shape
        # pad T to be divisible by period
        pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad), "reflect")
        T_padded = x.shape[-1]
        x = x.view(B, C, T_padded // self.period, self.period)  # (B, 1, T//p, p)

        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_maps


class MPD(nn.Module):
    """Multi-Period Discriminator."""
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in hcfg.mpd_periods
        ])

    def forward(self, real, fake):
        """Returns lists of (logits, feature_maps) for real and fake."""
        real_outs, fake_outs = [], []
        for disc in self.discriminators:
            r_out, r_fmaps = disc(real)
            f_out, f_fmaps = disc(fake)
            real_outs.append((r_out, r_fmaps))
            fake_outs.append((f_out, f_fmaps))
        return real_outs, fake_outs


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator. Operates on 1D waveform."""
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1,  padding=7)),
            norm(nn.Conv1d(128,  128,  41, 2,  padding=20, groups=4)),
            norm(nn.Conv1d(128,  256,  41, 2,  padding=20, groups=16)),
            norm(nn.Conv1d(256,  512,  41, 4,  padding=20, groups=16)),
            norm(nn.Conv1d(512,  1024, 41, 4,  padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1,  padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x):
        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_maps


class MSD(nn.Module):
    """
    Multi-Scale Discriminator.
    First sub-discriminator uses spectral norm (more stable at full resolution).
    Others use weight norm.
    """
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, real, fake):
        real_outs, fake_outs = [], []
        x_real, x_fake = real, fake
        for disc, pool in zip(self.discriminators, self.meanpools):
            x_real = pool(x_real)
            x_fake = pool(x_fake)
            r_out, r_fmaps = disc(x_real)
            f_out, f_fmaps = disc(x_fake)
            real_outs.append((r_out, r_fmaps))
            fake_outs.append((f_out, f_fmaps))
        return real_outs, fake_outs
