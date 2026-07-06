"""
HiFi-GAN Generator.

Architecture:
  mel spectrogram (B, n_mels, T)
      ↓
  Conv1d input projection
      ↓
  N × Upsample block (transposed conv to increase time resolution)
      each block contains M × Multi-Receptive Field Fusion (MRF) residual blocks
      ↓
  LeakyReLU + Conv1d + Tanh
      ↓
  waveform (B, 1, T*hop_length)

The upsampling stack must have: product(upsample_rates) == hop_length
With rates [8, 8, 2, 2]: 8*8*2*2 = 256 = hop_length. Correct.

MRF fuses outputs from residual blocks with different kernel sizes and dilations,
giving the generator multi-scale receptive fields to model both fine
waveform structure and longer-range periodicity simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from config import hifigan as hcfg, audio as acfg

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions.
    For each dilation in dilation_sizes: apply 2 conv layers with that dilation.
    Output is sum of all dilation branches + input residual.
    """
    def __init__(self, channels, kernel_size, dilation_sizes):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            ))
            for d in dilation_sizes
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            ))
            for _ in dilation_sizes
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs1 + self.convs2:
            remove_weight_norm(c)


class MRF(nn.Module):
    """
    Multi-Receptive Field Fusion.
    Runs multiple ResBlocks with different kernels/dilations in parallel,
    averages their outputs.
    """
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, k, d)
            for k, d in zip(hcfg.resblock_kernel_sizes, hcfg.resblock_dilation_sizes)
        ])

    def forward(self, x):
        out = None
        for block in self.blocks:
            y = block(x)
            out = y if out is None else out + y
        return out / len(self.blocks)

    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        initial_ch = hcfg.upsample_initial_channels

        self.conv_pre = weight_norm(nn.Conv1d(acfg.n_mels, initial_ch, 7, padding=3))

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = initial_ch
        for rate, k_size in zip(hcfg.upsample_rates, hcfg.upsample_kernel_sizes):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch, ch // 2,
                kernel_size=k_size,
                stride=rate,
                padding=(k_size - rate) // 2,
            )))
            ch //= 2
            self.mrfs.append(MRF(ch))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, mel):
        # mel: (B, n_mels, T) — note: channel first for conv
        x = self.conv_pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = mrf(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x   # (B, 1, T*hop_length)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for up in self.ups:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
