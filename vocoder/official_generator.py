"""
Official jik876/hifi-gan Generator architecture, for loading pretrained
checkpoints (LJ_V1, UNIVERSAL_V1) directly.

This is a SEPARATE class from vocoder/generator.py. That file's Generator
groups resblocks per upsample-stage under an MRF submodule (mrfs.N.blocks.N...),
which does not match the official checkpoint's flat resblocks.N... naming —
so state_dict keys never align and load_state_dict fails. This module mirrors
the official layout exactly so pretrained weights load without remapping.

Usage:
    from vocoder.official_generator import load_pretrained_generator
    generator = load_pretrained_generator("generator_v1", "config.json", device)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        pad = lambda k, d: (k * d - d) // 2
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                   padding=pad(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                   padding=pad(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    """Matches jik876/hifi-gan models.py Generator exactly (key names + shapes)."""
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                h.upsample_initial_channel // (2 ** i),
                h.upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def config_from_hcfg(hcfg):
    """Build the AttrDict Generator(h) expects from our config.hifigan dataclass."""
    return AttrDict({
        "resblock_kernel_sizes":    hcfg.resblock_kernel_sizes,
        "resblock_dilation_sizes":  hcfg.resblock_dilation_sizes,
        "upsample_rates":           hcfg.upsample_rates,
        "upsample_kernel_sizes":    hcfg.upsample_kernel_sizes,
        "upsample_initial_channel": hcfg.upsample_initial_channels,
    })


def load_pretrained_generator(ckpt_path, config_path, device):
    """
    Load an official HiFi-GAN generator checkpoint (e.g. generator_v1 /
    generator_universal) for inference.

    IMPORTANT: mel fed to this generator must be the UNNORMALISED natural-log
    mel (the official mel_spectrogram() scale), not the [-1, 1] normalised mel
    that FastSpeech2 is trained against — see inference.py's denorm step.
    """
    with open(config_path) as f:
        h = AttrDict(json.load(f))

    from config import audio as acfg
    mismatches = []
    if h.sampling_rate != acfg.sample_rate: mismatches.append("sample_rate")
    if h.hop_size != acfg.hop_length:       mismatches.append("hop_length")
    if h.win_size != acfg.win_length:       mismatches.append("win_length")
    if h.n_fft != acfg.n_fft:               mismatches.append("n_fft")
    if h.num_mels != acfg.n_mels:           mismatches.append("n_mels")
    if mismatches:
        raise ValueError(f"Audio config mismatch vs {config_path}: {mismatches}. "
                          f"FastSpeech2 was trained with a different mel spec than "
                          f"this vocoder expects — outputs will be garbled.")

    generator = Generator(h).to(device)
    state = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator
