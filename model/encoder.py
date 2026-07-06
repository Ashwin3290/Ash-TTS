"""
Encoder module for FastSpeech2.

Architecture:
  Phoneme Embedding
      ↓
  Positional Encoding
      ↓
  N × FFTBlock
      ↓
  Hidden sequence (B, L, d_model)

FFTBlock = Multi-Head Self-Attention + Conv-based Feed Forward
The feed-forward uses two 1D convolutions instead of linear layers —
this is the key difference from vanilla transformer FFN and is
more suited to the local structure of speech.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model as mcfg, audio as acfg


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard fixed sinusoidal PE from 'Attention is All You Need'.
    Not learned — saves parameters and generalises to unseen lengths.
    """
    def __init__(self, d_model, max_len=None):
        super().__init__()
        max_len = max_len or mcfg.max_seq_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # register as buffer — moves to GPU with model, not a learnable parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class ConvFeedForward(nn.Module):
    """
    Conv1D-based feed-forward block.
    Two conv layers with kernel_size > 1 to capture local context.
    This is what makes FFT different from a standard transformer block.
    """
    def __init__(self, d_model, d_ff, kernel_size, dropout):
        super().__init__()
        # padding = (kernel_size - 1) // 2 keeps sequence length unchanged
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_ff,    kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(d_ff,    d_model, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        # pad if sequence is shorter than kernel
        min_len = self.conv1.kernel_size[0]
        pad_amt = max(0, min_len - x.size(2))
        if pad_amt > 0:
            x = F.pad(x, (0, pad_amt))
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(self.conv2(x))
        if pad_amt > 0:
            x = x[:, :, :-pad_amt]
        x = x.transpose(1, 2)
        return x + residual


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer Block.
    Self-Attention + Conv FFN, both with pre-norm residual connections.
    Pre-norm (norm before sublayer) trains more stably than post-norm.
    """
    def __init__(self, d_model, n_heads, d_ff, kernel_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B, T, d) instead of (T, B, d)
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.ffn       = ConvFeedForward(d_model, d_ff, kernel_size, dropout)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, d_model)
        # key_padding_mask: (B, T) True where padded
        residual = x
        x = self.norm_attn(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.drop(x) + residual
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    """
    Phoneme encoder: maps (B, L) phoneme ids → (B, L, d_model) hidden states.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(mcfg.n_phonemes, mcfg.d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(mcfg.d_model)
        self.layers    = nn.ModuleList([
            FFTBlock(mcfg.d_model, mcfg.n_heads, mcfg.d_ff,
                     mcfg.encoder_kernel, mcfg.dropout)
            for _ in range(mcfg.encoder_layers)
        ])
        self.norm = nn.LayerNorm(mcfg.d_model)

    def forward(self, phonemes, ph_lens):
        """
        phonemes: (B, L) — padded phoneme id sequences
        ph_lens:  (B,)   — actual lengths (for masking padding)

        Returns: (B, L, d_model)
        """
        # build key padding mask: True at padding positions
        B, L = phonemes.shape
        mask = torch.arange(L, device=phonemes.device).unsqueeze(0) >= ph_lens.unsqueeze(1)

        x = self.embedding(phonemes)  # (B, L, d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        return self.norm(x), mask