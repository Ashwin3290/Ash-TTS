"""
Mel Decoder for FastSpeech2.

Identical FFT block architecture as the encoder, but:
  - input is the frame-level expanded hidden states from variance adaptor
  - output is projected to n_mels dimensions (mel spectrogram)

Non-autoregressive: predicts all frames in parallel.
This is fundamentally different from Tacotron2 which steps one frame at a time.
Consequence: much faster training and inference, but relies on the
variance adaptor to get timing right.
"""

import torch
import torch.nn as nn

from config import model as mcfg, audio as acfg
from model.encoder import FFTBlock, SinusoidalPositionalEncoding


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(mcfg.d_model)
        self.layers  = nn.ModuleList([
            FFTBlock(mcfg.d_model, mcfg.n_heads, mcfg.d_ff,
                     mcfg.decoder_kernel, mcfg.dropout)
            for _ in range(mcfg.decoder_layers)
        ])
        self.norm    = nn.LayerNorm(mcfg.d_model)
        # project hidden → mel
        self.linear  = nn.Linear(mcfg.d_model, acfg.n_mels)

    def forward(self, x, mel_lens):
        """
        x:        (B, T, d_model) — expanded hidden states from variance adaptor
        mel_lens: (B,)            — actual frame lengths (for padding mask)

        Returns:  (B, T, n_mels) — predicted mel spectrogram
        """
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= mel_lens.unsqueeze(1)

        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)
        x = self.norm(x)
        return self.linear(x)  # (B, T, n_mels)
