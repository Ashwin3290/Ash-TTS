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


class PostNet(nn.Module):
    """
    Tacotron2-style residual refiner: 5 Conv1d layers over the mel channel axis
    predicting a residual that is ADDED to the decoder's mel output
    (mel_after = mel_before + postnet(mel_before)).

    The final conv is zero-initialised, so a freshly constructed PostNet is an
    exact identity (residual = 0). This means:
      - checkpoints saved before the PostNet existed still load and produce
        identical output (strict=False leaves the residual at zero), and
      - warm-start fine-tuning begins from the backbone's converged operating
        point instead of adding random noise to it.
    """
    def __init__(self):
        super().__init__()
        n_mels = acfg.n_mels
        ch     = mcfg.postnet_channels
        k      = mcfg.postnet_kernel
        pad    = (k - 1) // 2
        n      = mcfg.postnet_layers

        convs = []
        for i in range(n):
            in_ch  = n_mels if i == 0 else ch
            out_ch = n_mels if i == n - 1 else ch
            convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=pad),
                nn.BatchNorm1d(out_ch),
            ))
        self.convs   = nn.ModuleList(convs)
        self.dropout = nn.Dropout(mcfg.postnet_dropout)

        nn.init.zeros_(self.convs[-1][0].weight)
        nn.init.zeros_(self.convs[-1][0].bias)

    def forward(self, x):
        """
        x: (B, T, n_mels) — decoder mel output
        Returns: (B, T, n_mels) — residual to add to x
        """
        out = x.transpose(1, 2)                    # (B, n_mels, T) for Conv1d
        for i, conv in enumerate(self.convs):
            out = conv(out)
            if i < len(self.convs) - 1:            # no tanh on the final layer
                out = torch.tanh(out)
            out = self.dropout(out)
        return out.transpose(1, 2)                 # (B, T, n_mels)
