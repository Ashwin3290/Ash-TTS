"""
HiFi-GAN loss functions.

Three losses combined:

1. Adversarial loss (generator + discriminator)
   Generator tries to fool the discriminators.
   Discriminators try to tell real from fake.
   Standard GAN least-squares formulation (more stable than BCE).

2. Feature matching loss
   Generator minimises L1 distance between real and fake feature maps
   at every intermediate discriminator layer.
   Acts as a perceptual loss — forces generator to match fine structure,
   not just fool the final logit.

3. Mel reconstruction loss
   L1 on mel spectrogram of generated vs real waveform.
   Provides a direct, non-adversarial training signal — crucial early in training
   before the discriminators are useful.
"""

import torch
import torch.nn.functional as F


def discriminator_loss(real_outs, fake_outs):
    """
    Least-squares discriminator loss.
    Real should score 1, fake should score 0.
    real_outs, fake_outs: list of (logits, feature_maps)
    """
    loss = 0.0
    for (r_out, _), (f_out, _) in zip(real_outs, fake_outs):
        loss += torch.mean((r_out - 1) ** 2)   # real → 1
        loss += torch.mean(f_out ** 2)           # fake → 0
    return loss


def generator_adversarial_loss(fake_outs):
    """Generator tries to make discriminator output 1 for fakes."""
    loss = 0.0
    for f_out, _ in fake_outs:
        loss += torch.mean((f_out - 1) ** 2)
    return loss


def feature_matching_loss(real_outs, fake_outs):
    """
    L1 between intermediate feature maps of real and fake.
    Averaged over all layers and all discriminators.
    """
    loss = 0.0
    count = 0
    for (_, r_fmaps), (_, f_fmaps) in zip(real_outs, fake_outs):
        for r_feat, f_feat in zip(r_fmaps, f_fmaps):
            loss += F.l1_loss(f_feat, r_feat.detach())
            count += 1
    return loss / max(count, 1)


def mel_reconstruction_loss(pred_wav, real_wav, mel_fn):
    """
    L1 on mel spectrogram of predicted vs real waveform.
    mel_fn: callable that takes (B, T) waveform → (B, n_mels, T_mel) mel
    """
    pred_mel = mel_fn(pred_wav.squeeze(1))
    real_mel = mel_fn(real_wav.squeeze(1))
    return F.l1_loss(pred_mel, real_mel)
