"""Adversarial (GAN) state objects for hook contexts.

Carries dual-model access for generator/discriminator paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdversarialModelState:
    """Model access for adversarial (GAN) training."""
    generator: nn.Module
    discriminator: nn.Module
    lr: dict[str, float] | None = None  # {'generator': ..., 'discriminator': ...}

    @property
    def model(self) -> nn.Module:
        """Primary model (generator) for hooks that expect a single model."""
        return self.generator

    @property
    def device(self) -> torch.device:
        """Device the generator lives on."""
        return next(self.generator.parameters()).device


@dataclass(frozen=True)
class AdversarialGradientState:
    """Gradient access for adversarial training."""
    g_gradients: dict[str, torch.Tensor] | None = None
    d_gradients: dict[str, torch.Tensor] | None = None
