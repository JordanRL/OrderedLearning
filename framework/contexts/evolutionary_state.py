"""Evolutionary state objects for hook contexts.

Carries population fitness information and optional pseudo-gradients
for the EVOLUTIONARY paradigm (Evolution Strategies, Genetic Algorithms).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EvolutionaryModelState:
    """Model access for evolutionary training.

    The model field is the center/elite model whose parameters represent
    the current best solution. lr is the outer step size (if an optimizer
    is present for ES pseudo-gradient updates).
    """
    model: nn.Module
    lr: float | None = None
    best_fitness: float | None = None
    mean_fitness: float | None = None

    @property
    def device(self) -> torch.device:
        """Device the center/elite model lives on."""
        return next(self.model.parameters()).device


@dataclass(frozen=True)
class EvolutionaryGradientState:
    """Gradient access for evolutionary training.

    pseudo_gradient is the ES-estimated gradient (fitness-weighted
    perturbation average). None for GA which has no gradient analog.
    fitness_values are per-individual fitness scores for the generation.
    """
    pseudo_gradient: dict[str, torch.Tensor] | None = None
    fitness_values: list[float] | None = None
