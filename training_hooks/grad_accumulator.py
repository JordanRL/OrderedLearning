"""
Gradient accumulation helpers.

Shared between the training loop and ModelDataContext.run_training_epoch().
Extracts the inline accumulation logic into reusable functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def create_accumulator(model: nn.Module) -> tuple[dict[str, torch.Tensor], int]:
    """Create a zero-initialized gradient accumulator.

    Args:
        model: The model whose parameters to accumulate gradients for.

    Returns:
        Tuple of (accumulator_dict, batch_count=0).
        The accumulator stays on the same device as the model parameters.
    """
    accum = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }
    return accum, 0


def accumulate(
    accum: dict[str, torch.Tensor],
    model: nn.Module,
    count: int,
) -> int:
    """Add current gradients to the accumulator.

    Call after loss.backward() and before optimizer.step().

    Args:
        accum: Gradient accumulator dict.
        model: Model with .grad populated from backward().
        count: Current batch count.

    Returns:
        Updated batch count (count + 1).
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            accum[name].add_(param.grad.detach())
    return count + 1


def finalize(
    accum: dict[str, torch.Tensor],
    count: int,
    to_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Normalize accumulated gradients and optionally move to CPU.

    Args:
        accum: Gradient accumulator dict (sum of batch gradients).
        count: Number of batches accumulated.
        to_cpu: If True, move result tensors to CPU.

    Returns:
        Dict of parameter_name -> mean gradient tensor.
    """
    if count > 0:
        for name in accum:
            accum[name].div_(count)
    if to_cpu:
        accum = {name: grad.cpu() for name, grad in accum.items()}
    return accum
