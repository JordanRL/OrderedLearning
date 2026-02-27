"""Gradient and parameter utilities.

Provides flattening, similarity metrics, accumulation, and snapshot
operations for model parameters and gradients.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def snapshot_params(model):
    """Clone current parameter values to CPU for trajectory recording."""
    return {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
    }


def get_gradient_vector(model):
    """Extract flattened gradient from model parameters."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads) if grads else None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


def flatten_grads(grads: Dict[str, torch.Tensor], exclude_bias: bool = True) -> torch.Tensor:
    """Flatten gradient dictionary to single vector."""
    grad_list = []
    for name, grad in sorted(grads.items()):  # Sort for consistency
        if exclude_bias and 'bias' in name:
            continue
        grad_list.append(grad.view(-1).float())
    return torch.cat(grad_list)


def flatten_params(params: Dict[str, torch.Tensor], exclude_bias: bool = True) -> torch.Tensor:
    """Flatten parameter dictionary to single vector."""
    param_list = []
    for name, param in sorted(params.items()):
        if exclude_bias and 'bias' in name:
            continue
        param_list.append(param.view(-1).float())
    return torch.cat(param_list)


# --- Gradient accumulation ---

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
