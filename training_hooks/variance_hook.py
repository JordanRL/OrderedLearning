"""Gradient variance observer hook (sliding window)."""

import torch
import numpy as np

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry
from analysis_tools.utils import flatten_grads


@HookRegistry.register
class VarianceHook(TrainingHook):
    """Compute gradient variance and coherence over a sliding window.

    Ported from analysis_tools/variance.py — same computation, live data.
    Maintains a sliding window of flattened gradient vectors.

    Memory: window_size * param_count * 4 bytes. For ~2.6M params and
    window=10, this is ~100MB. Use --hook-offload-state to move to CPU.
    """

    name = "variance"
    description = "Gradient variance/stability over sliding windows (interference detection)"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs_grads = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('gradient_variance', 'Mean element-wise variance across window',
                       'mean((g_i - g_mean)^2)', 'lower = more stable gradients',
                       label='Grad Variance'),
            MetricInfo('mean_pairwise_cos', 'Mean pairwise cosine similarity in window',
                       'mean(cos(g_i, g_j)) for all pairs', '+1 = coherent, 0 = random',
                       label='Pairwise Cos Sim'),
            MetricInfo('signal_to_noise', 'Ratio of mean gradient norm to std norm',
                       '||g_mean|| / ||std(g)||', 'higher = cleaner signal',
                       label='Signal-to-Noise'),
            MetricInfo('window_mean_norm', 'L2 norm of the mean gradient over window',
                       '||g_mean||_2', label='Window Mean Norm'),
        ]

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._window: list[torch.Tensor] = []

    def compute(self, ctx) -> dict[str, float]:
        if ctx.accumulated_grads is None:
            return {}

        self._window.append(flatten_grads(ctx.accumulated_grads))

        if len(self._window) > self._window_size:
            del self._window[0]

        # Need at least 2 gradients to compute meaningful variance metrics
        if len(self._window) < 2:
            return {}

        grad_stack = torch.stack(self._window)
        mean_grad = grad_stack.mean(dim=0)

        deviations = grad_stack - mean_grad
        variance = (deviations ** 2).mean().item()

        # Pairwise cosine similarities
        cos_sims = []
        for j in range(len(self._window)):
            for k in range(j + 1, len(self._window)):
                g1, g2 = grad_stack[j], grad_stack[k]
                cs = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)
                cos_sims.append(cs.item())

        mean_pairwise_cos = np.mean(cos_sims) if cos_sims else 0
        mean_norm = mean_grad.norm().item()
        std_norm = grad_stack.std(dim=0).norm().item()
        snr = mean_norm / (std_norm + 1e-8)

        return {
            'gradient_variance': variance,
            'mean_pairwise_cos': mean_pairwise_cos,
            'signal_to_noise': snr,
            'window_mean_norm': mean_norm,
        }

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        return {f'window_{i}': t for i, t in enumerate(self._window)}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        self._window = [
            tensors[f'window_{i}']
            for i in range(len(tensors))
            if f'window_{i}' in tensors
        ]

    def reset(self):
        self._window = []
