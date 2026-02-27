"""Consecutive gradient alignment observer hook."""

import torch
import numpy as np

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, GradientAvailability
from framework.utils import flatten_grads


@HookRegistry.register
class ConsecutiveHook(TrainingHook):
    """Compute cosine similarity between consecutive epoch gradients.

    Ported from analysis_tools/consecutive.py — same computation, live data.
    Maintains one previous flattened gradient vector as state.
    """

    name = "consecutive"
    description = "Cosine similarity between consecutive epoch gradients"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs = HookNeeds.ACCUMULATED_GRADS
    requires = HookRequirements(gradient_availability=GradientAvailability.GLOBAL_GRADIENTS)

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('cos_sim', 'Cosine similarity between consecutive epoch gradients',
                       'cos(g_t, g_{t-1})', '+1 = same direction, -1 = opposite',
                       label='Consecutive Cos Sim'),
            MetricInfo('angle_degrees', 'Angle between consecutive epoch gradients',
                       'arccos(cos_sim) * 180/pi', '0 = identical, 90 = orthogonal',
                       label='Consecutive Angle'),
        ]

    def __init__(self):
        self._prev_flat: torch.Tensor | None = None

    def compute(self, ctx, **state) -> dict[str, float]:
        gradient_state = state.get('gradient_state')
        if gradient_state is None or gradient_state.accumulated_grads is None:
            return {}

        curr_flat = flatten_grads(gradient_state.accumulated_grads)
        metrics = {}

        if self._prev_flat is not None:
            cos_sim = torch.dot(self._prev_flat, curr_flat) / (
                self._prev_flat.norm() * curr_flat.norm() + 1e-8
            )
            angle = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / np.pi

            metrics = {
                'cos_sim': cos_sim.item(),
                'angle_degrees': angle.item(),
            }

        self._prev_flat = curr_flat
        return metrics

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        if self._prev_flat is not None:
            return {'prev_flat': self._prev_flat}
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        if 'prev_flat' in tensors:
            self._prev_flat = tensors['prev_flat']

    def reset(self):
        self._prev_flat = None
