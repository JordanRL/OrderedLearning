"""Learning phase detection observer hook."""

import torch

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry
from analysis_tools.utils import flatten_grads


@HookRegistry.register
class PhasesHook(TrainingHook):
    """Detect learning phases via gradient and representation dynamics.

    Ported from analysis_tools/phases.py — same per-snapshot computation,
    live data. Maintains previous embedding, grad norm, and grad velocity.

    Note: Phase boundary detection (gaussian smoothing + curvature) is a
    post-hoc computation that requires the full trajectory. The hook
    provides the raw metrics; phase labels are based on accuracy thresholds.
    """

    name = "phases"
    description = "Learning phase detection (gradient velocity/acceleration, representation change)"
    hook_points = {HookPoint.SNAPSHOT}
    loop_points = {'epoch': {HookPoint.SNAPSHOT}}
    needs_grads = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('grad_velocity', 'First derivative of gradient norm',
                       '||g_t|| - ||g_{t-1}||', '+ = growing gradients',
                       label='Grad Velocity'),
            MetricInfo('grad_acceleration', 'Second derivative of gradient norm',
                       'v_t - v_{t-1}', '+ = accelerating growth',
                       label='Grad Acceleration'),
            MetricInfo('embedding_change', 'L2 distance between consecutive embedding matrices',
                       '||E_t - E_{t-1}||_2', label='Embedding Δ'),
            MetricInfo('embedding_change_normalized', 'Relative embedding change',
                       '||E_t - E_{t-1}|| / ||E_{t-1}||', label='Embedding Δ (Rel)'),
            MetricInfo('phase_code', 'Learning phase from val accuracy thresholds',
                       formula='0=pre, 1=early, 2=rapid, 3=refine, 4=converged',
                       label='Phase'),
        ]

    def __init__(self):
        self._prev_emb: torch.Tensor | None = None
        self._prev_grad_norm: float | None = None
        self._prev_grad_velocity: float | None = None

    def compute(self, ctx) -> dict[str, float]:
        if ctx.accumulated_grads is None:
            return {}

        grad_norm = flatten_grads(ctx.accumulated_grads).norm().item()

        # Find embedding parameters from live model
        emb = None
        for name, param in ctx.model.named_parameters():
            if 'embedding' in name.lower() and 'weight' in name.lower():
                emb = param.data.clone().detach()
                break

        # Representation change rate
        if self._prev_emb is not None and emb is not None:
            rep_change = (emb - self._prev_emb).norm().item()
            rep_change_normalized = rep_change / (self._prev_emb.norm().item() + 1e-8)
        else:
            rep_change = 0.0
            rep_change_normalized = 0.0

        # Gradient velocity (first derivative)
        if self._prev_grad_norm is not None:
            grad_velocity = grad_norm - self._prev_grad_norm
        else:
            grad_velocity = 0.0

        # Gradient acceleration (second derivative)
        if self._prev_grad_velocity is not None:
            grad_acceleration = grad_velocity - self._prev_grad_velocity
        else:
            grad_acceleration = 0.0

        # Phase label from accuracy thresholds
        acc = ctx.val_acc
        if acc < 5:
            phase_code = 0  # pre_learning
        elif acc < 20:
            phase_code = 1  # early_learning
        elif acc < 80:
            phase_code = 2  # rapid_learning
        elif acc < 99:
            phase_code = 3  # refinement
        else:
            phase_code = 4  # converged

        # Update state
        if self._prev_emb is not None:
            del self._prev_emb
        self._prev_emb = emb
        self._prev_grad_norm = grad_norm
        self._prev_grad_velocity = grad_velocity

        return {
            'grad_velocity': grad_velocity,
            'grad_acceleration': grad_acceleration,
            'embedding_change': rep_change,
            'embedding_change_normalized': rep_change_normalized,
            'phase_code': phase_code,
        }

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        if self._prev_emb is not None:
            return {'prev_emb': self._prev_emb}
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        if 'prev_emb' in tensors:
            self._prev_emb = tensors['prev_emb']

    def reset(self):
        self._prev_emb = None
        self._prev_grad_norm = None
        self._prev_grad_velocity = None
