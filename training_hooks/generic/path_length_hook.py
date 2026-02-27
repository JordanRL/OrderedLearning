"""Cumulative path length and displacement tracking hook."""

import torch

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookRequirements, ModelCapability


@HookRegistry.register
class PathLengthHook(TrainingHook):
    """Track cumulative path length and net displacement in parameter space.

    Accumulates the L2 norm of per-step parameter changes at POST_STEP,
    and emits metrics at SNAPSHOT intervals. This gives the true path
    length (sum of step-wise deltas), not the net displacement between
    snapshots.

    Metrics:
        path_length: cumulative sum of ||theta_{t+1} - theta_t||
        net_displacement: ||theta_t - theta_0||
        path_efficiency: net_displacement / path_length (1.0 = straight line)
    """

    name = "path_length"
    description = "Cumulative path length and displacement in parameter space"
    hook_points = {HookPoint.POST_STEP, HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    loop_points = {
        'epoch': {HookPoint.POST_STEP, HookPoint.POST_EPOCH},
        'step': {HookPoint.POST_STEP, HookPoint.SNAPSHOT},
    }
    requires = HookRequirements(model_capabilities=ModelCapability.PARAMETERS)

    def __init__(self, **kwargs):
        self._prev_params: torch.Tensor | None = None
        self._init_params: torch.Tensor | None = None
        self._path_length = 0.0

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('path_length',
                       'Cumulative L2 norm of per-step parameter changes',
                       'Sum ||theta_{t+1} - theta_t||_2',
                       sign_info='>=0, monotonically increasing',
                       label='Path Length'),
            MetricInfo('net_displacement',
                       'L2 distance from initial parameters',
                       '||theta_t - theta_0||_2',
                       sign_info='>=0',
                       label='Net Displacement'),
            MetricInfo('path_efficiency',
                       'Ratio of net displacement to path length (1.0 = straight line)',
                       '||theta_t - theta_0|| / path_length',
                       sign_info='[0, 1], higher = more direct path',
                       label='Path Efficiency'),
        ]

    def compute(self, ctx, **state) -> dict[str, float]:
        model_state = state.get('model_state')
        if model_state is None:
            return {}
        model = model_state.model

        # Flatten all parameters into a single vector
        params = torch.cat([
            p.data.detach().reshape(-1) for p in model.parameters()
        ])

        if self._prev_params is None:
            # First call — store initial state
            self._prev_params = params.detach()
            self._init_params = params.cpu()
            return {}

        # Accumulate path length
        self._path_length += torch.linalg.norm(
            params - self._prev_params
        ).item()
        self._prev_params = params.detach()

        # Only emit metrics at SNAPSHOT (step loop) or POST_EPOCH (epoch loop)
        if ctx.hook_point == HookPoint.POST_STEP:
            return {}

        # Compute net displacement from initial params
        init = self._init_params.to(params.device)
        net_disp = torch.linalg.norm(params - init).item()

        return {
            'path_length': self._path_length,
            'net_displacement': net_disp,
            'path_efficiency': (
                net_disp / self._path_length
                if self._path_length > 0 else 1.0
            ),
        }

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        if self._prev_params is not None:
            tensors['prev_params'] = self._prev_params
        if self._init_params is not None:
            tensors['init_params'] = self._init_params
        return tensors

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        if 'prev_params' in tensors:
            self._prev_params = tensors['prev_params']
        if 'init_params' in tensors:
            self._init_params = tensors['init_params']

    def reset(self):
        self._prev_params = None
        self._init_params = None
        self._path_length = 0.0
