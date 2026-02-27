"""Parameter update magnitude tracking hook."""

import torch

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookRequirements, ModelCapability


@HookRegistry.register
class ParameterDeltaHook(TrainingHook):
    """Track the magnitude of parameter changes per step/epoch.

    Compares current parameters to stored previous parameters, computing
    relative and absolute deltas plus the current parameter norm.

    Works with any model architecture (no gradient or batch format
    requirements). First call returns empty dict (no previous params).
    """

    name = "parameter_delta"
    description = "Parameter update magnitude tracking"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    loop_points = {
        'epoch': {HookPoint.POST_EPOCH},
        'step': {HookPoint.SNAPSHOT},
    }
    requires = HookRequirements(model_capabilities=ModelCapability.PARAMETERS)

    def __init__(self, **kwargs):
        self._prev_params: torch.Tensor | None = None

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('relative_delta',
                       'Relative parameter change: ||theta_new - theta_old|| / ||theta_old||',
                       '||Δθ|| / ||θ_old||',
                       sign_info='≥0, larger = bigger update',
                       label='Relative Δθ'),
            MetricInfo('absolute_delta',
                       'Absolute parameter change L2 norm',
                       '||θ_new - θ_old||_2',
                       sign_info='≥0',
                       label='Absolute Δθ'),
            MetricInfo('param_norm',
                       'Current parameter L2 norm',
                       '||θ_new||_2',
                       sign_info='≥0',
                       label='Param Norm'),
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

        param_norm = torch.linalg.norm(params).item()

        if self._prev_params is None:
            # First call — store and return empty
            self._prev_params = params.cpu()
            return {}

        prev = self._prev_params.to(params.device)
        delta = params - prev
        abs_delta = torch.linalg.norm(delta).item()
        prev_norm = torch.linalg.norm(prev).item()

        rel_delta = abs_delta / prev_norm if prev_norm > 0 else 0.0

        # Store current params on CPU for next comparison
        self._prev_params = params.cpu()

        return {
            'relative_delta': rel_delta,
            'absolute_delta': abs_delta,
            'param_norm': param_norm,
        }

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        if self._prev_params is not None:
            return {'prev_params': self._prev_params}
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        if 'prev_params' in tensors:
            self._prev_params = tensors['prev_params']

    def reset(self):
        self._prev_params = None
