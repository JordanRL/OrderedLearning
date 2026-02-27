"""Gradient norms observer hook."""

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, GradientAvailability
from framework.utils import flatten_grads


@HookRegistry.register
class NormsHook(TrainingHook):
    """Compute gradient magnitude metrics at snapshot epochs.

    Ported from analysis_tools/norms.py — same computation, live data.
    """

    name = "norms"
    description = "Gradient magnitude dynamics (L2 norm, max/mean components, per-layer)"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs = HookNeeds.ACCUMULATED_GRADS
    requires = HookRequirements(gradient_availability=GradientAvailability.GLOBAL_GRADIENTS)

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('total_norm', 'L2 norm of full flattened gradient', '||g||_2', label='Grad Norm'),
            MetricInfo('max_component', 'Largest absolute gradient component', 'max(|g_i|)', label='Max Grad Component'),
            MetricInfo('mean_component', 'Mean absolute gradient component', 'mean(|g_i|)', label='Mean Grad Component'),
            MetricInfo('norm_{layer}', 'Per-layer gradient L2 norm (weight layers only)', '||g_layer||_2', label='Layer Grad Norm'),
        ]

    def compute(self, ctx, **state) -> dict[str, float]:
        gradient_state = state.get('gradient_state')
        if gradient_state is None or gradient_state.accumulated_grads is None:
            return {}

        grads = gradient_state.accumulated_grads
        flat = flatten_grads(grads)

        metrics = {
            'total_norm': flat.norm().item(),
            'max_component': flat.abs().max().item(),
            'mean_component': flat.abs().mean().item(),
        }

        # Per-layer norms (excluding bias)
        for name, grad in grads.items():
            if 'bias' not in name:
                metrics[f'norm_{name}'] = grad.norm().item()

        return metrics
