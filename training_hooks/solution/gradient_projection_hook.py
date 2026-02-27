"""Gradient projection onto known solution hook."""

import torch

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, GradientAvailability


@HookRegistry.register
class GradientProjectionHook(TrainingHook):
    """Measure gradient alignment and parameter movement toward a known solution.

    Gradient projection: cosine_similarity(-∂L/∂θ, θ_ref - θ_prev).
    Positive means gradient descent pushes toward the reference solution.

    Displacement projection: cosine_similarity(θ_current - θ_prev, θ_ref - θ_prev).
    Positive means parameters actually moved toward the reference, accounting
    for optimizer effects (Adam momentum, adaptive LR, weight decay).

    Comparing the two reveals how the optimizer interacts with the training
    signal — e.g., whether momentum amplifies or dampens the ordering effect.

    Requires shared reference weights (HookNeeds.REFERENCE_WEIGHTS).
    HookManager provides these automatically via ReferenceWeights.
    """

    name = "gradient_projection"
    description = "Gradient & displacement projection onto known solution"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs = HookNeeds.ACCUMULATED_GRADS | HookNeeds.REFERENCE_WEIGHTS
    requires = HookRequirements(gradient_availability=GradientAvailability.GLOBAL_GRADIENTS)

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('grad_cossim_to_solution/{layer}', 'Per-layer gradient alignment to solution',
                       'cos(-dL/dtheta, theta_ref - theta_prev)', '+ = toward solution',
                       label='Layer Grad→Solution'),
            MetricInfo('disp_cossim_to_solution/{layer}', 'Per-layer displacement alignment to solution',
                       'cos(theta_curr - theta_prev, theta_ref - theta_prev)', '+ = moved toward solution',
                       label='Layer Disp→Solution'),
            MetricInfo('overall_grad_cossim_to_solution', 'All-parameter gradient alignment to solution',
                       'cos(cat(-grad), cat(theta_ref - theta_prev))', '+ = toward solution',
                       label='Grad→Solution Cos'),
            MetricInfo('overall_disp_cossim_to_solution', 'All-parameter displacement alignment to solution',
                       'cos(cat(displacement), cat(theta_ref - theta_prev))', '+ = moved toward solution',
                       label='Disp→Solution Cos'),
            MetricInfo('mean_layer_grad_cossim_to_solution', 'Mean of per-layer gradient alignments',
                       'mean(grad_cossim per layer)',
                       label='Mean Layer Grad→Sol'),
            MetricInfo('mean_layer_disp_cossim_to_solution', 'Mean of per-layer displacement alignments',
                       'mean(disp_cossim per layer)',
                       label='Mean Layer Disp→Sol'),
            MetricInfo('displacement_norm', 'Total parameter displacement this epoch',
                       '||theta_curr - theta_prev||_2',
                       label='Displacement Norm'),
            MetricInfo('distance_to_reference', 'Current distance from reference solution',
                       '||theta_ref - theta_curr||_2', 'lower = closer to solution',
                       label='Distance to Solution'),
        ]

    def __init__(self):
        self._prev_params: dict[str, torch.Tensor] | None = None

    def reset(self):
        self._prev_params = None

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        if self._prev_params is not None:
            return {f'prev/{k}': v for k, v in self._prev_params.items()}
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        prefix = 'prev/'
        restored = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        if restored:
            self._prev_params = restored

    def compute(self, ctx, **state) -> dict[str, float]:
        gradient_state = state.get('gradient_state')
        if gradient_state is None or gradient_state.accumulated_grads is None:
            return {}

        model_state = state.get('model_state')
        if model_state is None:
            return {}
        model = model_state.model

        device = next(model.parameters()).device
        self._ref.ensure_loaded(device)

        if not self._ref.available:
            return {}

        ref_weights = self._ref.weights
        grads = gradient_state.accumulated_grads
        has_prev = self._prev_params is not None
        metrics = {}

        # Collect per-layer data in a single pass
        all_grad_flat = []
        all_dir_flat = []
        all_disp_flat = []
        all_prev_dir_flat = []
        grad_projections = []
        disp_projections = []

        for name, param in model.named_parameters():
            ref_key = self._ref.resolve_key(name)
            if ref_key is None:
                continue

            ref_w = ref_weights[ref_key]
            current_w = param.data.float()
            is_weight = param.dim() >= 2 and 'bias' not in name

            # Both projections use prev_params as the starting point
            if has_prev and name in self._prev_params:
                prev_w = self._prev_params[name].to(device)

                # Direction from previous weights toward reference
                prev_direction = ref_w - prev_w
                prev_dir_flat = prev_direction.reshape(-1)
                prev_d_norm = prev_dir_flat.norm().item()

                # --- Gradient projection ---
                if name in grads and prev_d_norm > 1e-12:
                    grad_flat = -grads[name].float().reshape(-1)  # Negate: use descent direction
                    g_norm = grad_flat.norm().item()

                    if g_norm > 1e-12:
                        cos_sim = torch.dot(grad_flat, prev_dir_flat).item() / (g_norm * prev_d_norm)
                        if is_weight:
                            metrics[f'grad_cossim_to_solution/{name}'] = cos_sim
                        grad_projections.append(cos_sim)

                    all_grad_flat.append(grad_flat)
                    all_dir_flat.append(prev_dir_flat)

                # --- Displacement projection ---
                displacement = current_w - prev_w
                disp_flat = displacement.reshape(-1)
                disp_norm = disp_flat.norm().item()

                if prev_d_norm > 1e-12 and disp_norm > 1e-12:
                    disp_cos = torch.dot(disp_flat, prev_dir_flat).item() / (disp_norm * prev_d_norm)
                    if is_weight:
                        metrics[f'disp_cossim_to_solution/{name}'] = disp_cos
                    disp_projections.append(disp_cos)

                all_disp_flat.append(disp_flat)
                all_prev_dir_flat.append(prev_dir_flat)

        # Snapshot current params for next epoch's displacement
        self._prev_params = {
            name: param.data.cpu().clone()
            for name, param in model.named_parameters()
        }

        # --- Distance (always available, even on first epoch) ---
        dist_sq = 0.0
        for name, param in model.named_parameters():
            ref_key = self._ref.resolve_key(name)
            if ref_key is not None:
                diff = ref_weights[ref_key] - param.data.float()
                dist_sq += diff.norm().item() ** 2
        metrics['distance_to_reference'] = dist_sq ** 0.5

        # --- Overall gradient projection ---
        if all_grad_flat:
            g_cat = torch.cat(all_grad_flat)
            d_cat = torch.cat(all_dir_flat)
            metrics['overall_grad_cossim_to_solution'] = (
                torch.dot(g_cat, d_cat).item() / (g_cat.norm().item() * d_cat.norm().item() + 1e-8)
            )
            metrics['mean_layer_grad_cossim_to_solution'] = sum(grad_projections) / len(grad_projections)

        # --- Overall displacement projection ---
        if all_disp_flat:
            disp_cat = torch.cat(all_disp_flat)
            pd_cat = torch.cat(all_prev_dir_flat)
            metrics['overall_disp_cossim_to_solution'] = (
                torch.dot(disp_cat, pd_cat).item() / (disp_cat.norm().item() * pd_cat.norm().item() + 1e-8)
            )
            metrics['displacement_norm'] = disp_cat.norm().item()
            metrics['mean_layer_disp_cossim_to_solution'] = sum(disp_projections) / len(disp_projections)

        return metrics
