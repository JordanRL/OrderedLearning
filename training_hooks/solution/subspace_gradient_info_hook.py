"""Gradient subspace and information content observer hook (merged)."""

import torch

from console import OLConsole
from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, GradientAvailability
from framework.utils import flatten_grads


@HookRegistry.register
class SubspaceGradientInfoHook(TrainingHook):
    """Analyze gradient subspace dimensionality and information content via SVD.

    Merges the former subspace_hook and gradient_info_hook into a single hook,
    since both performed the same SVD on a sliding window of gradients and
    extracted complementary metrics from the singular values.

    Reports:
    - Subspace dimensionality: dims_for_90pct, participation_ratio
    - Information concentration: top1/5/10 explained variance
    - Dominance: top singular value ratio, total variance
    """

    name = "subspace_gradient_info"
    description = "Gradient subspace dimensionality and information content (sliding window SVD)"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs = HookNeeds.ACCUMULATED_GRADS | HookNeeds.REFERENCE_WEIGHTS
    requires = HookRequirements(gradient_availability=GradientAvailability.GLOBAL_GRADIENTS)

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('dims_for_90pct', 'SVD dimensions needed for 90% explained variance',
                       'min k s.t. cumsum(sigma^2)[:k] >= 0.9', 'lower = more concentrated',
                       label='Dims for 90%'),
            MetricInfo('participation_ratio', 'Effective dimensionality of gradient subspace',
                       'sum(sigma^2)^2 / sum(sigma^4)', 'higher = more distributed',
                       label='Participation Ratio'),
            MetricInfo('top_sv_ratio', 'Dominance of leading singular value',
                       'sigma_1 / sum(sigma_i)', 'higher = more rank-1 like',
                       label='Top SV Ratio'),
            MetricInfo('svd_total_variance', 'Total variance captured by SVD', 'sum(sigma^2)',
                       label='Total Variance'),
            MetricInfo('top1_explained', 'Variance explained by top 1 component', 'sigma_1^2 / sum(sigma^2)',
                       label='Top-1 Explained'),
            MetricInfo('top5_explained', 'Cumulative variance explained by top 5', 'sum(sigma_1..5^2) / sum(sigma^2)',
                       label='Top-5 Explained'),
            MetricInfo('top10_explained', 'Cumulative variance explained by top 10', 'sum(sigma_1..10^2) / sum(sigma^2)',
                       label='Top-10 Explained'),
            MetricInfo('grad_energy_fraction_toward_solution',
                       'Fraction of gradient subspace energy directed toward solution',
                       'sum(sigma_i^2 * (v_i . sol_hat)^2) / sum(sigma_i^2)',
                       'higher = gradient energy aims at solution',
                       label='Energy→Solution'),
            MetricInfo('top{k}_energy_fraction_toward_solution',
                       'Energy fraction toward solution from top k components only (k=1,5,10)',
                       'sum(sigma_1..k^2 * (v_1..k . sol_hat)^2) / sum(sigma^2)',
                       label='Top-k Energy→Sol'),
        ]

    def __init__(self, window_size: int = 50, n_components: int = 20):
        """
        Args:
            window_size: Number of gradient snapshots to retain. Determines
                         temporal lookback and maximum detectable dimensionality.
            n_components: Number of singular values to compute via svd_lowrank.
                          Capped at current window length.
        """
        self._window_size = window_size
        self._n_components = n_components
        self._window: list[torch.Tensor] = []

    def compute(self, ctx, **state) -> dict[str, float]:
        gradient_state = state.get('gradient_state')
        if gradient_state is None or gradient_state.accumulated_grads is None:
            return {}
        accumulated_grads = gradient_state.accumulated_grads

        model_state = state.get('model_state')
        if model_state is None:
            return {}
        model = model_state.model

        self._window.append(flatten_grads(accumulated_grads))

        if len(self._window) > self._window_size:
            del self._window[0]

        # Need at least 2 gradients to compute meaningful subspace metrics
        if len(self._window) < 2:
            return {}

        grad_matrix = torch.stack(self._window)
        grad_matrix = grad_matrix - grad_matrix.mean(dim=0)

        n = len(self._window)
        q = min(self._n_components, n)

        try:
            _, S, V = torch.svd_lowrank(grad_matrix, q=q)

            sv_squared = S ** 2
            total_var = sv_squared.sum().item() + 1e-8
            explained_var = sv_squared / total_var
            # cumsum_cuda_kernel lacks a deterministic implementation;
            # compute on CPU (tiny vector of ~20 elements).
            cumulative_var = torch.cumsum(explained_var.cpu(), dim=0)

            # Subspace dimensionality metrics
            dims_90 = (cumulative_var < 0.9).sum().item() + 1
            participation_ratio = (sv_squared.sum().item() ** 2) / (
                (sv_squared ** 2).sum().item() + 1e-8
            )
            top_sv_ratio = S[0].item() / (S.sum().item() + 1e-8)

            # Information concentration metrics
            top1 = explained_var[0].item()
            top5 = cumulative_var[min(4, len(cumulative_var) - 1)].item()
            top10 = cumulative_var[min(9, len(cumulative_var) - 1)].item()

            metrics = {
                'dims_for_90pct': dims_90,
                'participation_ratio': participation_ratio,
                'top_sv_ratio': top_sv_ratio,
                'svd_total_variance': total_var,
                'top1_explained': top1,
                'top5_explained': top5,
                'top10_explained': top10,
            }

            # --- Solution direction in gradient subspace ---
            # Measure what fraction of the solution direction is captured
            # by the top-k gradient principal components.
            device = next(model.parameters()).device
            self._ref.ensure_loaded(device)
            if self._ref.available:
                ref_weights = self._ref.weights
                # Build solution direction with same ordering as flatten_grads
                sol_parts = []
                for name in sorted(accumulated_grads.keys()):
                    if 'bias' in name:
                        continue
                    ref_key = self._ref.resolve_key(name)
                    if ref_key is None:
                        continue
                    param = dict(model.named_parameters())[name]
                    diff = (ref_weights[ref_key] - param.data.float()).reshape(-1)
                    sol_parts.append(diff)

                if sol_parts:
                    sol_dir = torch.cat(sol_parts).to(V.device)
                    sol_norm = sol_dir.norm().item()

                    if sol_norm > 1e-12:
                        # How much of the gradient's energy is directed toward the solution?
                        # Weight each principal direction's alignment by its energy fraction.
                        sol_hat = sol_dir / sol_norm
                        projections = V.T @ sol_hat  # (q,) cosine with each direction
                        proj_sq = projections ** 2
                        total_var = (sv_squared.sum().item() + 1e-16)

                        metrics['grad_energy_fraction_toward_solution'] = (
                            (sv_squared * proj_sq).sum().item() / total_var
                        )

                        # Per top-k: energy fraction toward solution from just the top k components
                        for k in [1, 5, 10]:
                            if k <= len(proj_sq):
                                metrics[f'top{k}_energy_fraction_toward_solution'] = (
                                    (sv_squared[:k] * proj_sq[:k]).sum().item() / total_var
                                )

            return metrics
        except Exception as e:
            OLConsole().print_warning(f"SubspaceGradientInfoHook: SVD failed: {e}")
            return {}

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
