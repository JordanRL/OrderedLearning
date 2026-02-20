"""Counterfactual K-sufficiency validation hook.

Validates whether K shuffled epochs are sufficient for the counterfactual
decomposition by running K+1 shuffles and composing all C(K+1, K) subsets.

If the K+1 content norm is strictly lower than every K-subset norm, and
the K-subset solution alignments cluster tightly around the K+1 alignment,
then K is sufficient and K-based measurements provide:
  - Upper bound on true content component norm
  - Lower bound on true ordering component norm
"""

import itertools

import torch

from console import OLConsole
from console.utils import apply_style
from .base import MetricInfo, HookPoint, DebugInterventionHook, HookRegistry
from analysis_tools.utils import flatten_grads


@HookRegistry.register
class CounterfactualValidatorHook(DebugInterventionHook):
    """Validate K-sufficiency of the counterfactual decomposition.

    Runs K+1 shuffled epochs from the pre-epoch checkpoint, then composes
    all C(K+1, K) subsets of size K.  Compares each K-subset's content
    estimate (mean shuffled gradient) against the K+1 estimate to check
    convergence in both norm and direction.
    """

    name = "counterfactual_validator"
    description = "Validate K-sufficiency of the counterfactual decomposition"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {
        'epoch': {
            HookPoint.POST_EPOCH: (None, 49),
            HookPoint.SNAPSHOT: (50, None),
        },
    }
    needs_grads = True
    needs_pre_epoch_state = True
    needs_reference_weights = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo(
                'kp1_content_norm',
                'Norm of K+1 content estimate (mean shuffled gradient)',
                '||mean(g_shuffled, K+1)||_2',
                label='K+1 Content Norm',
            ),
            MetricInfo(
                'k_content_norm_max',
                'Max norm among K-subset content estimates',
                'max_i ||mean(g_shuffled, K, combo_i)||_2',
                label='K Content Norm (Max)',
            ),
            MetricInfo(
                'k_content_norm_min',
                'Min norm among K-subset content estimates',
                'min_i ||mean(g_shuffled, K, combo_i)||_2',
                label='K Content Norm (Min)',
            ),
            MetricInfo(
                'k_content_norm_mean',
                'Mean norm among K-subset content estimates',
                'mean_i ||mean(g_shuffled, K, combo_i)||_2',
                label='K Content Norm (Mean)',
            ),
            MetricInfo(
                'kp1_norm_strictly_lower',
                'Whether K+1 norm < all K-subset norms',
                '1.0 if true else 0.0',
                '1.0 = convergence confirmed',
                label='K+1 Strictly Lower',
            ),
            MetricInfo(
                'norm_convergence_gap',
                'Relative improvement from K to K+1',
                '(mean_k - kp1) / kp1',
                'smaller = K already near convergence',
                label='Norm Convergence Gap',
            ),
            MetricInfo(
                'k_to_kp1_cossim_mean',
                'Mean directional agreement between K and K+1 content vectors',
                'mean_i cos(mean_cf_k_i, mean_cf_kp1)',
                'closer to 1 = stable direction',
                label='K→K+1 Cos (Mean)',
            ),
            MetricInfo(
                'k_to_kp1_cossim_min',
                'Worst-case directional agreement',
                'min_i cos(mean_cf_k_i, mean_cf_kp1)',
                'closer to 1 = stable direction',
                label='K→K+1 Cos (Min)',
            ),
            MetricInfo(
                'kp1_cossim_to_solution',
                'K+1 content alignment to solution',
                'cos(-mean_cf_kp1, theta_ref - theta_prev)',
                '+ = content pushes toward solution',
                label='K+1→Solution Cos',
            ),
            MetricInfo(
                'k_cossim_to_solution_mean',
                'Mean K-subset content-to-solution alignment',
                'mean_i cos(-mean_cf_k_i, theta_ref - theta_prev)',
                label='K→Solution Cos (Mean)',
            ),
            MetricInfo(
                'k_cossim_to_solution_std',
                'Std of K-subset content-to-solution alignments',
                'std_i cos(-mean_cf_k_i, theta_ref - theta_prev)',
                'smaller = more stable',
                label='K→Solution Cos (Std)',
            ),
            MetricInfo(
                'k_cossim_to_solution_spread',
                'Spread of K-subset content-to-solution alignments',
                'max - min of cos values',
                'smaller = more stable',
                label='K→Solution Cos Spread',
            ),
        ]

    def __init__(self, k: int = 3):
        """
        Args:
            k: The base K to validate. Runs K+1 shuffles, composes
               C(K+1, K) subsets of size K.
        """
        self._k = k

    def intervene(self, run_ctx, model_ctx) -> dict[str, float]:
        actual_grads = run_ctx.accumulated_grads
        if actual_grads is None:
            return {}

        kp1 = self._k + 1

        # Save post-epoch state
        post_epoch_token = model_ctx.save_checkpoint()

        # Capture pre-epoch params for solution direction
        model_ctx.restore_pre_epoch()
        pre_epoch_params = {
            name: param.data.cpu().clone()
            for name, param in model_ctx.model.named_parameters()
        }

        # Run K+1 shuffled epochs from pre-epoch state
        console = OLConsole()
        task_name = "_cf_validator_shuffles"
        console.create_progress_task(
            task_name,
            apply_style("Validation shuffles", "status"),
            total=kp1,
        )

        shuffle_grads: list[dict[str, torch.Tensor]] = []
        for _ in range(kp1):
            model_ctx.restore_pre_epoch()
            shuffled_loader = model_ctx.get_shuffled_loader()
            cf_grad = model_ctx.run_training_epoch(shuffled_loader, step=True)
            cf_grad = {name: g.cpu() for name, g in cf_grad.items()}
            shuffle_grads.append(cf_grad)
            console.update_progress_task(task_name, advance=1)

        console.remove_progress_task(task_name)

        # Restore post-epoch state
        model_ctx.restore_checkpoint(post_epoch_token)
        model_ctx.discard_checkpoint(post_epoch_token)

        # --- Compute content estimates ---
        param_names = list(actual_grads.keys())
        device = model_ctx.device

        def _compute_mean(indices) -> dict[str, torch.Tensor]:
            mean = {name: torch.zeros_like(actual_grads[name]) for name in param_names}
            for i in indices:
                for name in param_names:
                    mean[name].add_(shuffle_grads[i][name].to(device))
            for name in param_names:
                mean[name].div_(len(indices))
            return mean

        # K+1 content vector (all shuffles)
        mean_kp1 = _compute_mean(range(kp1))
        flat_kp1 = flatten_grads(mean_kp1)
        norm_kp1 = flat_kp1.norm().item()

        # All C(K+1, K) subsets of size K
        combos = list(itertools.combinations(range(kp1), self._k))
        mean_k_dicts: list[dict[str, torch.Tensor]] = []
        flat_k_combos: list[torch.Tensor] = []
        norms_k: list[float] = []
        for combo in combos:
            mean_k = _compute_mean(combo)
            mean_k_dicts.append(mean_k)
            flat_k = flatten_grads(mean_k)
            flat_k_combos.append(flat_k)
            norms_k.append(flat_k.norm().item())

        # --- Norm validation ---
        metrics: dict[str, float] = {}
        metrics['kp1_content_norm'] = norm_kp1
        metrics['k_content_norm_max'] = max(norms_k)
        metrics['k_content_norm_min'] = min(norms_k)
        k_mean_norm = sum(norms_k) / len(norms_k)
        metrics['k_content_norm_mean'] = k_mean_norm
        metrics['kp1_norm_strictly_lower'] = 1.0 if norm_kp1 < min(norms_k) else 0.0
        metrics['norm_convergence_gap'] = (
            (k_mean_norm - norm_kp1) / (norm_kp1 + 1e-16)
        )

        # --- Directional validation (K vs K+1 content vectors) ---
        cossims_to_kp1: list[float] = []
        for flat_k in flat_k_combos:
            k_norm = flat_k.norm().item()
            cs = torch.dot(flat_k, flat_kp1).item() / (k_norm * norm_kp1 + 1e-12)
            cossims_to_kp1.append(cs)

        metrics['k_to_kp1_cossim_mean'] = sum(cossims_to_kp1) / len(cossims_to_kp1)
        metrics['k_to_kp1_cossim_min'] = min(cossims_to_kp1)

        # --- Solution alignment validation ---
        # Build matched solution direction (same approach as CounterfactualHook)
        self._ref.ensure_loaded(device)
        if self._ref.available:
            ref_weights = self._ref.weights

            sol_parts: list[torch.Tensor] = []
            kp1_parts: list[torch.Tensor] = []
            k_combo_parts: list[list[torch.Tensor]] = [[] for _ in combos]

            for name in sorted(param_names):
                if 'bias' in name:
                    continue
                ref_key = self._ref.resolve_key(name)
                if ref_key is None or name not in pre_epoch_params:
                    continue

                ref_w = ref_weights[ref_key].to(device)
                pre_w = pre_epoch_params[name].to(device)
                sol_part = (ref_w - pre_w).reshape(-1)

                if sol_part.norm().item() < 1e-12:
                    continue

                sol_parts.append(sol_part)
                kp1_parts.append((-mean_kp1[name]).reshape(-1))
                for ci, mean_k in enumerate(mean_k_dicts):
                    k_combo_parts[ci].append((-mean_k[name]).reshape(-1))

            if sol_parts:
                sol_dir = torch.cat(sol_parts)
                sol_norm = sol_dir.norm().item()

                if sol_norm > 1e-12:
                    def _cossim(parts: list[torch.Tensor]) -> float:
                        v = torch.cat(parts)
                        v_norm = v.norm().item()
                        if v_norm < 1e-12:
                            return 0.0
                        return torch.dot(v, sol_dir).item() / (v_norm * sol_norm)

                    metrics['kp1_cossim_to_solution'] = _cossim(kp1_parts)

                    k_sol_cossims = [_cossim(parts) for parts in k_combo_parts]
                    k_mean_cs = sum(k_sol_cossims) / len(k_sol_cossims)
                    metrics['k_cossim_to_solution_mean'] = k_mean_cs
                    metrics['k_cossim_to_solution_std'] = (
                        sum((cs - k_mean_cs) ** 2 for cs in k_sol_cossims)
                        / len(k_sol_cossims)
                    ) ** 0.5
                    metrics['k_cossim_to_solution_spread'] = (
                        max(k_sol_cossims) - min(k_sol_cossims)
                    )

        return metrics

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        pass

    def reset(self):
        pass
