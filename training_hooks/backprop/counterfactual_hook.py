"""Counterfactual ordering analysis intervention hook."""

import torch

from console import OLConsole
from console.utils import apply_style
from framework.hooks import MetricInfo, HookPoint, InterventionHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, TrainingParadigm, GradientAvailability


@HookRegistry.register
class CounterfactualHook(InterventionHook):
    """Counterfactual ordering analysis via shuffled epoch comparisons.

    At each snapshot epoch:
    1. Restores pre-epoch state (same starting weights as the real epoch)
    2. Runs K shuffled training epochs, collecting gradients
    3. Compares actual gradients to mean shuffled gradients
    4. Restores actual post-epoch state and continues

    The "ordering contribution" is the difference between the actual
    gradient (from structured ordering) and the mean gradient from
    randomly shuffled orderings. This measures how much the data
    ordering contributes to the gradient signal.

    When reference weights are available, also measures how each gradient
    component aligns with the direction toward the known solution. This
    reveals whether the ordering-specific signal specifically steers
    toward the solution basin.
    """

    name = "counterfactual"
    description = "Counterfactual ordering analysis via shuffled epoch comparisons"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {
        'epoch': {
            HookPoint.POST_EPOCH: (None, 49),
            HookPoint.SNAPSHOT: (50, None),
        },
    }
    needs = HookNeeds.ACCUMULATED_GRADS | HookNeeds.REFERENCE_WEIGHTS | HookNeeds.PRE_EPOCH_STATE
    requires = HookRequirements(
        paradigm=TrainingParadigm.BACKPROP,
        gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
    )

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('counterfactual_mean_norm', 'Norm of mean counterfactual gradient',
                       '||mean(g_shuffled)||_2',
                       label='CF Mean Norm'),
            MetricInfo('content_component_norm', 'Norm of content component (ordering-invariant signal)',
                       '||proj(g_actual onto g_cf)||_2',
                       label='Content Norm'),
            MetricInfo('ordering_component_norm', 'Norm of ordering-specific component',
                       '||g_actual - content||_2',
                       label='Ordering Norm'),
            MetricInfo('ordering_fraction', 'Fraction of gradient energy from ordering',
                       '||ordering||^2 / ||actual||^2', '0 = all content, 1 = all ordering',
                       label='Ordering Fraction'),
            MetricInfo('ordering_alignment', 'Cosine similarity between actual and counterfactual gradients',
                       'cos(g_actual, g_cf)', '1 = ordering has no effect',
                       label='Ordering Alignment'),
            MetricInfo('content_grad_cossim_to_solution', 'Content component alignment to solution',
                       'cos(-content, theta_ref - theta_prev)', '+ = content pushes toward solution',
                       label='Content→Solution Cos'),
            MetricInfo('ordering_grad_cossim_to_solution', 'Ordering component alignment to solution',
                       'cos(-ordering, theta_ref - theta_prev)', '+ = ordering pushes toward solution',
                       label='Ordering→Solution Cos'),
            MetricInfo('cf_grad_cossim_to_solution', 'Counterfactual gradient alignment to solution',
                       'cos(-g_cf, theta_ref - theta_prev)', '+ = any-ordering pushes toward solution',
                       label='CF→Solution Cos'),
            MetricInfo('content_energy_fraction_toward_solution',
                       'Content subspace energy directed toward solution (sliding window SVD)',
                       'sum(sigma_i^2 * (v_i . sol_hat)^2) / sum(sigma_i^2)',
                       'higher = content energy aims at solution',
                       label='Content Energy→Sol'),
            MetricInfo('ordering_energy_fraction_toward_solution',
                       'Ordering subspace energy directed toward solution (sliding window SVD)',
                       'sum(sigma_i^2 * (v_i . sol_hat)^2) / sum(sigma_i^2)',
                       'higher = ordering energy aims at solution',
                       label='Ordering Energy→Sol'),
        ]

    def __init__(self, k: int = 3, window_size: int = 20):
        """
        Args:
            k: Number of counterfactual shuffled epochs per snapshot.
            window_size: Number of past content/ordering vectors to retain
                         for subspace analysis via SVD.
        """
        self.k = k
        self._window_size = window_size
        self._content_window: list[torch.Tensor] = []
        self._ordering_window: list[torch.Tensor] = []

    def intervene(self, ctx, model_ctx, **state) -> dict[str, float]:
        if self.k <= 0:
            return {}

        gradient_state = state.get('gradient_state')
        if gradient_state is None:
            return {}
        actual_grads = gradient_state.accumulated_grads
        if actual_grads is None:
            return {}

        # Save current (post-epoch) state so we can restore after
        post_epoch_token = model_ctx.save_checkpoint()

        # Capture pre-epoch params for reference direction alignment
        model_ctx.restore_pre_epoch()
        pre_epoch_params = {
            name: param.data.cpu().clone()
            for name, param in model_ctx.model.named_parameters()
        }

        # Run K shuffled epochs
        console = OLConsole()
        task_name = "_counterfactual_shuffles"
        console.create_progress_task(
            task_name,
            apply_style("Counterfactual shuffles", "status"),
            total=self.k,
        )

        counterfactual_grads = []
        for _ in range(self.k):
            # Restore to pre-epoch state so each shuffle starts from the
            # same weights the actual training epoch started from
            model_ctx.restore_pre_epoch()

            # Create shuffled loader and run a training epoch
            shuffled_loader = model_ctx.get_shuffled_loader()
            cf_grad = model_ctx.run_training_epoch(shuffled_loader, step=True)

            # Move to CPU to avoid accumulating VRAM
            cf_grad = {name: g.cpu() for name, g in cf_grad.items()}
            counterfactual_grads.append(cf_grad)
            console.update_progress_task(task_name, advance=1)

        console.remove_progress_task(task_name)

        # Restore actual post-epoch state for training to continue
        model_ctx.restore_checkpoint(post_epoch_token)
        model_ctx.discard_checkpoint(post_epoch_token)

        # Compute mean counterfactual gradient
        mean_cf_grad = {
            name: torch.zeros_like(actual_grads[name])
            for name in actual_grads
        }
        for cf_grad in counterfactual_grads:
            for name in mean_cf_grad:
                mean_cf_grad[name].add_(cf_grad[name].to(mean_cf_grad[name].device))
        for name in mean_cf_grad:
            mean_cf_grad[name].div_(len(counterfactual_grads))

        # Norms and dot product
        actual_norm_sq = sum(g.norm().item() ** 2 for g in actual_grads.values())
        cf_norm_sq = sum(g.norm().item() ** 2 for g in mean_cf_grad.values())
        dot_product = sum(
            (actual_grads[n] * mean_cf_grad[n]).sum().item()
            for n in actual_grads
        )

        actual_norm = actual_norm_sq ** 0.5
        cf_norm = cf_norm_sq ** 0.5

        # Projection-based decomposition:
        #   content  = proj(g_actual onto g_cf) — what any random ordering would produce
        #   ordering = g_actual - content         — signal from the specific ordering
        # These are orthogonal: ||content||² + ||ordering||² = ||actual||²
        proj_coeff = dot_product / (cf_norm_sq + 1e-16)
        content_norm_sq = dot_product ** 2 / (cf_norm_sq + 1e-16)
        ordering_norm_sq = max(0.0, actual_norm_sq - content_norm_sq)

        # Ordering fraction: variance explained by ordering, bounded [0, 1]
        ordering_fraction = ordering_norm_sq / (actual_norm_sq + 1e-16)

        # Alignment: cosine between actual and counterfactual gradients
        alignment = dot_product / (actual_norm * cf_norm + 1e-8)

        metrics = {
            'counterfactual_mean_norm': cf_norm,
            'content_component_norm': content_norm_sq ** 0.5,
            'ordering_component_norm': ordering_norm_sq ** 0.5,
            'ordering_fraction': ordering_fraction,
            'ordering_alignment': alignment,
        }

        # Per-parameter decomposition
        for name in actual_grads:
            if 'bias' in name or name not in mean_cf_grad:
                continue
            p_actual = actual_grads[name].reshape(-1)
            p_cf = mean_cf_grad[name].reshape(-1)
            p_actual_norm_sq = p_actual.norm().item() ** 2
            p_cf_norm_sq = p_cf.norm().item() ** 2
            p_dot = torch.dot(p_actual, p_cf).item()
            p_content_norm_sq = p_dot ** 2 / (p_cf_norm_sq + 1e-16)
            p_ordering_norm_sq = max(0.0, p_actual_norm_sq - p_content_norm_sq)
            metrics[f'content_component_norm/{name}'] = p_content_norm_sq ** 0.5
            metrics[f'ordering_component_norm/{name}'] = p_ordering_norm_sq ** 0.5
            metrics[f'ordering_fraction/{name}'] = (
                p_ordering_norm_sq / (p_actual_norm_sq + 1e-16)
            )
            p_actual_norm = p_actual_norm_sq ** 0.5
            p_cf_norm = p_cf_norm_sq ** 0.5
            if p_actual_norm > 1e-12 and p_cf_norm > 1e-12:
                metrics[f'ordering_alignment/{name}'] = (
                    p_dot / (p_actual_norm * p_cf_norm)
                )

        # --- Reference direction alignment ---
        # Measure how each gradient component aligns with the direction
        # toward the known solution from the pre-epoch starting point.
        # Uses negated gradients so positive = toward solution.
        self._ref.ensure_loaded(model_ctx.device)
        if self._ref.available:
            ref_weights = self._ref.weights
            device = model_ctx.device

            all_content = []
            all_ordering = []
            all_cf = []
            all_ref_dir = []
            included_names = []  # Track parameter order for subspace analysis

            for name in sorted(actual_grads.keys()):
                ref_key = self._ref.resolve_key(name)
                if ref_key is None or name not in pre_epoch_params:
                    continue

                ref_w = ref_weights[ref_key]
                pre_w = pre_epoch_params[name].to(device)
                ref_dir = (ref_w - pre_w).reshape(-1)

                if ref_dir.norm().item() < 1e-12:
                    continue

                g_actual = -actual_grads[name].reshape(-1)
                g_cf = -mean_cf_grad[name].reshape(-1)
                g_content = proj_coeff * g_cf
                g_ordering = g_actual - g_content

                # Per-parameter solution alignment
                if 'bias' not in name:
                    ref_dir_norm = ref_dir.norm().item()
                    g_content_norm = g_content.norm().item()
                    g_ordering_norm = g_ordering.norm().item()
                    g_cf_norm = g_cf.norm().item()
                    if g_content_norm > 1e-12 and ref_dir_norm > 1e-12:
                        metrics[f'content_grad_cossim_to_solution/{name}'] = (
                            torch.dot(g_content, ref_dir).item()
                            / (g_content_norm * ref_dir_norm)
                        )
                    if g_ordering_norm > 1e-12 and ref_dir_norm > 1e-12:
                        metrics[f'ordering_grad_cossim_to_solution/{name}'] = (
                            torch.dot(g_ordering, ref_dir).item()
                            / (g_ordering_norm * ref_dir_norm)
                        )
                    if g_cf_norm > 1e-12 and ref_dir_norm > 1e-12:
                        metrics[f'cf_grad_cossim_to_solution/{name}'] = (
                            torch.dot(g_cf, ref_dir).item()
                            / (g_cf_norm * ref_dir_norm)
                        )

                included_names.append(name)
                all_content.append(g_content)
                all_ordering.append(g_ordering)
                all_cf.append(g_cf)
                all_ref_dir.append(ref_dir)

            if all_ref_dir:
                ref_cat = torch.cat(all_ref_dir)
                ref_norm = ref_cat.norm().item()

                def _cossim(vecs):
                    v = torch.cat(vecs)
                    v_norm = v.norm().item()
                    if v_norm < 1e-12 or ref_norm < 1e-12:
                        return 0.0
                    return torch.dot(v, ref_cat).item() / (v_norm * ref_norm)

                metrics['content_grad_cossim_to_solution'] = _cossim(all_content)
                metrics['ordering_grad_cossim_to_solution'] = _cossim(all_ordering)
                metrics['cf_grad_cossim_to_solution'] = _cossim(all_cf)

                # --- Subspace analysis (sliding window) ---
                # Accumulate flattened content and ordering vectors across
                # epochs. SVD each window to find the subspace each component
                # has explored, then measure how much of the solution
                # direction each subspace captures.
                content_flat = torch.cat(all_content).cpu()
                ordering_flat = torch.cat(all_ordering).cpu()

                self._content_window.append(content_flat)
                self._ordering_window.append(ordering_flat)
                if len(self._content_window) > self._window_size:
                    del self._content_window[0]
                    del self._ordering_window[0]

                if len(self._content_window) >= 2 and ref_norm > 1e-12:
                    sol = ref_cat
                    sol_norm_sq = ref_norm ** 2

                    def _weighted_subspace_alignment(window):
                        mat = torch.stack(window).to(sol.device)  # (N, n_params)
                        q = min(len(window), mat.shape[1])
                        try:
                            _, S, V = torch.svd_lowrank(mat, q=q)
                            sol_hat = sol / (sol.norm() + 1e-12)
                            proj = V.T @ sol_hat  # (q,) cosine of each direction with solution
                            sv_sq = S ** 2
                            # Weight each direction's alignment by its energy fraction
                            return (sv_sq * proj ** 2).sum().item() / (sv_sq.sum().item() + 1e-16)
                        except Exception as e:
                            OLConsole().print_warning(f"CounterfactualHook: subspace SVD failed: {e}")
                            return None

                    content_capture = _weighted_subspace_alignment(self._content_window)
                    ordering_capture = _weighted_subspace_alignment(self._ordering_window)

                    if content_capture is not None:
                        metrics['content_energy_fraction_toward_solution'] = content_capture
                    if ordering_capture is not None:
                        metrics['ordering_energy_fraction_toward_solution'] = ordering_capture

        return metrics

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        for i, t in enumerate(self._content_window):
            tensors[f'content_window_{i}'] = t
        for i, t in enumerate(self._ordering_window):
            tensors[f'ordering_window_{i}'] = t
        return tensors

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        self._content_window = [
            tensors[f'content_window_{i}']
            for i in range(len(tensors))
            if f'content_window_{i}' in tensors
        ]
        self._ordering_window = [
            tensors[f'ordering_window_{i}']
            for i in range(len(tensors))
            if f'ordering_window_{i}' in tensors
        ]

    def reset(self):
        self._content_window = []
        self._ordering_window = []
