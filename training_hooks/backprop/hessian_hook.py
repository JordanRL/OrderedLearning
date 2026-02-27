"""Per-step entanglement term measurement via finite-difference Hv product.

Measures the cross-batch entanglement term H_B * g_A from the paper's
theoretical decomposition:

    nabla L_B(theta') ~ nabla L_B(theta) - eta * H_B(theta) * nabla L_A(theta)
                         ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         content term        entanglement term

Uses a dual-phase design:
  - PRE_STEP: captures the pre-step model state (theta before train_step).
  - POST_STEP: restores to the *previous* pre-step state (theta_{N-1}) and
    computes H_B(theta_{N-1}) * g_A(theta_{N-1}) via finite-difference.

Fires in configurable bursts: every_n_steps apart, each burst lasting
burst_length consecutive steps. The first step in each burst captures
state but produces no metrics (no previous checkpoint yet), so a burst
of 11 yields 10 data points.
"""

import copy

import torch

from framework.hooks import MetricInfo, HookPoint, StepSchedule, InterventionHook, HookRegistry
from framework.capabilities import HookNeeds, HookRequirements, TrainingParadigm, GradientAvailability
from framework.utils import flatten_grads, flatten_params


@HookRegistry.register
class HessianHook(InterventionHook):
    """Estimate the per-step entanglement term H_B * g_A.

    Uses finite-difference Hv approximation on the current batch (B) with
    the previous step's gradient direction (g_A):

        H_B * g_A ~ (nabla L_B(theta + eps * g_A_hat) - nabla L_B(theta)) / eps * ||g_A||

    Requires two extra forward-backward passes on a single batch per
    measurement. Uses BackpropInterventionContext for checkpoint save/restore.
    """

    name = "hessian"
    description = "Per-step entanglement term (H_B * g_A) via finite-difference Hv"
    hook_points = {HookPoint.PRE_STEP, HookPoint.POST_STEP}
    loop_points = {
        'epoch': {HookPoint.PRE_STEP, HookPoint.POST_STEP},
        'step': {HookPoint.PRE_STEP, HookPoint.POST_STEP},
    }
    needs = HookNeeds.PREV_STEP_GRADS | HookNeeds.REFERENCE_WEIGHTS
    requires = HookRequirements(
        paradigm=TrainingParadigm.BACKPROP,
        gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
    )

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo(
                'entanglement_norm',
                'L2 norm of the entanglement term eta * H_B * g_A',
                '||eta * H_B * g_A||_2',
                'larger = stronger ordering signal',
                label='Entanglement Norm',
            ),
            MetricInfo(
                'content_norm',
                'L2 norm of the estimated content term (ordering-invariant)',
                '||g_B + eta * H_B * g_A||_2',
                '',
                label='Content Norm',
            ),
            MetricInfo(
                'observed_grad_norm',
                'L2 norm of the observed batch gradient g_B',
                '||g_B||_2',
                '',
                label='Observed Grad Norm',
            ),
            MetricInfo(
                'entanglement_energy_ratio',
                'Energy ratio: how much ordering dominates',
                '||ent||^2 / ||g_B||^2',
                '0 = no ordering effect, >1 = ordering dominates',
                label='Entanglement Energy',
            ),
            MetricInfo(
                'entanglement_content_cossim',
                'Cosine similarity between entanglement and content terms',
                'cos(ent, content)',
                '+ = ordering reinforces content',
                label='Ent-Content Cos Sim',
            ),
            MetricInfo(
                'rayleigh_quotient',
                'Rayleigh quotient along g_A direction',
                'g_A^T H_B g_A / ||g_A||^2',
                '+ = positive curvature in A direction',
                label='Rayleigh Quotient',
            ),
            MetricInfo(
                'amplification_ratio',
                'Hessian amplification of g_A',
                '||H_B g_A|| / ||g_A||',
                '>1 = Hessian amplifies A gradient',
                label='Hessian Amplification',
            ),
            MetricInfo(
                'edge_of_stability',
                'Stability boundary indicator',
                'amplification * 2 * eta',
                '>1 = beyond stability boundary',
                label='Edge of Stability',
            ),
            MetricInfo(
                'entanglement_cossim_to_solution',
                'Cosine similarity of -entanglement toward known solution',
                'cos(-ent, theta_ref - theta)',
                '+ = ordering pushes toward solution',
                label='Ent→Solution Cos',
            ),
            MetricInfo(
                'content_cossim_to_solution',
                'Cosine similarity of -content toward known solution',
                'cos(-content, theta_ref - theta)',
                '+ = content pushes toward solution',
                label='Content→Solution Cos',
            ),
            MetricInfo(
                'entanglement_coherence',
                'Cosine similarity between consecutive entanglement vectors',
                'cos(ent_t, ent_{t-1})',
                '+ = consistent ordering direction',
                label='Ent Coherence',
            ),
        ]

    def __init__(
        self,
        epsilon: float = 1e-4,
        every_n_steps: int = 1000,
        burst_length: int = 11,
    ):
        """
        Args:
            epsilon: Perturbation magnitude for finite-difference approximation.
            every_n_steps: Spacing between measurement bursts. First burst
                fires at step every_n_steps (not step 0).
            burst_length: Number of consecutive steps per burst. The first
                step captures state but produces no metrics, so a burst of
                11 yields 10 data points.
        """
        self._epsilon = epsilon

        # Step schedule (overrides class default based on constructor params)
        self.step_schedule = StepSchedule(
            mode='burst',
            stride=every_n_steps,
            burst_length=burst_length,
            warmup=every_n_steps,
        )

        # Full pre-step checkpoints (model + optimizer + scheduler) for
        # correct Hessian evaluation at theta_{N-1}. Kept on GPU.
        self._current_checkpoint: dict | None = None
        self._previous_checkpoint: dict | None = None

        # Entanglement coherence tracking
        self._prev_entanglement_flat: torch.Tensor | None = None

    def _save_full_state(self, model_ctx) -> dict:
        """Capture full training state on GPU."""
        return {
            'model': {k: v.clone() for k, v in model_ctx.model.state_dict().items()},
            'optimizer': copy.deepcopy(model_ctx._optimizer.state_dict()),
            'scheduler': model_ctx._scheduler.state_dict(),
        }

    def _restore_full_state(self, state: dict, model_ctx):
        """Restore full training state into model_ctx's model/optimizer/scheduler."""
        model_ctx.model.load_state_dict(state['model'])
        model_ctx._optimizer.load_state_dict(state['optimizer'])
        model_ctx._scheduler.load_state_dict(state['scheduler'])

    def intervene(self, ctx, model_ctx, **state) -> dict[str, float]:
        # PRE_STEP: rotate checkpoints, capture full pre-step state
        if ctx.hook_point == HookPoint.PRE_STEP:
            self._previous_checkpoint = self._current_checkpoint
            self._current_checkpoint = self._save_full_state(model_ctx)
            return {}

        # POST_STEP: compute Hessian at previous pre-step state
        if self._previous_checkpoint is None:
            return {}

        # Get g_A (previous step's gradient, computed at theta_{N-1})
        gradient_state = state.get('gradient_state')
        if gradient_state is None:
            return {}
        g_A = gradient_state.prev_step_grads
        if g_A is None:
            return {}

        # Flatten g_A and compute norm for normalization
        g_A_flat = flatten_grads(g_A)
        g_A_norm = g_A_flat.norm()
        if g_A_norm.item() < 1e-12:
            return {}

        # Normalized direction for numerical stability in finite-difference
        g_A_hat = {name: grad / g_A_norm for name, grad in g_A.items()}

        # Get learning rate at theta_{N-1} from the saved optimizer state
        eta = self._previous_checkpoint['optimizer']['param_groups'][0]['lr']

        # Save current training state (theta'') for restoration
        token = model_ctx.save_checkpoint(full=True)

        try:
            # Restore to theta_{N-1} (previous pre-step state)
            self._restore_full_state(self._previous_checkpoint, model_ctx)

            # Baseline gradient: g_B(theta_{N-1})
            baseline_grads = model_ctx.compute_batch_gradients()

            # Perturb: theta_{N-1} + epsilon * g_A_hat
            model_ctx.apply_perturbation(g_A_hat, scale=self._epsilon)

            # Perturbed gradient: g_B(theta_{N-1} + eps * g_A_hat)
            perturbed_grads = model_ctx.compute_batch_gradients()

            # Hv approximation: (perturbed - baseline) / epsilon * ||g_A||
            # This gives H_B(theta_{N-1}) * g_A (the actual entanglement direction)
            hv = {}
            for name in g_A:
                if name in perturbed_grads and name in baseline_grads:
                    hv[name] = (
                        (perturbed_grads[name] - baseline_grads[name])
                        / self._epsilon * g_A_norm
                    )
        finally:
            # Always restore to theta'' (current training state)
            model_ctx.restore_checkpoint(token)
            model_ctx.discard_checkpoint(token)

        # --- Compute metrics (model is now restored to theta'') ---
        metrics = self._compute_metrics(
            hv, baseline_grads, g_A_flat, g_A_norm, eta, ctx, model_ctx,
        )

        # Store entanglement vector for coherence tracking
        entanglement_flat = flatten_grads(hv)
        self._prev_entanglement_flat = entanglement_flat.detach().cpu()

        return metrics

    def _compute_metrics(
        self,
        hv: dict[str, torch.Tensor],
        baseline_grads: dict[str, torch.Tensor],
        g_A_flat: torch.Tensor,
        g_A_norm: torch.Tensor,
        eta: float,
        ctx,
        model_ctx,
    ) -> dict[str, float]:
        """Compute all 11 metrics from the Hv product and context."""
        hv_flat = flatten_grads(hv)
        g_B_flat = flatten_grads(baseline_grads)

        # Entanglement term: eta * H_B * g_A
        entanglement_flat = eta * hv_flat
        entanglement_norm = entanglement_flat.norm().item()

        # Content term: g_B + eta * H_B * g_A  (the ordering-invariant part)
        content_flat = g_B_flat + entanglement_flat
        content_norm = content_flat.norm().item()

        # Observed gradient
        g_B_norm = g_B_flat.norm().item()

        metrics: dict[str, float] = {}

        # Per-parameter decomposition
        for name in hv:
            if 'bias' in name or name not in baseline_grads:
                continue
            p_hv = hv[name].reshape(-1)
            p_gB = baseline_grads[name].reshape(-1)
            p_ent = eta * p_hv
            p_content = p_gB + p_ent
            p_ent_norm = p_ent.norm().item()
            p_gB_norm = p_gB.norm().item()
            metrics[f'entanglement_norm/{name}'] = p_ent_norm
            metrics[f'content_norm/{name}'] = p_content.norm().item()
            metrics[f'entanglement_energy_ratio/{name}'] = (
                p_ent_norm ** 2 / (p_gB_norm ** 2 + 1e-12)
            )

        # 1. Entanglement norm
        metrics['entanglement_norm'] = entanglement_norm

        # 2. Content norm
        metrics['content_norm'] = content_norm

        # 3. Observed gradient norm
        metrics['observed_grad_norm'] = g_B_norm

        # 4. Entanglement fraction
        metrics['entanglement_energy_ratio'] = (
            entanglement_norm ** 2 / (g_B_norm ** 2 + 1e-12)
        )

        # 5. Entanglement-content cosine similarity
        metrics['entanglement_content_cossim'] = (
            torch.dot(entanglement_flat, content_flat).item()
            / (entanglement_norm * content_norm + 1e-12)
        )

        # 6. Rayleigh quotient: g_A^T H_B g_A / ||g_A||^2
        metrics['rayleigh_quotient'] = (
            torch.dot(g_A_flat, hv_flat).item()
            / (g_A_norm.item() ** 2 + 1e-12)
        )

        # 7. Amplification ratio: ||H_B g_A|| / ||g_A||
        hv_norm = hv_flat.norm().item()
        amplification = hv_norm / (g_A_norm.item() + 1e-12)
        metrics['amplification_ratio'] = amplification

        # 8. Edge of stability: amplification * 2 * eta
        metrics['edge_of_stability'] = amplification * 2 * eta

        # 9-10. Solution alignment (requires reference weights)
        if hasattr(self, '_ref') and self._ref is not None:
            self._ref.ensure_loaded(model_ctx.device)
            ref_params = self._ref.weights
            if ref_params is not None:
                # Direction to solution: theta_ref - theta
                direction_parts = []
                for name, param in sorted(model_ctx.model.named_parameters()):
                    if 'bias' in name:
                        continue
                    ref_key = self._ref.resolve_key(name)
                    if ref_key is not None:
                        ref_dir = (ref_params[ref_key].to(param.device) - param.data).view(-1).float()
                        direction_parts.append(ref_dir)

                        # Per-parameter solution alignment
                        ref_dir_norm = ref_dir.norm().item()
                        if ref_dir_norm > 1e-12 and name in hv and name in baseline_grads:
                            p_ent = -(eta * hv[name].reshape(-1))
                            p_content = -(baseline_grads[name].reshape(-1) + eta * hv[name].reshape(-1))
                            p_ent_norm = p_ent.norm().item()
                            p_content_norm = p_content.norm().item()
                            if p_ent_norm > 1e-12:
                                metrics[f'entanglement_cossim_to_solution/{name}'] = (
                                    torch.dot(p_ent, ref_dir).item()
                                    / (p_ent_norm * ref_dir_norm)
                                )
                            if p_content_norm > 1e-12:
                                metrics[f'content_cossim_to_solution/{name}'] = (
                                    torch.dot(p_content, ref_dir).item()
                                    / (p_content_norm * ref_dir_norm)
                                )

                if direction_parts:
                    direction_flat = torch.cat(direction_parts)
                    direction_norm = direction_flat.norm()

                    # 9. Entanglement to solution: cos(-ent, theta_ref - theta)
                    neg_ent = -entanglement_flat
                    metrics['entanglement_cossim_to_solution'] = (
                        torch.dot(neg_ent, direction_flat).item()
                        / (entanglement_norm * direction_norm.item() + 1e-12)
                    )

                    # 10. Content to solution: cos(-content, theta_ref - theta)
                    neg_content = -content_flat
                    metrics['content_cossim_to_solution'] = (
                        torch.dot(neg_content, direction_flat).item()
                        / (content_norm * direction_norm.item() + 1e-12)
                    )

        # 11. Coherence: cos(ent_t, ent_{t-1}) — only when previous step also fired
        if (
            self._prev_entanglement_flat is not None
            and ctx.step is not None
            and self.step_schedule is not None
            and self.step_schedule.is_active(ctx.step - 1)
        ):
            prev = self._prev_entanglement_flat.to(entanglement_flat.device)
            prev_norm = prev.norm().item()
            metrics['entanglement_coherence'] = (
                torch.dot(entanglement_flat, prev).item()
                / (entanglement_norm * prev_norm + 1e-12)
            )

        return metrics

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        if self._prev_entanglement_flat is not None:
            tensors['prev_entanglement'] = self._prev_entanglement_flat
        return tensors

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        if 'prev_entanglement' in tensors:
            self._prev_entanglement_flat = tensors['prev_entanglement']

    def reset(self):
        self._current_checkpoint = None
        self._previous_checkpoint = None
        self._prev_entanglement_flat = None
