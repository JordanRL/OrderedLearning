"""Adam optimizer dynamics measurement hook.

Measures how Adam's internal state (momentum, adaptive learning rates)
interacts with the training signal under different data ordering strategies.

Three tiers of metrics with increasing data requirements:
- Tier 1 (direction-agnostic): Compares Adam update to raw gradient
- Tier 2 (solution-dependent): Alignment with direction to known solution
- Tier 3 (probe-dependent): Alignment with strategy's target gradient

The key experimental question: is the optimizer differentially amplifying
the curriculum/ordering signal toward the target, or is it neutral?
"""

import torch

from .base import MetricInfo, HookPoint, StepSchedule, InterventionHook, HookRegistry
from analysis_tools.utils import flatten_grads


def _cossim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors, handling edge cases."""
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return torch.dot(a, b).item() / (norm_a.item() * norm_b.item())


@HookRegistry.register
class AdamDynamicsHook(InterventionHook):
    """Measure Adam optimizer internal state dynamics.

    Requires InterventionHook for read-only access to optimizer state via
    ModelDataContext. Does not modify any training state.

    Fires at POST_STEP in both loop types with a burst schedule (like
    HessianHook). Default: bursts of 10 steps every 500 steps. Reads
    param.grad directly for the current batch gradient.

    Adam update is computed analytically from optimizer state:
        m_hat = exp_avg / (1 - beta1^step)
        v_hat = exp_avg_sq / (1 - beta2^step)
        update = lr * m_hat / (sqrt(v_hat) + eps)

    Weight decay (AdamW) is intentionally excluded to isolate adaptive
    learning rate dynamics from regularization.
    """

    name = "adam_dynamics"
    description = "Adam optimizer state dynamics (momentum, amplification, alignment)"
    hook_points = {HookPoint.POST_STEP}
    loop_points = {
        'step': {HookPoint.POST_STEP},
        'epoch': {HookPoint.POST_STEP},
    }
    needs_grads = False
    needs_prev_step_grads = False
    needs_reference_weights = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            # Tier 1 — Direction-agnostic
            MetricInfo(
                'momentum_grad_cossim',
                'Cosine similarity between Adam first moment and current gradient',
                'cos(exp_avg, grad)',
                '+ = optimizer memory agrees with current data',
                label='Mom-Grad Cos Sim (Adam)',
            ),
            MetricInfo(
                'amplification_ratio',
                'Ratio of Adam update norm to raw SGD update norm',
                '||adam_update|| / ||lr * grad||',
                '>1 = optimizer amplifies, <1 = dampens',
                label='Amplification Ratio (Adam)',
            ),
            MetricInfo(
                'update_deflection',
                'Fraction of Adam update orthogonal to current gradient (momentum + adaptive scaling)',
                '||update_orth|| / ||update||',
                '0 = follows gradient, 1 = fully redirected',
                label='Update Deflection (Adam)',
            ),
            MetricInfo(
                'effective_lr_cv',
                'Coefficient of variation of per-element effective learning rates',
                'std(lr / sqrt(v_hat + eps)) / mean(...)',
                'higher = more focused/non-uniform adaptation',
                label='Effective LR CV (Adam)',
            ),
            # Tier 2 — Solution-dependent
            MetricInfo(
                'momentum_solution_cossim',
                'Cosine similarity between first moment and direction to solution',
                'cos(exp_avg, theta_ref - theta)',
                '+ = optimizer has accumulated toward solution',
                label='Mom→Solution Cos',
            ),
            MetricInfo(
                'update_solution_cossim',
                'Cosine similarity between Adam update and direction to solution',
                'cos(adam_update, theta_ref - theta)',
                '+ = effective step aims at solution',
                label='Update→Solution Cos',
            ),
            MetricInfo(
                'grad_solution_cossim',
                'Cosine similarity between raw gradient and direction to solution',
                'cos(grad, theta_ref - theta)',
                '+ = raw data signal points at solution (baseline)',
                label='Grad→Solution Cos',
            ),
            MetricInfo(
                'optimizer_solution_amplification',
                'Difference in solution alignment: Adam update vs raw gradient',
                'update_solution_cossim - grad_solution_cossim',
                '+ = optimizer makes step MORE solution-aligned',
                label='Optimizer Sol Amp',
            ),
            # Tier 3 — Probe-dependent
            MetricInfo(
                'momentum_probe_cossim',
                'Cosine similarity between first moment and target/probe gradient',
                'cos(exp_avg, target_grad)',
                '+ = optimizer has accumulated toward probe target',
                label='Mom→Probe Cos',
            ),
            MetricInfo(
                'update_probe_cossim',
                'Cosine similarity between Adam update and target/probe gradient',
                'cos(adam_update, target_grad)',
                '+ = effective step aims at probe target',
                label='Update→Probe Cos',
            ),
            MetricInfo(
                'grad_probe_cossim',
                'Cosine similarity between raw gradient and target/probe gradient',
                'cos(grad, target_grad)',
                '+ = raw data signal points at probe target (baseline)',
                label='Grad→Probe Cos',
            ),
            MetricInfo(
                'optimizer_probe_amplification',
                'Difference in probe alignment: Adam update vs raw gradient',
                'update_probe_cossim - grad_probe_cossim',
                '+ = optimizer makes step MORE probe-aligned',
                label='Optimizer Probe Amp',
            ),
        ]

    def __init__(
        self,
        every_n_steps: int = 500,
        burst_length: int = 10,
    ):
        """
        Args:
            every_n_steps: Spacing between measurement bursts. First burst
                fires at step every_n_steps (not step 0).
            burst_length: Number of consecutive steps per burst.
        """
        self.step_schedule = StepSchedule(
            mode='burst',
            stride=every_n_steps,
            burst_length=burst_length,
            warmup=every_n_steps,
        )

    def intervene(self, run_ctx, model_ctx) -> dict[str, float]:
        optimizer = model_ctx._optimizer
        model = model_ctx.model

        # --- Extract optimizer hyperparams ---
        pg = optimizer.param_groups[0]
        lr = pg['lr']
        beta1, beta2 = pg.get('betas', (0.9, 0.999))
        eps = pg.get('eps', 1e-8)

        # --- Collect per-parameter data ---
        current_grads = self._get_current_gradient(run_ctx, model_ctx)
        if current_grads is None:
            return {}

        momentum_parts = []
        adam_update_parts = []
        grad_parts = []
        effective_lr_parts = []
        has_optimizer_state = False

        for name, param in sorted(model.named_parameters()):
            if 'bias' in name:
                continue
            if name not in current_grads:
                continue
            if param not in optimizer.state:
                continue

            state = optimizer.state[param]
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            step_count = state.get('step')
            if exp_avg is None or exp_avg_sq is None or step_count is None:
                continue
            # step_count can be a tensor in newer PyTorch
            if hasattr(step_count, 'item'):
                step_count = step_count.item()
            if step_count < 1:
                continue

            has_optimizer_state = True
            grad = current_grads[name]

            # Bias-corrected estimates
            bc1 = 1 - beta1 ** step_count
            bc2 = 1 - beta2 ** step_count
            m_hat = exp_avg / bc1
            v_hat = exp_avg_sq / bc2

            # Adam update (excluding weight decay)
            adam_upd = lr * m_hat / (torch.sqrt(v_hat) + eps)

            # Per-element effective learning rate
            eff_lr = lr / (torch.sqrt(v_hat) + eps)

            momentum_parts.append(exp_avg.detach().view(-1).float())
            adam_update_parts.append(adam_upd.detach().view(-1).float())
            grad_parts.append(grad.detach().view(-1).float())
            effective_lr_parts.append(eff_lr.detach().view(-1).float())

        if not has_optimizer_state:
            return {}

        # --- Flatten to single vectors ---
        momentum_flat = torch.cat(momentum_parts)
        adam_update_flat = torch.cat(adam_update_parts)
        grad_flat = torch.cat(grad_parts)
        effective_lr_flat = torch.cat(effective_lr_parts)

        metrics: dict[str, float] = {}

        # === Tier 1: Direction-agnostic ===
        self._compute_tier1(
            metrics, momentum_flat, adam_update_flat, grad_flat,
            effective_lr_flat, lr,
        )

        # === Tier 2: Solution-dependent ===
        solution_dir = self._get_solution_direction(model_ctx)
        if solution_dir is not None:
            self._compute_directional(
                metrics, momentum_flat, adam_update_flat, grad_flat,
                solution_dir, prefix='solution',
            )

        # === Tier 3: Probe-dependent ===
        target_grad_flat = self._get_target_grad(run_ctx)
        if target_grad_flat is not None:
            self._compute_directional(
                metrics, momentum_flat, adam_update_flat, grad_flat,
                target_grad_flat, prefix='probe',
            )

        return metrics

    def _compute_tier1(
        self,
        metrics: dict[str, float],
        momentum_flat: torch.Tensor,
        adam_update_flat: torch.Tensor,
        grad_flat: torch.Tensor,
        effective_lr_flat: torch.Tensor,
        lr: float,
    ):
        """Compute direction-agnostic metrics."""
        # 1. Momentum-gradient cosine similarity
        metrics['momentum_grad_cossim'] = _cossim(momentum_flat, grad_flat)

        # 2. Amplification ratio: ||adam_update|| / ||lr * grad||
        sgd_norm = lr * grad_flat.norm().item()
        adam_norm = adam_update_flat.norm().item()
        metrics['amplification_ratio'] = (
            adam_norm / sgd_norm if sgd_norm > 1e-12 else 0.0
        )

        # 3. Momentum contribution: fraction of update orthogonal to gradient
        grad_norm_sq = grad_flat.dot(grad_flat)
        if grad_norm_sq > 1e-12 and adam_norm > 1e-12:
            proj_coeff = adam_update_flat.dot(grad_flat) / grad_norm_sq
            proj = proj_coeff * grad_flat
            orthogonal = adam_update_flat - proj
            metrics['update_deflection'] = (
                orthogonal.norm().item() / adam_norm
            )
        else:
            metrics['update_deflection'] = 0.0

        # 4. Effective learning rate coefficient of variation
        mean_lr = effective_lr_flat.mean()
        if mean_lr > 1e-12:
            metrics['effective_lr_cv'] = (
                effective_lr_flat.std().item() / mean_lr.item()
            )
        else:
            metrics['effective_lr_cv'] = 0.0

    def _compute_directional(
        self,
        metrics: dict[str, float],
        momentum_flat: torch.Tensor,
        adam_update_flat: torch.Tensor,
        grad_flat: torch.Tensor,
        direction: torch.Tensor,
        prefix: str,
    ):
        """Compute directional metrics (Tier 2 or Tier 3)."""
        metrics[f'momentum_{prefix}_cossim'] = _cossim(momentum_flat, direction)
        update_cos = _cossim(adam_update_flat, direction)
        grad_cos = _cossim(grad_flat, direction)
        metrics[f'update_{prefix}_cossim'] = update_cos
        metrics[f'grad_{prefix}_cossim'] = grad_cos
        metrics[f'optimizer_{prefix}_amplification'] = update_cos - grad_cos

    def _get_current_gradient(self, run_ctx, model_ctx):
        """Get current training gradient from appropriate source."""
        # Prefer accumulated_grads (epoch loop — more representative)
        if run_ctx.accumulated_grads is not None:
            return run_ctx.accumulated_grads

        # Fall back to param.grad (step loop — single batch)
        grads = {}
        for name, param in model_ctx.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
        return grads if grads else None

    def _get_solution_direction(self, model_ctx):
        """Compute flattened solution direction (theta_ref - theta_current)."""
        if self._ref is None:
            return None
        self._ref.ensure_loaded(model_ctx.device)
        if not self._ref.available:
            return None

        ref_weights = self._ref.weights
        parts = []
        for name, param in sorted(model_ctx.model.named_parameters()):
            if 'bias' in name:
                continue
            ref_key = self._ref.resolve_key(name)
            if ref_key is not None:
                diff = (ref_weights[ref_key].to(param.device) - param.data).view(-1).float()
                parts.append(diff)
        return torch.cat(parts) if parts else None

    def _get_target_grad(self, run_ctx):
        """Get flattened target/probe gradient if available."""
        if not hasattr(run_ctx, 'target_grad') or run_ctx.target_grad is None:
            return None
        return flatten_grads(run_ctx.target_grad)

    def reset(self):
        pass
