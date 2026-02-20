"""Standard training diagnostics hook.

Collects batch-level statistics that are routinely monitored in
production training runs: loss volatility, gradient noise proxies,
update-to-weight ratios, and loss predictability. All metrics are
derived from scalars (loss, gradient norm, weight norm) — no gradient
tensors are stored.

These represent what a standard auditing or monitoring pipeline would
see, independent of any ordering-aware analysis.
"""

import math

import torch

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class TrainingDiagnosticsHook(TrainingHook):
    """Standard training diagnostics from per-step scalar statistics.

    Accumulates per-step loss, gradient norm, and weight norm values,
    then emits summary statistics at each emission point. All
    computation is scalar-only — no gradient tensor storage required.
    """

    name = "training_diagnostics"
    description = "Standard training diagnostics (loss volatility, gradient noise, update ratios)"
    hook_points = {HookPoint.POST_STEP, HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    loop_points = {
        'epoch': {HookPoint.POST_STEP, HookPoint.POST_EPOCH},
        'step': {HookPoint.POST_STEP, HookPoint.SNAPSHOT},
    }
    needs_grads = False

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            # Loss statistics
            MetricInfo('loss_mean', 'Mean loss over emission period',
                       label='Mean Loss'),
            MetricInfo('loss_std', 'Standard deviation of per-step loss',
                       sign_info='lower = smoother training',
                       label='Loss Std'),
            MetricInfo('loss_volatility', 'Coefficient of variation of loss (std/mean)',
                       sign_info='lower = more stable training',
                       label='Loss Volatility'),
            MetricInfo('loss_autocorrelation',
                       'Lag-1 autocorrelation of per-step loss sequence',
                       'corr(L_t, L_{t+1})',
                       sign_info='high = predictable loss sequence',
                       label='Loss Autocorrelation'),

            # Gradient norm statistics
            MetricInfo('grad_norm_mean', 'Mean gradient norm over emission period',
                       label='Mean Grad Norm'),
            MetricInfo('grad_norm_std', 'Standard deviation of gradient norms',
                       label='Grad Norm Std'),
            MetricInfo('grad_norm_cv', 'Coefficient of variation of gradient norms (std/mean)',
                       sign_info='proxy for gradient noise scale; lower = more coherent',
                       label='Grad Norm CV'),
            MetricInfo('grad_norm_max', 'Maximum gradient norm in emission period',
                       sign_info='spike detection',
                       label='Max Grad Norm'),

            # Update-to-weight ratio
            MetricInfo('update_ratio_mean',
                       'Mean update-to-weight ratio (lr * ||grad|| / ||params||)',
                       sign_info='standard learning rate diagnostic',
                       label='Mean Update Ratio'),
            MetricInfo('update_ratio_std',
                       'Std of update-to-weight ratio',
                       sign_info='lower = more consistent updates',
                       label='Update Ratio Std'),

            # Weight norm
            MetricInfo('weight_norm', 'Total parameter L2 norm at emission time',
                       label='Weight Norm'),
        ]

    def __init__(self):
        self._losses: list[float] = []
        self._grad_norms: list[float] = []
        self._update_ratios: list[float] = []
        self._last_weight_norm: float = 0.0

    def compute(self, ctx) -> dict[str, float]:
        if ctx.hook_point == HookPoint.POST_STEP:
            return self._accumulate(ctx)
        return self._emit()

    def _accumulate(self, ctx) -> dict[str, float]:
        """Collect per-step scalars."""
        # Loss
        if ctx.loss is not None:
            self._losses.append(ctx.loss)

        if ctx.model is None:
            return {}

        # Compute norms on GPU — single .item() sync each instead of per-parameter
        weight_norm = sum(
            p.data.norm().square() for p in ctx.model.parameters()
        ).sqrt().item()
        self._last_weight_norm = weight_norm

        grad_sq_terms = [
            p.grad.norm().square()
            for p in ctx.model.parameters() if p.grad is not None
        ]
        if grad_sq_terms:
            grad_norm = sum(grad_sq_terms).sqrt().item()
            self._grad_norms.append(grad_norm)

            # Update-to-weight ratio: lr * ||grad|| / ||params||
            lr = ctx.lr if ctx.lr is not None else 0.0
            if weight_norm > 1e-12:
                self._update_ratios.append(lr * grad_norm / weight_norm)

        return {}

    def _emit(self) -> dict[str, float]:
        """Compute and emit summary statistics, then reset."""
        metrics = {}

        # Materialize GPU tensor losses to CPU floats in one transfer
        # (avoids per-element CUDA sync)
        if self._losses and isinstance(self._losses[0], torch.Tensor):
            self._losses = torch.stack(self._losses).tolist()

        # --- Loss statistics ---
        if len(self._losses) >= 2:
            n = len(self._losses)
            mean = sum(self._losses) / n
            var = sum((l - mean) ** 2 for l in self._losses) / (n - 1)
            std = math.sqrt(var)

            metrics['loss_mean'] = mean
            metrics['loss_std'] = std
            if abs(mean) > 1e-12:
                metrics['loss_volatility'] = std / abs(mean)

            # Lag-1 autocorrelation
            if n >= 3 and var > 1e-16:
                diffs = [l - mean for l in self._losses]
                cov = sum(diffs[i] * diffs[i + 1] for i in range(n - 1))
                metrics['loss_autocorrelation'] = cov / (var * (n - 1))

        elif len(self._losses) == 1:
            metrics['loss_mean'] = self._losses[0]

        # --- Gradient norm statistics ---
        if len(self._grad_norms) >= 2:
            n = len(self._grad_norms)
            mean = sum(self._grad_norms) / n
            var = sum((g - mean) ** 2 for g in self._grad_norms) / (n - 1)
            std = math.sqrt(var)

            metrics['grad_norm_mean'] = mean
            metrics['grad_norm_std'] = std
            if mean > 1e-12:
                metrics['grad_norm_cv'] = std / mean
            metrics['grad_norm_max'] = max(self._grad_norms)

        elif len(self._grad_norms) == 1:
            metrics['grad_norm_mean'] = self._grad_norms[0]
            metrics['grad_norm_max'] = self._grad_norms[0]

        # --- Update-to-weight ratio ---
        if len(self._update_ratios) >= 2:
            n = len(self._update_ratios)
            mean = sum(self._update_ratios) / n
            var = sum((r - mean) ** 2 for r in self._update_ratios) / (n - 1)
            metrics['update_ratio_mean'] = mean
            metrics['update_ratio_std'] = math.sqrt(var)
        elif len(self._update_ratios) == 1:
            metrics['update_ratio_mean'] = self._update_ratios[0]

        # --- Weight norm (snapshot) ---
        if self._last_weight_norm > 0:
            metrics['weight_norm'] = self._last_weight_norm

        # Reset accumulators
        self._losses = []
        self._grad_norms = []
        self._update_ratios = []

        return metrics

    def reset(self):
        self._losses = []
        self._grad_norms = []
        self._update_ratios = []
        self._last_weight_norm = 0.0
