"""Batch-level gradient dynamics hook.

Measures temporal structure of gradient signals across training steps:
- Autocorrelation: cosine similarity decay across multiple lag offsets
- Accumulation efficiency: constructive vs destructive interference over windows
- Subspace dimensionality: effective rank of recent gradient history

All metrics share a single GPU ring buffer of flattened batch gradients,
so the VRAM cost is max_lag * n_params * 4 bytes regardless of which
metric families are active.
"""

import math
from collections import deque

import torch

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookRequirements, TrainingParadigm, GradientAvailability
from framework.utils import flatten_grads


@HookRegistry.register
class BatchDynamicsHook(TrainingHook):
    """Batch-level gradient dynamics analysis.

    At each training step, captures the batch gradient from the model's
    .grad attributes and stores it in a GPU ring buffer. Computes
    per-step metrics (autocorrelation) and periodic emission metrics
    (accumulation efficiency, subspace dimensionality).

    Memory cost: max_lag * n_params * 4 bytes of VRAM.
    """

    name = "batch_dynamics"
    description = "Batch-level gradient dynamics (autocorrelation, accumulation efficiency, subspace rank)"
    hook_points = {HookPoint.POST_STEP, HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    loop_points = {
        'epoch': {HookPoint.POST_STEP, HookPoint.POST_EPOCH},
        'step': {HookPoint.POST_STEP, HookPoint.SNAPSHOT},
    }
    requires = HookRequirements(
        paradigm=TrainingParadigm.BACKPROP,
        gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
    )

    def __init__(self, max_lag: int = 50, lags: list[int] | None = None,
                 windows: list[int] | None = None):
        """
        Args:
            max_lag: Ring buffer size. Determines the maximum lag/window
                that can be computed.
            lags: Lag offsets for autocorrelation. Defaults to
                [1, 2, 5, 10, 20, 50]. Values exceeding max_lag are
                filtered out.
            windows: Window sizes for accumulation efficiency. Defaults
                to [2, 5, 10, 20, 50]. Values exceeding max_lag are
                filtered out.
        """
        self._max_lag = max_lag
        default_lags = [1, 2, 5, 10, 20, 50]
        default_windows = [2, 5, 10, 20, 50]
        self._lags = sorted(l for l in (lags or default_lags) if l <= max_lag)
        self._windows = sorted(w for w in (windows or default_windows) if w <= max_lag)
        self._buffer: deque[tuple[torch.Tensor, float]] = deque(maxlen=max_lag)
        self._accumulators: dict[int, list[float]] = {l: [] for l in self._lags}

    def describe_metrics(self) -> list[MetricInfo]:
        metrics = []

        # Autocorrelation
        for lag in self._lags:
            metrics.append(MetricInfo(
                f'lag_{lag}',
                f'Mean cosine similarity between gradients {lag} steps apart',
                f'mean(cos(g_t, g_{{t-{lag}}}))',
                sign_info='+1 = correlated, 0 = uncorrelated, -1 = anti-correlated',
                label=f'Lag-{lag} Cos Sim',
            ))
        metrics.append(MetricInfo(
            'autocorrelation_mean',
            'Mean autocorrelation across all measured lags',
            'mean(lag_k for all k)',
            label='Mean Autocorrelation',
        ))

        # Accumulation efficiency
        for w in self._windows:
            metrics.append(MetricInfo(
                f'efficiency_{w}',
                f'Gradient accumulation efficiency over {w}-step window',
                f'||sum(g_{{t-{w}+1}}..g_t)|| / sum(||g_i||)',
                sign_info='1.0 = perfect constructive, 1/sqrt(K) = random walk',
                label=f'Efficiency ({w}-step)',
            ))

        # Subspace dimensionality
        metrics.append(MetricInfo(
            'effective_rank',
            'Effective rank of recent gradient history (exp of SV entropy)',
            'exp(-sum(p_i * log(p_i))) where p_i = sigma_i^2 / sum(sigma^2)',
            sign_info='low = structured/ordered, high = random/diverse',
            label='Effective Rank (Grads)',
        ))
        metrics.append(MetricInfo(
            'top1_variance',
            'Fraction of gradient variance explained by top singular vector',
            'sigma_1^2 / sum(sigma_i^2)',
            sign_info='high = gradients concentrated in one direction',
            label='Top-1 Variance (Grads)',
        ))

        return metrics

    def compute(self, ctx, **state) -> dict[str, float]:
        if ctx.hook_point == HookPoint.POST_STEP:
            return self._accumulate(ctx, **state)

        # Emission point (POST_EPOCH or SNAPSHOT)
        return self._emit()

    def _accumulate(self, ctx, **state) -> dict[str, float]:
        """Capture gradient and compute per-step autocorrelations."""
        model_state = state.get('model_state')
        if model_state is None:
            return {}
        model = model_state.model

        # Extract current batch gradient from model
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach()

        if not grads:
            return {}

        flat = flatten_grads(grads)
        norm = flat.norm().item()

        if norm < 1e-12:
            self._buffer.append((flat.detach(), norm))
            return {}

        # Batch-compute cosine similarities across all active lags
        # (single matmul + single .tolist() transfer instead of per-lag .item() syncs)
        buf_len = len(self._buffer)
        active_lags = []
        active_prev = []
        active_norms = []
        for lag in self._lags:
            if lag > buf_len:
                continue
            prev_flat, prev_norm = self._buffer[-lag]
            if prev_norm < 1e-12:
                continue
            active_lags.append(lag)
            active_prev.append(prev_flat)
            active_norms.append(prev_norm)

        if active_lags:
            dots = (torch.stack(active_prev) @ flat).tolist()
            for i, lag in enumerate(active_lags):
                self._accumulators[lag].append(dots[i] / (norm * active_norms[i]))

        # Store current gradient (after computing similarities)
        self._buffer.append((flat.detach(), norm))
        return {}

    def _emit(self) -> dict[str, float]:
        """Emit all metrics, then reset per-period accumulators."""
        metrics = {}

        # --- Autocorrelation ---
        lag_values = []
        for lag in self._lags:
            acc = self._accumulators[lag]
            if acc:
                mean_val = sum(acc) / len(acc)
                metrics[f'lag_{lag}'] = mean_val
                lag_values.append(mean_val)

        if lag_values:
            metrics['autocorrelation_mean'] = sum(lag_values) / len(lag_values)

        # --- Accumulation efficiency (from buffer, single cumulative pass) ---
        buf_list = list(self._buffer)
        buf_len = len(buf_list)
        window_set = set(self._windows)
        max_window = max(self._windows) if self._windows else 0
        if buf_len >= 2 and max_window >= 2:
            grad_sum = buf_list[-1][0].clone()
            norm_sum = buf_list[-1][1]
            for w_idx in range(2, min(max_window, buf_len) + 1):
                entry_flat, entry_norm = buf_list[-w_idx]
                grad_sum.add_(entry_flat)
                norm_sum += entry_norm
                if w_idx in window_set and norm_sum > 1e-12:
                    accumulated_norm = grad_sum.norm().item()
                    metrics[f'efficiency_{w_idx}'] = accumulated_norm / norm_sum

        # --- Subspace dimensionality (from buffer, SVD) ---
        if buf_len >= 2:
            subspace_metrics = self._compute_subspace_metrics(buf_list)
            metrics.update(subspace_metrics)

        # Reset per-period accumulators
        self._accumulators = {l: [] for l in self._lags}
        return metrics

    def _compute_subspace_metrics(self, buf_list) -> dict[str, float]:
        """Compute effective rank and top-1 variance from gradient history.

        Uses the Gram matrix approach: for N buffered gradients of dimension D
        where N << D, computing the NxN Gram matrix and its eigenvalues is
        far cheaper than SVD on the full NxD matrix.
        """
        metrics = {}

        grads = [flat for flat, norm in buf_list if norm >= 1e-12]
        if len(grads) < 2:
            return metrics

        mat = torch.stack(grads)  # (N, D) — stays on GPU

        try:
            # Gram matrix: (N, D) @ (D, N) = (N, N) — tiny compared to full SVD
            gram = mat @ mat.T
            # Eigenvalues of the Gram matrix = squared singular values
            eigenvalues = torch.linalg.eigvalsh(gram)
            # eigvalsh returns ascending order; clamp negatives from numerical noise
            sv_sq = eigenvalues.clamp(min=0).flip(0)
        except Exception:
            return metrics

        total = sv_sq.sum().item()
        if total < 1e-16:
            return metrics

        # Top-1 variance fraction
        metrics['top1_variance'] = sv_sq[0].item() / total

        # Effective rank: exp(entropy of normalized singular value squares)
        p = sv_sq / sv_sq.sum()
        p = p[p > 0]
        entropy = -(p * p.log()).sum().item()
        metrics['effective_rank'] = math.exp(entropy)

        return metrics

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        for i, (flat, norm) in enumerate(self._buffer):
            tensors[f'buffer_{i}'] = flat
            tensors[f'norm_{i}'] = torch.tensor(norm)
        return tensors

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        self._buffer.clear()
        i = 0
        while f'buffer_{i}' in tensors:
            flat = tensors[f'buffer_{i}']
            norm = tensors[f'norm_{i}'].item()
            self._buffer.append((flat, norm))
            i += 1

    def reset(self):
        self._buffer.clear()
        self._accumulators = {l: [] for l in self._lags}
