"""Token-level gradient distribution observer hook."""

import math

import torch
import numpy as np

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class TokenGradientHook(TrainingHook):
    """Analyze gradient distribution across embedding tokens.

    Ported from analysis_tools/token_gradient.py — same computation, live data.
    """

    name = "token_gradient"
    description = "Per-token gradient distribution (sparsity, Gini, stride groups, concentration)"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs_grads = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('gradient_sparsity', 'Fraction of token rows below 10% of mean norm',
                       'count(||g_row|| < 0.1 * mean) / p', 'higher = sparser',
                       label='Grad Sparsity'),
            MetricInfo('gradient_gini', 'Gini coefficient of token gradient norms',
                       formula='Gini index of row norms', sign_info='0 = equal, 1 = maximally unequal',
                       label='Grad Gini'),
            MetricInfo('stride_group_variance', 'Variance of mean gradient norm across stride groups',
                       'var(mean(||g_row||) per stride group)', 'higher = stride-dependent learning',
                       label='Stride Group Var'),
            MetricInfo('stride_group_max_ratio', 'Max stride group norm / mean stride group norm',
                       'max(group_means) / mean(group_means)',
                       label='Stride Group Ratio'),
            MetricInfo('tokens_for_50pct', 'Number of tokens capturing 50% of total gradient norm',
                       label='Tokens for 50%'),
            MetricInfo('tokens_for_90pct', 'Number of tokens capturing 90% of total gradient norm',
                       label='Tokens for 90%'),
            MetricInfo('concentration_ratio', 'Fraction of tokens capturing 50% of gradient norm',
                       'tokens_for_50pct / p', 'lower = more concentrated',
                       label='Concentration Ratio'),
        ]

    def compute(self, ctx) -> dict[str, float]:
        if ctx.accumulated_grads is None:
            return {}

        # Find embedding gradient
        emb_grad = None
        for name, grad in ctx.accumulated_grads.items():
            if 'embedding' in name.lower() and 'weight' in name.lower():
                emb_grad = grad
                break

        if emb_grad is None:
            return {}

        row_norms = emb_grad.norm(dim=1).float().cpu()
        actual_p = row_norms.shape[0]

        # 1. Overall sparsity
        threshold = row_norms.mean() * 0.1
        sparsity = (row_norms < threshold).float().mean().item()

        # 2. Gini coefficient
        sorted_norms, _ = torch.sort(row_norms)
        n = len(sorted_norms)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=row_norms.device)
        gini = (2 * index.dot(sorted_norms) - (n + 1) * sorted_norms.sum()) / (n * sorted_norms.sum() + 1e-8)

        # 3. Stride-group analysis
        stride = int(math.sqrt(actual_p))
        stride_group_norms = []
        for group in range(stride):
            indices = torch.arange(group, actual_p, stride)
            if len(indices) > 0:
                stride_group_norms.append(row_norms[indices].mean().item())

        stride_group_norms = np.array(stride_group_norms)
        stride_group_variance = np.var(stride_group_norms) if len(stride_group_norms) > 0 else 0
        stride_group_max_ratio = (
            stride_group_norms.max() / (stride_group_norms.mean() + 1e-8)
            if len(stride_group_norms) > 0 else 0
        )

        # 4. Concentration metrics
        sorted_norms_desc, _ = torch.sort(row_norms, descending=True)
        cumsum_norm = torch.cumsum(sorted_norms_desc, dim=0) / (sorted_norms_desc.sum() + 1e-8)
        tokens_50pct = (cumsum_norm < 0.5).sum().item() + 1
        tokens_90pct = (cumsum_norm < 0.9).sum().item() + 1

        return {
            'gradient_sparsity': sparsity,
            'gradient_gini': gini.item(),
            'stride_group_variance': stride_group_variance,
            'stride_group_max_ratio': stride_group_max_ratio,
            'tokens_for_50pct': tokens_50pct,
            'tokens_for_90pct': tokens_90pct,
            'concentration_ratio': tokens_50pct / actual_p,
        }
