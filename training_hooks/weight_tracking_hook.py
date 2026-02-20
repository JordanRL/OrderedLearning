"""Weight-space tracking observer hook."""

import torch

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class WeightTrackingHook(TrainingHook):
    """Track weight norms, spectral properties, and gradient-weight alignment.

    For each weight matrix (excluding biases), computes:
    - Weight L2 norm
    - Top singular value (spectral norm)
    - Effective rank (sum(sigma)^2 / sum(sigma^2))
    - Cosine similarity between gradient and weight (gradient-weight alignment)

    Also computes whole-model summary statistics.
    """

    name = "weight_tracking"
    description = "Weight norms, spectral properties, effective rank, and gradient-weight alignment per layer"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs_grads = True

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('weight_norm/{layer}', 'Per-layer weight L2 norm', '||W||_2',
                       label='Layer Weight Norm'),
            MetricInfo('top_sv/{layer}', 'Per-layer top singular value (spectral norm)', 'sigma_1(W)',
                       label='Layer Top SV'),
            MetricInfo('effective_rank/{layer}', 'Per-layer effective rank of weight matrix',
                       '(sum(sigma))^2 / sum(sigma^2)',
                       'lower during grokking = algorithmic low-rank structure',
                       label='Layer Effective Rank'),
            MetricInfo('grad_weight_align/{layer}', 'Per-layer cosine similarity between gradient and weight',
                       'cos(g, W)', '+ = gradient reinforces weight direction',
                       label='Layer Grad-Weight Align'),
            MetricInfo('total_weight_norm', 'Whole-model weight L2 norm', 'sqrt(sum(||W_i||^2))',
                       label='Total Weight Norm'),
            MetricInfo('mean_weight_norm', 'Mean per-layer weight norm', 'mean(||W_i||)',
                       label='Mean Weight Norm'),
            MetricInfo('mean_top_sv', 'Mean top singular value across layers', 'mean(sigma_1(W_i))',
                       label='Mean Top SV'),
            MetricInfo('max_top_sv', 'Maximum top singular value across layers', 'max(sigma_1(W_i))',
                       label='Max Top SV'),
            MetricInfo('mean_effective_rank', 'Mean effective rank across layers',
                       'mean((sum(sigma))^2 / sum(sigma^2))',
                       label='Mean Effective Rank'),
            MetricInfo('mean_grad_weight_align', 'Mean gradient-weight alignment across layers',
                       'mean(cos(g_i, W_i))',
                       label='Mean Grad-Weight Align'),
        ]

    def compute(self, ctx) -> dict[str, float]:
        model = ctx.model
        grads = ctx.accumulated_grads

        metrics = {}
        all_weight_norms = []
        all_top_svs = []
        all_effective_ranks = []
        all_alignments = []

        for name, param in model.named_parameters():
            if param.dim() < 2 or 'bias' in name:
                continue

            w = param.data.float()
            w_norm = w.norm().item()
            metrics[f'weight_norm/{name}'] = w_norm
            all_weight_norms.append(w_norm)

            # Singular value decomposition
            # For small matrices, use full SVD; for large ones, use lowrank
            try:
                min_dim = min(w.shape)
                if min_dim <= 64:
                    S = torch.linalg.svdvals(w)
                else:
                    _, S, _ = torch.svd_lowrank(w, q=min(20, min_dim))

                top_sv = S[0].item()
                metrics[f'top_sv/{name}'] = top_sv
                all_top_svs.append(top_sv)

                # Effective rank: (sum(sigma))^2 / sum(sigma^2)
                sv_sum = S.sum().item()
                sv_sq_sum = (S ** 2).sum().item()
                if sv_sq_sum > 1e-16:
                    eff_rank = sv_sum ** 2 / sv_sq_sum
                    metrics[f'effective_rank/{name}'] = eff_rank
                    all_effective_ranks.append(eff_rank)
            except Exception:
                pass

            # Gradient-weight cosine similarity
            if grads is not None and name in grads:
                g = grads[name].float()
                g_flat = g.reshape(-1)
                w_flat = w.reshape(-1)
                cos_sim = torch.dot(g_flat, w_flat) / (
                    g_flat.norm() * w_flat.norm() + 1e-8
                )
                alignment = cos_sim.item()
                metrics[f'grad_weight_align/{name}'] = alignment
                all_alignments.append(alignment)

        # Summary statistics
        if all_weight_norms:
            metrics['total_weight_norm'] = sum(n ** 2 for n in all_weight_norms) ** 0.5
            metrics['mean_weight_norm'] = sum(all_weight_norms) / len(all_weight_norms)
        if all_top_svs:
            metrics['mean_top_sv'] = sum(all_top_svs) / len(all_top_svs)
            metrics['max_top_sv'] = max(all_top_svs)
        if all_effective_ranks:
            metrics['mean_effective_rank'] = sum(all_effective_ranks) / len(all_effective_ranks)
        if all_alignments:
            metrics['mean_grad_weight_align'] = sum(all_alignments) / len(all_alignments)

        return metrics
