"""Attention structure observer hook."""

import torch
import numpy as np

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry
from framework.capabilities import HookRequirements, ModelCapability


@HookRegistry.register
class AttentionHook(TrainingHook):
    """Analyze attention weight structure and attention patterns on data.

    Two analysis modes:

    1. **Weight SVD** (always): SVD of Q/K/V projection matrices to measure
       rank concentration and effective dimensionality. Does not require data.

    2. **Attention patterns on data** (when loader available): Runs a forward
       pass on a batch, capturing actual attention weights from each layer.
       Measures entropy (peaked = algorithmic, uniform = memorization) and
       cross-input consistency (low variance = input-independent algorithm,
       high variance = input-dependent memorization).
    """

    name = "attention"
    description = "Attention weight structure (SVD) and attention patterns on data"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {
        'epoch': {HookPoint.POST_EPOCH},
        'step': {HookPoint.SNAPSHOT},
    }
    requires = HookRequirements(model_capabilities=ModelCapability.ATTENTION)

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            # Weight SVD metrics
            MetricInfo('sv_concentration', 'Mean top-1 singular value fraction across Q/K/V',
                       'mean(sigma_1 / sum(sigma_i))', 'higher = more rank-1 like',
                       label='SV Concentration'),
            MetricInfo('effective_rank', 'Mean effective rank of Q/K/V weight matrices',
                       'mean(sum(sigma)^2 / sum(sigma^2))', 'higher = more distributed',
                       label='Effective Rank'),
            MetricInfo('top5_explained_var', 'Mean variance explained by top 5 singular values',
                       'mean(sum(sigma_1..5^2) / sum(sigma^2))',
                       label='Top-5 Var Explained'),

            # Attention pattern metrics (data-dependent, epoch loop only)
            MetricInfo('attn_entropy', 'Mean normalized attention entropy across heads and layers',
                       'mean(H(attn) / log(seq_len))',
                       '0 = fully peaked (algorithmic), 1 = uniform (no structure)',
                       label='Attn Entropy'),
            MetricInfo('attn_entropy/{layer}', 'Per-layer mean attention entropy',
                       label='Layer Attn Entropy'),
            MetricInfo('attn_variance', 'Mean cross-input variance of attention patterns',
                       'mean(Var_batch(attn))',
                       'low = input-independent (algorithmic), high = input-dependent (memorization)',
                       label='Attn Variance'),
            MetricInfo('attn_variance/{layer}', 'Per-layer cross-input attention variance',
                       label='Layer Attn Variance'),
        ]

    def compute(self, ctx, **state) -> dict[str, float]:
        model_state = state.get('model_state')
        if model_state is None:
            return {}
        model = model_state.model

        num_layers = getattr(ctx.config, 'layers', 2) if ctx.config else 2

        metrics = {}

        # === Weight SVD analysis (always) ===
        metrics.update(self._weight_svd(model, num_layers))

        # === Attention patterns on data (when loader available) ===
        batch_state = state.get('batch_state')
        loader = batch_state.loader if batch_state is not None else None
        if loader is not None:
            pattern_metrics = self._attention_patterns(model, loader, num_layers)
            metrics.update(pattern_metrics)

        return metrics

    def _weight_svd(self, model, num_layers) -> dict:
        """SVD analysis of Q/K/V weight matrices."""
        concentrations = []
        effective_ranks = []
        top5_vars = []

        for layer_idx in range(num_layers):
            prefix = f'transformer.layers.{layer_idx}.self_attn'
            in_proj_key = f'{prefix}.in_proj_weight'

            # Find the parameter (endswith handles torch.compile _orig_mod. prefix)
            in_proj = None
            for name, param in model.named_parameters():
                if name.endswith(in_proj_key):
                    in_proj = param.data.float()
                    break

            if in_proj is None:
                continue

            d_model = in_proj.shape[1]
            W_Q = in_proj[:d_model]
            W_K = in_proj[d_model:2 * d_model]
            W_V = in_proj[2 * d_model:]

            for W in [W_Q, W_K, W_V]:
                try:
                    _, S, _ = torch.svd_lowrank(W, q=min(20, d_model))
                except Exception:
                    continue

                total_sv = S.sum().item()
                concentration = S[0].item() / (total_sv + 1e-8)
                effective_rank = (total_sv ** 2) / ((S ** 2).sum().item() + 1e-8)

                sv_squared = S ** 2
                total_var = sv_squared.sum().item()
                top5_var = sv_squared[:5].sum().item() / (total_var + 1e-8)

                concentrations.append(concentration)
                effective_ranks.append(effective_rank)
                top5_vars.append(top5_var)

        if not concentrations:
            return {}

        return {
            'sv_concentration': np.mean(concentrations),
            'effective_rank': np.mean(effective_ranks),
            'top5_explained_var': np.mean(top5_vars),
        }

    @torch.no_grad()
    def _attention_patterns(self, model, loader, num_layers) -> dict:
        """Compute attention patterns on actual data.

        Manually propagates a batch through the transformer layers,
        capturing attention weights at each layer. Works with
        torch.compiled models by unwrapping to _orig_mod.
        """
        raw_model = getattr(model, '_orig_mod', model)
        device = next(raw_model.parameters()).device

        # Get a batch from the loader
        try:
            batch = next(iter(loader))
        except StopIteration:
            return {}

        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
        else:
            inputs = batch.to(device)

        # Mod arithmetic batches are (batch, 3) = [a, b, target]; model expects [a, b]
        pos_len = raw_model.pos_embedding.shape[0]
        if inputs.dim() == 2 and inputs.shape[1] > pos_len:
            inputs = inputs[:, :pos_len]

        # Compute embeddings
        was_training = raw_model.training
        raw_model.eval()

        try:
            x = raw_model.embedding(inputs) + raw_model.pos_embedding
            seq_len = x.shape[1]

            # Propagate through transformer layers, capturing attention
            all_attn_weights = []
            layers = raw_model.transformer.layers

            for layer in layers:
                # norm_first=True: norm1 -> self_attn -> residual -> norm2 -> ff -> residual
                x_normed = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(
                    x_normed, x_normed, x_normed,
                    need_weights=True, average_attn_weights=False,
                )
                all_attn_weights.append(attn_w)  # (batch, heads, seq, seq)

                # Complete layer forward (dropout is no-op in eval)
                x = x + attn_out
                x2 = layer.norm2(x)
                x2 = layer.linear1(x2)
                x2 = layer.activation(x2)
                x2 = layer.linear2(x2)
                x = x + x2
        finally:
            if was_training:
                raw_model.train()

        if not all_attn_weights:
            return {}

        # Compute metrics from attention weights
        metrics = {}
        max_entropy = np.log(seq_len) if seq_len > 1 else 1.0
        all_entropies = []
        all_variances = []

        for layer_idx, attn_w in enumerate(all_attn_weights):
            # attn_w: (batch, heads, seq, seq)
            # Each row is a probability distribution over seq positions

            # Entropy: mean across batch, heads, query positions
            # H = -sum(p * log(p))
            entropy = -(attn_w * torch.log(attn_w + 1e-12)).sum(dim=-1)  # (batch, heads, seq)
            norm_entropy = entropy / max_entropy
            layer_entropy = norm_entropy.mean().item()
            metrics[f'attn_entropy/{layer_idx}'] = layer_entropy
            all_entropies.append(layer_entropy)

            # Cross-input variance: for each (head, query_pos, key_pos),
            # compute variance across the batch dimension.
            # High variance = attention depends on input (memorization)
            # Low variance = attention is input-independent (algorithm)
            batch_var = attn_w.var(dim=0)  # (heads, seq, seq)
            layer_var = batch_var.mean().item()
            metrics[f'attn_variance/{layer_idx}'] = layer_var
            all_variances.append(layer_var)

        if all_entropies:
            metrics['attn_entropy'] = sum(all_entropies) / len(all_entropies)
        if all_variances:
            metrics['attn_variance'] = sum(all_variances) / len(all_variances)

        return metrics
