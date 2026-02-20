# Token Gradient

> Per-token gradient distribution analysis.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The token gradient hook analyzes how gradient magnitudes are distributed across individual tokens in the embedding layer. It measures whether gradients are concentrated on a few tokens or spread evenly, and whether tokens within stride groups show coherent gradient behavior.

This is particularly relevant for the modular arithmetic experiment where the embedding layer maps discrete tokens (numbers 0 to p-1) to learned vectors. Stride-based ordering groups tokens that are related modulo the stride, so this hook tests whether those groups receive coherent gradient signal. High concentration on specific tokens suggests the network is learning particular number representations; stride group coherence suggests the ordering is creating meaningful gradient structure.

## Computational Cost

Negligible. Operates on the embedding gradient matrix already present after the backward pass. Computes per-token norms and simple statistics.

## Assumptions and Compatibility

- Requires a model with an embedding layer that has populated gradients
- Designed for the modular arithmetic experiment's token structure
- Epoch-loop only
- Stride group metrics require a stride value in the run context

## Metrics

### `gradient_sparsity`
- **Formula:** Fraction of tokens with gradient norm below 1% of the maximum token gradient norm
- **Range:** [0, 1]
- **Interpretation:** High sparsity means most tokens receive negligible gradient; learning is focused on a few tokens.

### `gradient_gini`
- **Formula:** Gini coefficient of the per-token gradient norms
- **Range:** [0, 1]
- **Interpretation:** 0 = perfectly equal gradient distribution across tokens. 1 = all gradient concentrated on a single token. Measures inequality of gradient allocation.

### `stride_group_variance`
- **Formula:** Variance of mean gradient norms across stride groups (groups of tokens spaced by the stride value)
- **Range:** [0, +inf)
- **Interpretation:** High variance means some stride groups receive much more gradient signal than others, indicating the ordering creates differential learning rates across groups.

### `stride_group_max_ratio`
- **Formula:** Ratio of the maximum stride group mean norm to the minimum stride group mean norm
- **Range:** [1, +inf)
- **Interpretation:** How much more gradient the most-active stride group receives compared to the least-active. Large ratios indicate strong stride-dependent gradient structure.

### `tokens_for_50pct`
- **Formula:** Minimum number of tokens (sorted by gradient norm descending) needed to account for 50% of total gradient energy
- **Range:** [1, num_tokens]
- **Interpretation:** Lower values indicate more concentrated gradient. If 10 tokens out of 9973 account for half the gradient energy, learning is highly focused.

### `tokens_for_90pct`
- **Formula:** Minimum number of tokens needed to account for 90% of total gradient energy
- **Range:** [1, num_tokens]
- **Interpretation:** Similar to `tokens_for_50pct` but captures the broader tail of gradient activity.

### `concentration_ratio`
- **Formula:** `tokens_for_50pct / total_tokens`
- **Range:** (0, 1]
- **Interpretation:** Normalized measure of gradient concentration. Values near 0 mean extreme concentration; values near 0.5 mean roughly uniform distribution.
