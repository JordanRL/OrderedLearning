# Attention

> Attention weight structure (SVD) and attention patterns.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH (epoch loop), SNAPSHOT (step loop)
**Loop Compatibility:** both

## What It Does

The attention hook analyzes the structure of learned attention weights and the attention patterns they produce on data. It performs SVD on the attention weight matrices to measure their effective rank and singular value concentration, and runs a forward pass on sample data to capture actual attention distributions.

For grokking experiments, attention structure changes are a key indicator of learning transitions. During memorization, attention patterns tend to be diffuse or position-dependent. During generalization, attention sharpens and becomes more structured, often developing low-rank patterns that reflect the algebraic structure of the task.

## Computational Cost

Low to moderate. Performs SVD on attention weight matrices (small for 2-layer transformers). Runs one forward pass on sample data to capture attention patterns. In step-loop mode, fires only at SNAPSHOT intervals to limit cost.

## Assumptions and Compatibility

- Requires a transformer model with accessible attention weight matrices
- Forward pass for attention pattern capture uses a sample batch from the run context
- Works in both epoch and step loops (with different lifecycle points)

## Metrics

### `sv_concentration`
- **Formula:** `σ_1 / Σ σ_i` — fraction of total singular value mass in the top singular value, averaged across attention heads and layers
- **Range:** [0, 1]
- **Interpretation:** How concentrated the attention weight spectrum is. High concentration means attention weights are approximately rank-1, indicating simple attention patterns.

### `effective_rank`
- **Formula:** `exp(H(σ))` where H is the Shannon entropy of the normalized singular value distribution, averaged across heads and layers
- **Range:** [1, min(d_k, d_v)]
- **Interpretation:** Smooth measure of the dimensionality of the attention weight matrix. Low effective rank indicates structured, low-dimensional attention. High effective rank indicates complex or random-like attention weights.

### `top5_explained_var`
- **Formula:** Fraction of total variance (sum of squared singular values) explained by the top 5 singular vectors, averaged across heads and layers
- **Range:** [0, 1]
- **Interpretation:** How much of the attention weight structure is captured by 5 components. Values near 1 indicate low-rank attention patterns.

### `attn_entropy`
- **Formula:** Shannon entropy of the attention distribution, averaged across positions, heads, and layers: `-Σ a_i log(a_i)`
- **Range:** [0, log(seq_len)]
- **Interpretation:** How diffuse or concentrated attention is. Low entropy means attention is sharply focused on specific positions. High entropy means attention is spread uniformly. A global average across all layers and heads.

### `attn_entropy/{layer}`
- **Formula:** Same as `attn_entropy` but computed per layer
- **Range:** [0, log(seq_len)]
- **Interpretation:** Per-layer attention entropy. Allows tracking how attention sharpness differs across network depth.

### `attn_variance`
- **Formula:** Variance of attention weights across positions, averaged across heads and layers
- **Range:** [0, +inf)
- **Interpretation:** How much the attention distribution varies across positions. Low variance means attention is nearly uniform; high variance means attention is differentiated.

### `attn_variance/{layer}`
- **Formula:** Same as `attn_variance` but computed per layer
- **Range:** [0, +inf)
- **Interpretation:** Per-layer attention variance.
