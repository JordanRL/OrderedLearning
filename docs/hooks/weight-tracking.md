# Weight Tracking

> Weight norms, spectral properties, effective rank, and gradient-weight alignment per layer.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The weight tracking hook monitors the structural properties of the model's weight matrices across training. For each layer, it computes the weight norm, the top singular value (spectral norm), the effective rank of the weight matrix, and the alignment between the gradient and the weight matrix itself.

These metrics provide insight into the complexity and conditioning of the learned representations. Growing spectral norms can indicate instability; changing effective rank reveals whether the network is learning low-dimensional or high-dimensional transformations. Gradient-weight alignment measures whether the gradient is pushing the weights in a direction aligned with their current structure (reinforcing) or orthogonal to it (diversifying).

## Computational Cost

Moderate. Performs SVD on each weight matrix to obtain singular values. For the small transformers used in grokking experiments (d_model=128), this is fast. Cost scales with model size.

## Assumptions and Compatibility

- Works with any model that has named parameter matrices
- SVD is computed on 2D weight matrices; 1D parameters (biases) have simpler metrics
- Epoch-loop only
- Gradients must be populated for the `grad_weight_align` metric

## Metrics

### `weight_norm/{layer}`
- **Formula:** `||W||_F` — Frobenius norm of the weight matrix for the named layer
- **Range:** [0, +inf)
- **Interpretation:** Overall magnitude of the weight matrix. Growing norms may indicate lack of regularization; stable norms indicate balanced learning.

### `top_sv/{layer}`
- **Formula:** `σ_1(W)` — largest singular value of the weight matrix
- **Range:** [0, +inf)
- **Interpretation:** Spectral norm of the layer. Controls the maximum amplification the layer can apply to inputs. Related to Lipschitz constant and training stability.

### `effective_rank/{layer}`
- **Formula:** `exp(H(σ))` where H is the entropy of the normalized singular value distribution
- **Range:** [1, min(rows, cols)]
- **Interpretation:** How many dimensions the weight matrix effectively uses. Low effective rank means the layer's transformation is approximately low-dimensional. Changes in effective rank track representation complexity.

### `grad_weight_align/{layer}`
- **Formula:** Cosine similarity between the vectorized gradient and the vectorized weight matrix: `cos(vec(∇W), vec(W))`
- **Range:** [-1, 1]
- **Interpretation:** +1 means the gradient reinforces the existing weight structure (amplifying current representations). -1 means the gradient opposes current weights. 0 means the gradient is orthogonal (adding new structure). Transitions from reinforcing to orthogonal often coincide with phase transitions.

### `total_weight_norm`
- **Formula:** Sum of Frobenius norms across all layers
- **Range:** [0, +inf)
- **Interpretation:** Total model weight magnitude.

### `mean_weight_norm`
- **Formula:** Mean Frobenius norm across layers
- **Range:** [0, +inf)
- **Interpretation:** Average per-layer weight magnitude.

### `mean_top_sv` / `max_top_sv`
- **Formula:** Mean and maximum top singular values across all layers
- **Range:** [0, +inf)
- **Interpretation:** Summary statistics for spectral norms. `max_top_sv` identifies the most amplifying layer.

### `mean_effective_rank`
- **Formula:** Mean effective rank across all layers
- **Range:** [1, max(min(rows, cols))]
- **Interpretation:** Average representation complexity across the network.

### `mean_grad_weight_align`
- **Formula:** Mean gradient-weight alignment across all layers
- **Range:** [-1, 1]
- **Interpretation:** Overall tendency of gradients to reinforce or restructure current weights.
