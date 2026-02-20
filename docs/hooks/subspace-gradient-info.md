# Subspace Gradient Info

> Gradient subspace dimensionality and information content via sliding-window SVD.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The subspace gradient info hook maintains a sliding window of recent gradient vectors and performs SVD (singular value decomposition) to analyze the dimensionality and structure of the gradient subspace. It answers the question: how many independent directions does the optimizer actually explore over recent epochs?

Low-dimensional gradient subspaces (where a few singular vectors explain most variance) indicate that optimization is constrained to a low-dimensional manifold — a sign of structured learning. High-dimensional subspaces indicate diverse, potentially conflicting gradient directions. The hook also projects the gradient subspace onto the direction toward a known solution to measure how much of the gradient information content is aligned with the solution.

## Computational Cost

Moderate to high. Performs truncated SVD on a (window_size × parameter_count) matrix at every epoch. With default `n_components=20` and `window_size=50`, this is a rank-20 SVD of a 50-by-millions matrix, which takes a few seconds. Memory usage: `window_size` copies of the full gradient vector.

## Assumptions and Compatibility

- Requires gradients to be populated on model parameters
- Meaningful metrics require the window to be sufficiently full (at least `n_components` epochs)
- Epoch-loop only
- `needs_reference_weights=True` — requires a reference solution checkpoint for solution-aligned metrics
- Memory scales as `window_size × parameter_count`

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `50` | Number of recent gradients in the sliding window |
| `n_components` | `20` | Number of SVD components to compute |

## Metrics

### `dims_for_90pct`
- **Formula:** Minimum number of singular vectors needed to explain 90% of the total gradient variance
- **Range:** [1, n_components]
- **Interpretation:** Effective dimensionality of the gradient subspace. Low values (1-3) mean gradients are confined to a low-dimensional manifold. High values indicate diverse gradient directions.

### `participation_ratio`
- **Formula:** `(Σ σ_i)² / Σ σ_i²` where σ_i are singular values
- **Range:** [1, n_components]
- **Interpretation:** A smooth measure of effective dimensionality. PR = 1 means all variance is in one direction; PR = n means variance is uniformly spread across n directions.

### `top_sv_ratio`
- **Formula:** `σ_1 / Σ σ_i` — fraction of total singular value mass in the top component
- **Range:** [0, 1]
- **Interpretation:** Dominance of the leading gradient direction. Values near 1 mean one direction dominates; low values mean gradient information is distributed.

### `svd_total_variance`
- **Formula:** Sum of all computed singular values
- **Range:** [0, +inf)
- **Interpretation:** Total gradient "energy" in the subspace. Tracks overall gradient activity level.

### `top1_explained` / `top5_explained` / `top10_explained`
- **Formula:** Fraction of total variance explained by the top 1/5/10 singular vectors
- **Range:** [0, 1]
- **Interpretation:** How much of the gradient information is captured by the leading components. Rapid saturation (e.g., top5 ≈ 0.99) indicates a very low-dimensional subspace.

### `grad_energy_fraction_toward_solution`
- **Formula:** Fraction of total gradient variance that lies in the direction of the reference solution
- **Range:** [0, 1]
- **Interpretation:** How much of the gradient subspace's information content is aligned with the known solution direction. Higher values mean the optimizer is spending more of its "budget" moving toward the solution.

### `top{k}_energy_fraction_toward_solution`
- **Formula:** For each of the top-k singular vectors, the squared projection onto the solution direction, summed and normalized
- **Range:** [0, 1]
- **Interpretation:** How much of the leading gradient components are aligned with the solution. Compares the solution alignment of the dominant gradient directions vs. the full subspace.
