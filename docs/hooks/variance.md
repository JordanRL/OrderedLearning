# Variance

> Gradient variance and stability over sliding windows.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The variance hook maintains a sliding window of recent gradient vectors and computes statistics that characterize gradient stability and interference. It measures how much gradients vary across epochs within the window, the mean pairwise alignment between gradients in the window, and a signal-to-noise ratio.

This hook is designed to detect gradient interference — when different data orderings cause the optimizer to receive conflicting gradient signals across epochs. Low pairwise cosine similarity within the window indicates that successive gradient updates point in different directions, which can slow convergence. Stride-based ordering is hypothesized to reduce this interference compared to random ordering.

## Computational Cost

Moderate. Stores a sliding window of flattened gradient vectors (window_size × parameter_count tensors). Pairwise cosine similarity computation is O(window_size²) dot products. Default window size of 10 keeps this manageable.

## Assumptions and Compatibility

- Requires gradients to be populated on model parameters
- Metrics are only meaningful once the window is full (first `window_size` epochs return partial statistics)
- Epoch-loop only
- Memory usage scales linearly with `window_size × parameter_count`

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `10` | Number of recent gradients to keep in the sliding window |

## Metrics

### `gradient_variance`
- **Formula:** Element-wise variance of gradient vectors in the sliding window, summed across all parameters
- **Range:** [0, +inf)
- **Interpretation:** Total gradient variance across the window. High variance indicates the gradient direction is unstable across epochs. Low variance indicates consistent optimization direction.

### `mean_pairwise_cos`
- **Formula:** Mean cosine similarity across all pairs of gradient vectors in the window
- **Range:** [-1, 1]
- **Interpretation:** Average directional agreement between gradients in the window. Values near +1 indicate all recent gradients point in similar directions (low interference). Values near 0 indicate nearly orthogonal gradients (high interference). Negative values indicate opposing gradients.

### `signal_to_noise`
- **Formula:** `||mean(g)||² / variance(g)` — ratio of squared mean gradient norm to gradient variance
- **Range:** [0, +inf)
- **Interpretation:** How much of the gradient signal is consistent vs. noise. High SNR means the mean gradient direction is strong relative to epoch-to-epoch fluctuations. Low SNR means the useful signal is buried in gradient noise.

### `window_mean_norm`
- **Formula:** `||mean(g_1, g_2, ..., g_w)||_2` — L2 norm of the mean gradient vector in the window
- **Range:** [0, +inf)
- **Interpretation:** Magnitude of the average gradient direction. Can be much smaller than individual gradient norms if gradients point in different directions (cancellation). Compare with individual norms from the `norms` hook to assess cancellation.
