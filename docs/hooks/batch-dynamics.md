# Batch Dynamics

> Batch-level gradient dynamics: autocorrelation, accumulation efficiency, and subspace rank.

**Type:** Observer
**Lifecycle Points:** POST_STEP + POST_EPOCH (epoch loop); POST_STEP + SNAPSHOT (step loop)
**Loop Compatibility:** both

## What It Does

The batch dynamics hook analyzes the temporal structure of gradients at the batch level. It maintains a buffer of recent per-step gradients and computes:

1. **Gradient autocorrelation** at multiple lags — how correlated the gradient at step t is with the gradient at step t-k. Under ordered data, systematic correlations at specific lags can emerge.

2. **Accumulation efficiency** over multiple windows — how much the accumulated gradient over a window of steps compares to what random accumulation would produce. Efficiency > 1 indicates constructive interference (gradients reinforce each other); efficiency < 1 indicates destructive interference.

3. **Gradient subspace rank** — the effective dimensionality of the recent gradient buffer, measuring how diverse the gradient directions are.

These metrics are reported at aggregate points (POST_EPOCH or SNAPSHOT) after accumulating per-step data.

## Computational Cost

Moderate. Stores a buffer of recent gradient vectors (size = max window length × parameter count). Autocorrelation and efficiency computations involve dot products across the buffer. Cost scales with buffer size and parameter count.

## Assumptions and Compatibility

- Works in both epoch and step loops
- Per-step gradients are accumulated internally; aggregate metrics are reported at POST_EPOCH/SNAPSHOT
- Memory: gradient buffer of size max(max_lag, max(windows)) × parameter count

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_lag` | `50` | Maximum autocorrelation lag |
| `lags` | `None` | Specific lag values to compute (if None, uses `[1, 2, 5, 10, 20, 50]` up to `max_lag`) |
| `windows` | `None` | Accumulation efficiency window sizes (if None, uses `[5, 10, 20, 50]`) |

## Metrics

### `lag_{k}`
- **Formula:** `cos(g_t, g_{t-k})` averaged over recent steps — autocorrelation at lag k
- **Range:** [-1, 1]
- **Interpretation:** Gradient correlation at lag k. Positive values mean gradients k steps apart tend to point in similar directions. Under stride-ordered data, specific lags corresponding to the stride pattern may show elevated correlation.

### `autocorrelation_mean`
- **Formula:** Mean autocorrelation across all computed lags
- **Range:** [-1, 1]
- **Interpretation:** Overall temporal correlation of gradients. Higher values indicate more temporally structured gradient sequences.

### `efficiency_{w}`
- **Formula:** `||Σ_{i=0}^{w-1} g_{t-i}||² / (w · Σ_{i=0}^{w-1} ||g_{t-i}||²)` — ratio of accumulated gradient energy to sum of individual energies over window w
- **Range:** [0, w]
- **Interpretation:** Values > 1/w indicate constructive interference (gradients tend to align within the window). Value of 1 means perfect constructive interference. Value of 1/w means completely random directions. Ordered data should produce higher efficiency than random data.

### `effective_rank`
- **Formula:** Effective rank (exponential of singular value entropy) of the gradient buffer matrix
- **Range:** [1, buffer_size]
- **Interpretation:** How many independent gradient directions the recent buffer contains. Low rank means gradients are confined to a low-dimensional subspace. High rank means diverse gradient directions.

### `top1_variance`
- **Formula:** Fraction of total gradient variance explained by the first singular vector of the buffer
- **Range:** [0, 1]
- **Interpretation:** Dominance of the leading gradient direction. High values indicate a single direction dominates recent gradient activity.
