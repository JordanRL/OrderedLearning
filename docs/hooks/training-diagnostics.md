# Training Diagnostics

> Standard training diagnostics: loss volatility, gradient noise, and update ratios.

**Type:** Observer
**Lifecycle Points:** POST_STEP + POST_EPOCH (epoch loop); POST_STEP + SNAPSHOT (step loop)
**Loop Compatibility:** both

## What It Does

The training diagnostics hook collects standard training health metrics at each step and reports aggregate statistics at epoch/snapshot boundaries. It tracks loss statistics (mean, variance, autocorrelation), gradient norm statistics, and the ratio of parameter updates to parameter magnitudes.

These are general-purpose diagnostics useful for any training run, independent of the ordering or curriculum being used. They help identify training instabilities, learning rate issues, and convergence problems.

## Computational Cost

Negligible. Records scalar values per step (loss, gradient norm) and computes summary statistics at reporting intervals. No additional forward or backward passes.

## Assumptions and Compatibility

- Works in both epoch and step loops
- Gradient norms require gradients to be populated on model parameters
- Per-step values are accumulated internally; aggregate metrics reported at POST_EPOCH/SNAPSHOT

## Metrics

### `loss_mean`
- **Formula:** Mean of per-step losses over the reporting interval
- **Range:** [0, +inf)
- **Interpretation:** Average training loss. Tracks overall optimization progress.

### `loss_std`
- **Formula:** Standard deviation of per-step losses over the interval
- **Range:** [0, +inf)
- **Interpretation:** Loss variability within the interval. High std indicates noisy or unstable training.

### `loss_volatility`
- **Formula:** `loss_std / loss_mean` — coefficient of variation of the loss
- **Range:** [0, +inf)
- **Interpretation:** Normalized loss variability. Values > 1 indicate the loss fluctuations are larger than the mean loss — likely a problem.

### `loss_autocorrelation`
- **Formula:** Lag-1 autocorrelation of the per-step loss sequence
- **Range:** [-1, 1]
- **Interpretation:** Whether consecutive step losses are correlated. High positive autocorrelation means the loss changes slowly (smooth landscape). Low or negative autocorrelation means the loss jumps around (rough landscape or high learning rate).

### `grad_norm_mean`
- **Formula:** Mean gradient L2 norm over the interval
- **Range:** [0, +inf)
- **Interpretation:** Average gradient magnitude. Tracks gradient signal strength.

### `grad_norm_std`
- **Formula:** Standard deviation of gradient norms over the interval
- **Range:** [0, +inf)
- **Interpretation:** Gradient magnitude variability.

### `grad_norm_cv`
- **Formula:** `grad_norm_std / grad_norm_mean` — coefficient of variation of gradient norms
- **Range:** [0, +inf)
- **Interpretation:** Normalized gradient noise. High values indicate the gradient magnitude varies significantly between steps.

### `grad_norm_max`
- **Formula:** Maximum gradient norm observed in the interval
- **Range:** [0, +inf)
- **Interpretation:** Worst-case gradient magnitude. Detects gradient spikes that could destabilize training.

### `update_ratio_mean`
- **Formula:** Mean of `||Δθ|| / ||θ||` per step — ratio of update magnitude to parameter magnitude
- **Range:** [0, +inf)
- **Interpretation:** How much parameters change relative to their size each step. Values around 1e-3 to 1e-4 are typical. Much larger values indicate the learning rate may be too high.

### `update_ratio_std`
- **Formula:** Standard deviation of the per-step update ratios
- **Range:** [0, +inf)
- **Interpretation:** Variability of update magnitudes relative to parameter size.

### `weight_norm`
- **Formula:** Total parameter L2 norm at the end of the interval
- **Range:** [0, +inf)
- **Interpretation:** Current model weight magnitude. Tracks weight growth or decay.
