# Parameter Delta

> Parameter update magnitude tracking.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH (epoch loop), SNAPSHOT (step loop)
**Loop Compatibility:** both

## What It Does

The parameter delta hook measures how much model parameters change between observations. It stores a snapshot of the model's parameters and, at the next firing, computes the absolute and relative magnitude of the parameter update.

This provides a direct measure of learning rate at the parameter level — how much the model is actually changing. During grokking, parameter deltas often show distinct phases: large updates during memorization, small updates during the plateau, and renewed updates during generalization.

## Computational Cost

Negligible. Stores one copy of the parameter vector and computes norms. No additional forward or backward passes.

## Assumptions and Compatibility

- Works in both epoch and step loops
- First firing stores initial parameters; metrics are available from the second firing onward
- Memory: one copy of the full parameter vector

## Metrics

### `relative_delta`
- **Formula:** `||θ_t - θ_{t-1}|| / ||θ_{t-1}||` — relative parameter change
- **Range:** [0, +inf)
- **Interpretation:** Fractional change in parameters since last observation. Values of 0.01 mean parameters changed by 1% relative to their magnitude. Tracks effective learning speed.

### `absolute_delta`
- **Formula:** `||θ_t - θ_{t-1}||_2` — L2 norm of the parameter change
- **Range:** [0, +inf)
- **Interpretation:** Absolute magnitude of the parameter update. Compare across strategies to see which orderings produce larger parameter movements.

### `param_norm`
- **Formula:** `||θ_t||_2` — current parameter vector norm
- **Range:** [0, +inf)
- **Interpretation:** Total magnitude of the model's parameters. Tracks weight growth or decay over training.
