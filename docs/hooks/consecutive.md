# Consecutive

> Cosine similarity between consecutive epoch gradients.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The consecutive hook measures the directional alignment between the gradient computed in the current epoch and the gradient from the previous epoch. It stores the previous epoch's gradient and computes cosine similarity and angular distance between successive gradient directions.

This metric reveals gradient consistency across epochs. High similarity indicates the optimization is following a stable direction; low or negative similarity indicates the gradient direction is changing significantly between epochs. In ordered training, stride-based data ordering can produce more consistent gradient directions than random ordering, potentially explaining faster convergence.

## Computational Cost

Negligible. Stores one copy of the flattened gradient vector (same size as total parameter count). Cosine similarity is a single dot product plus two norms.

## Assumptions and Compatibility

- Requires gradients to be populated on model parameters
- Returns no metrics on the first epoch (no previous gradient to compare)
- Epoch-loop only

## Metrics

### `cos_sim`
- **Formula:** `cos(g_t, g_{t-1}) = (g_t · g_{t-1}) / (||g_t|| · ||g_{t-1}||)`
- **Range:** [-1, 1]
- **Interpretation:** +1 means identical gradient directions, 0 means orthogonal, -1 means opposite. Values consistently near +1 suggest the loss landscape has a stable descent direction. Values near 0 suggest high curvature or interference between data points.

### `angle_degrees`
- **Formula:** `arccos(cos_sim) × 180 / π`
- **Range:** [0, 180]
- **Interpretation:** Angular distance in degrees. 0° = identical direction, 90° = orthogonal, 180° = opposite. More intuitive than cosine similarity for understanding how much the gradient "turns" between epochs.
