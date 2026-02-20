# Counterfactual

> Counterfactual ordering analysis via shuffled epoch comparisons.

**Type:** Intervention
**Lifecycle Points:** POST_EPOCH (epochs ≤ 49), SNAPSHOT (epochs ≥ 50)
**Loop Compatibility:** epoch

## What It Does

The counterfactual hook decomposes the observed gradient into a **content component** (what is learned regardless of ordering) and an **ordering component** (what depends specifically on the data order). It does this by running K additional forward-backward passes on randomly shuffled versions of the current epoch's data, computing their gradients, and comparing with the observed gradient from the ordered data.

The content component is estimated as the mean gradient across shuffled orderings (representing what the data teaches independent of order). The ordering component is the difference between the observed gradient and this content mean, representing the gradient signal that arises specifically from the chosen data ordering.

This decomposition is the core analytical tool for quantifying how much data ordering contributes to learning, and whether that contribution is aligned with the solution direction.

Fires every epoch during early training (≤49) when dynamics change rapidly, then switches to SNAPSHOT intervals for efficiency.

## Computational Cost

**High.** Runs K additional forward-backward passes per firing, plus shuffling and data loading. With K=3 (default), this roughly 4× the cost of a normal epoch when firing. Switches to SNAPSHOT intervals after epoch 49 to amortize cost.

Memory: stores a sliding window of K shuffled gradients for running statistics.

## Assumptions and Compatibility

- Epoch-loop only (requires full-epoch gradient computation)
- `needs_reference_weights=True` — requires a reference solution checkpoint for solution-alignment metrics
- Intervention hook: temporarily recomputes gradients on shuffled data, then restores model state
- K=3 is sufficient for the decomposition (validated by `counterfactual_validator`)

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `k` | `3` | Number of shuffled replicates per firing |
| `window_size` | `20` | Sliding window for running statistics |

## Metrics

### `counterfactual_mean_norm`
- **Formula:** `||g_shuffled_mean||_2` — L2 norm of the mean shuffled gradient
- **Range:** [0, +inf)
- **Interpretation:** Magnitude of the content component. Represents the gradient signal from data content alone.

### `content_component_norm`
- **Formula:** Same as `counterfactual_mean_norm` (the content direction magnitude)
- **Range:** [0, +inf)
- **Interpretation:** How strong the ordering-independent gradient signal is.

### `ordering_component_norm`
- **Formula:** `||g_observed - g_shuffled_mean||_2` — L2 norm of the ordering component
- **Range:** [0, +inf)
- **Interpretation:** Magnitude of the gradient signal attributable to data ordering. Larger values mean ordering has a stronger effect on the gradient.

### `ordering_fraction`
- **Formula:** `ordering_component_norm / (content_component_norm + ordering_component_norm)`
- **Range:** [0, 1]
- **Interpretation:** What fraction of the total gradient signal comes from ordering. Values near 0 mean ordering doesn't matter; values near 1 mean the gradient is dominated by ordering effects.

### `ordering_alignment`
- **Formula:** Cosine similarity between the ordering component and the observed gradient
- **Range:** [-1, 1]
- **Interpretation:** Whether the ordering effect reinforces (+1) or opposes (-1) the overall gradient direction. Positive alignment means ordering helps; negative means it hurts.

### `content_grad_cossim_to_solution`
- **Formula:** Cosine similarity between the content component and the direction toward the reference solution
- **Range:** [-1, 1]
- **Interpretation:** How well the content component (ordering-independent learning) is aligned with the known solution.

### `ordering_grad_cossim_to_solution`
- **Formula:** Cosine similarity between the ordering component and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** How well the ordering-dependent gradient is aligned with the solution. Positive values indicate that data ordering is specifically helping the network move toward the solution.

### `cf_grad_cossim_to_solution`
- **Formula:** Cosine similarity between the mean shuffled gradient and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Solution alignment of the counterfactual (shuffled) gradients.

### `content_energy_fraction_toward_solution`
- **Formula:** Squared projection of the content component onto the solution direction, divided by total content energy
- **Range:** [0, 1]
- **Interpretation:** What fraction of the content component's energy is directed toward the solution.

### `ordering_energy_fraction_toward_solution`
- **Formula:** Squared projection of the ordering component onto the solution direction, divided by total ordering energy
- **Range:** [0, 1]
- **Interpretation:** What fraction of the ordering signal's energy is directed toward the solution. Key metric: high values mean ordering is efficiently pushing toward the answer.
