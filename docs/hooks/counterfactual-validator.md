# Counterfactual Validator

> Validate K-sufficiency of the counterfactual decomposition.

**Type:** Debug Intervention
**Lifecycle Points:** POST_EPOCH (epochs ≤ 49), SNAPSHOT (epochs ≥ 50)
**Loop Compatibility:** epoch

## What It Does

The counterfactual validator checks whether K=3 shuffled replicates (used by the `counterfactual` hook) is sufficient for a stable content/ordering decomposition. It runs K+1 shuffled forward-backward passes and checks whether adding one more replicate materially changes the content component estimate.

If K is sufficient, the K+1 content estimate should be close to the K estimate (low norm gap, high cosine similarity). If K is insufficient, the estimates will diverge, indicating that more replicates are needed for a reliable decomposition.

This is a **debug intervention** — it is excluded from bulk selection (`all`, `observers`) and only activates when explicitly named or via `with_debug`. It is computationally expensive and intended for validation, not routine use.

## Computational Cost

**Very high.** Runs K+1 additional forward-backward passes per firing. With K=3, this is 4 extra full-epoch computations on top of the counterfactual hook's own 3. Only use for validation purposes.

## Assumptions and Compatibility

- Epoch-loop only
- `needs_reference_weights=True` — requires a reference solution checkpoint
- Must be explicitly enabled (debug hook, excluded from bulk selection)
- Designed to validate the `counterfactual` hook's methodology

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `k` | `3` | Number of shuffled replicates (matches the counterfactual hook's K) |

## Metrics

### `kp1_content_norm`
- **Formula:** `||mean(g_1, ..., g_{K+1})||_2` — content estimate from K+1 replicates
- **Range:** [0, +inf)
- **Interpretation:** Content component magnitude estimated with one additional replicate.

### `k_content_norm_max` / `k_content_norm_min` / `k_content_norm_mean`
- **Formula:** Statistics over leave-one-out K-subsets: for each subset of K replicates (out of K+1), compute the content norm
- **Range:** [0, +inf)
- **Interpretation:** Spread of K-subset content estimates. Small spread indicates K is sufficient.

### `kp1_norm_strictly_lower`
- **Formula:** Boolean (1 or 0) — whether `kp1_content_norm < k_content_norm_max`
- **Range:** {0, 1}
- **Interpretation:** A necessary condition for convergence: adding more replicates should not increase the estimated content norm. If consistently 1, K is likely sufficient.

### `norm_convergence_gap`
- **Formula:** `(k_content_norm_max - kp1_content_norm) / kp1_content_norm`
- **Range:** (-inf, +inf)
- **Interpretation:** Relative gap between K and K+1 estimates. Small positive values indicate convergence. Large or negative values indicate K is insufficient.

### `k_to_kp1_cossim_mean` / `k_to_kp1_cossim_min`
- **Formula:** Cosine similarity between each K-subset content estimate and the K+1 estimate
- **Range:** [-1, 1]
- **Interpretation:** Directional agreement between K and K+1 estimates. Values near 1 indicate K captures the correct content direction. Minimum across subsets is the most conservative check.

### `kp1_cossim_to_solution`
- **Formula:** Cosine similarity between the K+1 content estimate and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Solution alignment of the more precise (K+1) content estimate.

### `k_cossim_to_solution_mean` / `k_cossim_to_solution_std` / `k_cossim_to_solution_spread`
- **Formula:** Statistics of solution alignment across K-subset content estimates
- **Range:** mean/std in [-1,1]/[0,1]; spread = max - min
- **Interpretation:** Stability of solution alignment across subsets. Low spread and std indicate the decomposition is robust to the specific set of shuffled replicates used.
