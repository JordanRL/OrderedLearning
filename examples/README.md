# Examples

This directory contains example configurations, sample outputs, and tutorial notebooks for the OrderedLearning framework.

## Quick Start

Run a quick verification experiment using the included config:

```bash
python run_experiment.py --config examples/configs/quick_mod_arithmetic.json
```

This runs a small modular arithmetic experiment (p=97, 100 epochs) that completes in under a minute on CPU.

## Directory Structure

### `configs/`

Example experiment configuration files compatible with `--config`:

- **`quick_mod_arithmetic.json`** — A fast mod_arithmetic run (small prime, few epochs) for verifying installation and understanding the workflow.

### `sample_output/`

Reference output files showing the structure produced by a completed experiment run:

- **`sample_output/mod_arithmetic/stride/experiment_config.json`** — Configuration snapshot saved at experiment start, including environment metadata.
- **`sample_output/mod_arithmetic/stride/summary.json`** — Results summary saved at experiment completion, including timing, model info, and evaluation deltas.

These files illustrate the output schema described in the [README](../README.md#expected-output).

### `notebooks/`

- **`notebooks/analysis_walkthrough.ipynb`** — Interactive tutorial demonstrating the analysis tools with synthetic data. No prior experiment run is required.

## See Also

- [Getting Started Guide](../docs/getting-started.md) — Full onboarding tutorial
- [README](../README.md) — Project overview and paper replication commands
