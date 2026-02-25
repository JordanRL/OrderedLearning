# OrderedLearning

An experiment framework for studying how data ordering strategies affect neural network learning. Companion code for the paper: *[TODO: paper title and link]*.

## Overview

OrderedLearning provides a unified system for running ordered dataset learning experiments with:

- **Standardized training loops** -- experiments define building blocks; the framework handles orchestration
- **19 instrumentation hooks** for observing and intervening in training, with 4 sink types (console, CSV, JSONL, W&B)
- **Rich console output** with full-screen live display, silent, and headless modes
- **Strict reproducibility** -- deterministic seeding, environment metadata capture, config snapshots

The primary experiment is `mod_arithmetic`, which trains a small transformer on modular addition `(a + b) mod p` and studies how data ordering affects the grokking phenomenon -- the delayed generalization that emerges long after memorization.

## Installation

```bash
git clone https://github.com/JordanRL/OrderedLearning.git
cd OrderedLearning

# To reproduce paper results, use the exact code version:
git checkout paper-data-v3

# For general use (allows compatible newer versions):
pip install -r requirements.txt

# For exact reproducibility (pinned versions matching the paper):
pip install -r requirements-lock.txt
```

`requirements.txt` uses `>=` minimum bounds for general compatibility. `requirements-lock.txt` pins exact versions for bit-identical reproducibility across environments. Use the lock file when replicating paper results; use the regular file for development or when running on newer hardware that may need updated packages.

Requires Python 3.11+ and PyTorch 2.0+. The `mod_arithmetic` experiment can run on CPU, though it was run on a NVidia 4090 GPU on RunPod for the paper; LM experiments require a CUDA-capable GPU. Apple Silicon (MPS) is supported as a fallback when CUDA is unavailable.

## Replicating the Paper Experiment

The `mod_arithmetic` experiment is self-contained -- no external data or GPU required. This trains a 2-layer transformer on `(a + b) mod 9973` using all four data ordering strategies (stride, random, fixed-random, target) and reports grokking behavior for each.

> **Important:** To reproduce paper results, make sure you are on the `paper-data-v1` tag (`git checkout paper-data-v1`). Later commits may change framework behavior.

### Exact Commands Used For Paper Results

First, these commands were run to generate the solution weights for each strategy:

```bash
python run_experiment.py mod_arithmetic --strategy stride --with-hooks minimal --seed 199 --save-checkpoints
python run_experiment.py mod_arithmetic --strategy random --with-hooks minimal --seed 199 --save-checkpoints
python run_experiment.py mod_arithmetic --strategy target --with-hooks minimal --seed 199 --save-checkpoints
python run_experiment.py mod_arithmetic --strategy fixed-random --with-hooks minimal --seed 199 --save-checkpoints
```

Reference weights for solution-dependent metrics are now resolved automatically from the output directory (e.g., `output/mod_arithmetic/stride/stride_final.pt`). No manual copying is needed — the second run finds the `_final.pt` files written by the first run.

Then the fully instrumented experiment was run on the same pod:

```bash
python run_experiment.py mod_arithmetic --strategy stride --with-hooks full --seed 199 --hook-jsonl --validate-checkpoints
python run_experiment.py mod_arithmetic --strategy random --with-hooks full --seed 199 --hook-jsonl --validate-checkpoints
python run_experiment.py mod_arithmetic --strategy target --with-hooks full --seed 199 --hook-jsonl --validate-checkpoints
python run_experiment.py mod_arithmetic --strategy fixed-random --with-hooks full --seed 199 --hook-jsonl --validate-checkpoints
```

The `analysis_tools/replay_to_wandb.py` script was then used to get the collected metrics from the JSONL logs and upload them to W&B.

**NOTE:** The data the paper discusses was gathered on seed 199, however it is not the only seed that produced comparable results. The experiment was also run without full instrumentation on seeds 31, 42, 242, and 9973, all of which generalized to 99.5% test accuracy within 700 epochs. One seed tested, 555, did not generalize, stalling at approximately 24% test accuracy. Convergence times ranged from 465 epochs (seed 42) to 696 epochs (seed 9973). Five of six seeds tested generalize at the 0.3% data fraction.

### Key configuration flags

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `all` | Ordering strategy: `stride`, `random`, `fixed-random`, `target`, `all` |
| `--epochs` | `5000` | Total training epochs |
| `--p` | `9973` | Prime modulus |
| `--batch-size` | `256` | Batch size |
| `--seed` | `199` | Random seed |
| `--stride` | `floor(sqrt(p))` | Stride value for `stride` ordering |
| `--snapshot-every` | `10` | Interval for hook snapshots |

### With full instrumentation

```bash
# Enable all hooks and write metrics to JSONL
python run_experiment.py mod_arithmetic --with-hooks full --hook-jsonl

# Full instrumentation with W&B logging
python run_experiment.py mod_arithmetic --with-hooks full --hook-wandb my_project

# Full instrumentation with live display
python run_experiment.py mod_arithmetic --strategy stride --with-hooks full --hook-jsonl --live
```

### Expected output

Each strategy run produces files in `output/mod_arithmetic/{strategy}/`:

| File | Description |
|---|---|
| `experiment_config.json` | Full config + environment metadata |
| `summary.json` | Initial and final evaluation metrics |
| `{strategy}_final.pt` | Final model weights |
| `{strategy}.csv` | Hook metrics time series (with `--hook-csv`) |
| `{strategy}.jsonl` | Hook metrics in JSONL (with `--hook-jsonl`) |
| `checkpoints/` | Periodic checkpoints (with `--save-checkpoints`) |
| `traj.pt` | Parameter trajectory (with `--record-trajectory`) |

## Reproducibility and Determinism

A single `--seed` value controls all random number generators:

- Python `random`, NumPy, PyTorch CPU and CUDA RNGs are all seeded
- `torch.use_deterministic_algorithms(True)` is enforced (not `warn_only`)
- cuDNN: `deterministic=True`, `benchmark=False`
- cuBLAS: workspace config set to `:4096:8` for deterministic reductions
- Flash attention and memory-efficient attention are disabled (non-deterministic)
- Hook RNG state is saved and restored around hook execution to prevent hooks from perturbing training

Each run captures full environment metadata in `experiment_config.json`:

```json
{
  "environment": {
    "torch_version": "2.x.x",
    "cuda_version": "12.x",
    "cudnn_version": "...",
    "gpu_name": "NVIDIA ...",
    "gpu_capability": "8.9",
    "float32_matmul_precision": "high",
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "cublas_workspace_config": ":4096:8"
  }
}
```

**Guarantee:** Same seed + same hardware + same code = bit-identical results.

**What can differ:** Different GPU architectures, different CUDA/cuDNN versions, or different PyTorch versions may produce numerically different results due to implementation differences in low-level kernels. The framework warns when loading reference weights from a different environment.

## Running Your Own Experiments

The scaffold generator creates a new experiment package with all required files:

```bash
python create_experiment.py
```

This walks you through experiment setup (base class, strategies, training step type, hyperparameters) and generates a package under `experiments/` with stub implementations ready to fill in.

### Experiment package structure

```
experiments/{name}/
├── __init__.py       # imports runner to trigger @ExperimentRegistry.register
├── runner.py         # runner class extending LMRunner or GrokkingRunner
├── config.py         # @dataclass extending BaseConfig
├── generator.py      # DatasetGenerator subclass
└── loader.py         # DatasetLoader subclass (if needed)
```

Experiments are auto-discovered -- any package under `experiments/` that imports a registered runner class will appear in `python run_experiment.py --list`.

## Instrumentation Hooks

The framework includes 19 training hooks for deep observability into the learning process. Hooks fire at lifecycle points during training, compute metrics, and dispatch them to configurable sinks.

```bash
# Enable a curated hook group
python run_experiment.py mod_arithmetic --with-hooks full --hook-jsonl

# Enable specific hooks
python run_experiment.py mod_arithmetic --hooks norms fourier attention --hook-jsonl

# List all available hooks
python run_experiment.py mod_arithmetic --hooks-list

# Describe a hook's metrics
python run_experiment.py mod_arithmetic --hooks-describe fourier
```

See [docs/instrumentation-hooks.md](docs/instrumentation-hooks.md) for the full hook reference.

## Analysis Tools

Post-experiment analysis tools are provided for visualizing metrics, comparing strategies, and exporting publication-ready tables. See [docs/getting-started.md](docs/getting-started.md) for a walkthrough.

```bash
# List available analysis tools
python analyze_experiment.py --list

# Plot training metrics
python analyze_experiment.py mod_arithmetic metric_plot \
    --metrics training_metrics/loss training_metrics/val_acc \
    --layout overlay --smooth 0.9

# Export a LaTeX comparison table
python analyze_experiment.py mod_arithmetic export_table \
    --metrics training_metrics/val_acc --format latex
```

### Standalone Scripts

**Replay to W&B** — `analysis_tools/replay_to_wandb.py` reads JSONL metric logs and uploads them to Weights & Biases without re-running the experiment.

```bash
python -m analysis_tools.replay_to_wandb \
    --jsonl output/mod_arithmetic/stride/stride.jsonl \
    --project my-project --group experiment-group --strategy stride \
    --config output/mod_arithmetic/stride/experiment_config.json
```

**Dataset DFT Analysis** — `analysis_tools/dataset_dft_analysis.py` performs spectral analysis of mod_arithmetic dataset orderings, comparing frequency content across data ordering strategies.

```bash
python -m analysis_tools.dataset_dft_analysis --p 9973 --train-size 300000 --seed 42
```

## Console Modes

| Flag | Mode | Behavior |
|---|---|---|
| *(default)* | NORMAL | Standard scrolling Rich output |
| `--live` | LIVE | Full-screen layout with real-time metrics sidebar |
| `--silent` | SILENT | Progress bar only; all other output suppressed |
| `--no-console-output` | NULL | No console output at all |

**Priority:** When multiple flags are given, the highest-priority mode wins: NULL (`--no-console-output`) > SILENT (`--silent`) > LIVE (`--live`) > NORMAL (default). For example, `--live --silent` produces SILENT mode.

## Security Note

Checkpoint files (`.pt`) are loaded with `torch.load(weights_only=False)` because they contain optimizer state dicts, RNG states, and experiment-specific training state that require unpickling. **Only load checkpoints that you have generated yourself or that come from a trusted source.** Loading an untrusted checkpoint file can execute arbitrary code. See the [PyTorch serialization docs](https://pytorch.org/docs/stable/notes/serialization.html) for details.

## Docker

A Dockerfile is provided for GPU cloud environments (tested on RunPod). Dependencies are baked into the image; code is pulled at startup via `entrypoint.sh`.

```bash
# Build
DOCKER_IMAGE_NAME=your-user/orderedlearning ./build-docker.sh

# Or build directly
docker build -t orderedlearning .
```

Environment variables for `entrypoint.sh`:

| Variable | Default | Description |
|---|---|---|
| `REPO_URL` | *(must be set)* | GitHub repo URL without protocol (e.g., `github.com/user/OrderedLearning.git`) |
| `REPO_BRANCH` | `master` | Branch to clone/pull |
| `GITHUB_TOKEN` | *(optional)* | GitHub PAT for private repos |
| `WORKSPACE_DIR` | `/workspace/OrderedLearning` | Local clone directory |

## License

MIT -- see [LICENSE](LICENSE).
