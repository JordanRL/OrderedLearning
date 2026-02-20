"""Replay JSONL metric logs into Weights & Biases.

Reads a JSONL file produced by JSONLSink and creates a new W&B run
with the same metrics, without the console logging bloat that Live
mode screen refreshes produce.

Each JSONL file corresponds to one strategy (one W&B run). To replay
a full experiment, invoke once per strategy with the same --group.

Usage:
    python -m analysis_tools.replay_to_wandb \
        --jsonl path/to/metrics.jsonl \
        --project my-project \
        --group experiment-group \
        --strategy stride \
        [--config path/to/experiment_config.json]

    # Replay multiple strategies into the same group:
    for s in stride random; do
        python -m analysis_tools.replay_to_wandb \
            --jsonl "output/${s}/metrics.jsonl" \
            --project my-project \
            --group clean-replay \
            --strategy "$s" \
            --config "output/${s}/experiment_config.json"
    done
"""

import argparse
import json
from collections import OrderedDict


def replay(jsonl_path, project, group, strategy, config_path=None):
    """Read a JSONL file and log all metrics to a new W&B run."""
    import wandb

    # Load experiment config if provided
    run_config = {}
    if config_path:
        with open(config_path) as f:
            run_config = json.load(f)
    run_config['strategy'] = strategy

    # Init wandb run (console logging disabled)
    wandb.init(
        project=project,
        group=group,
        name=strategy,
        config=run_config,
        reinit=True,
        settings=wandb.Settings(console="off"),
    )

    # Pass 1: Read JSONL and accumulate metrics per step.
    # Multiple hook_points (POST_STEP, POST_EPOCH, SNAPSHOT) can emit
    # at the same step. Merging them avoids multiple wandb.log() calls
    # at the same step value.
    step_metrics = OrderedDict()  # step → {key: value, ...}
    n_records = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_records += 1

            step = record.pop('epoch', 0)
            record.pop('hook_point', None)

            if step not in step_metrics:
                step_metrics[step] = {}
            # Later records at the same step overwrite earlier ones
            # for the same key (matches original last-write-wins behavior)
            step_metrics[step].update(record)

    # Pass 2: Transform and log once per step
    n_logged = 0
    for step, metrics in step_metrics.items():
        logged = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Nested dict → flatten with /
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, bool)):
                        logged[f'{key}/{sub_key}'] = sub_value
            elif isinstance(value, (list, tuple)):
                # Accumulated step metrics → histogram + mean
                scalars = [v for v in value if isinstance(v, (int, float))]
                if scalars:
                    logged[key] = wandb.Histogram(scalars)
                    logged[f"{key}_mean"] = sum(scalars) / len(scalars)
            elif isinstance(value, (int, float, bool)):
                logged[key] = value

        if logged:
            wandb.log(logged, step=step)
            n_logged += 1

    wandb.finish()
    print(f"Done: {n_records} records → {n_logged} steps logged "
          f"→ {project}/{group}/{strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay JSONL metrics into Weights & Biases",
    )
    parser.add_argument('--jsonl', required=True,
                        help='Path to JSONL metrics file')
    parser.add_argument('--project', required=True,
                        help='W&B project name')
    parser.add_argument('--group', required=True,
                        help='W&B group name (shared across strategies)')
    parser.add_argument('--strategy', required=True,
                        help='Strategy / run name')
    parser.add_argument('--config', default=None,
                        help='Path to experiment_config.json (optional)')
    args = parser.parse_args()

    replay(args.jsonl, args.project, args.group, args.strategy, args.config)


if __name__ == '__main__':
    main()
