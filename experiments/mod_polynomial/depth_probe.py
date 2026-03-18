"""Depth probe for modular polynomial experiments.

Trains multiple models at different depths concurrently on the same data,
printing side-by-side comparison tables at each eval point. Gives fast
signal on whether model depth affects learning before committing to full
experiment runs.

Usage:
    python -m experiments.mod_polynomial.depth_probe --layers 2 4 6 8
    python -m experiments.mod_polynomial.depth_probe --layers 2 4 6 8 --strategy stride --epochs 2000
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from rich.table import Table
from rich import box

from console import OLConsole

from experiments.mod_arithmetic.model import GrokkingTransformer
from experiments.mod_arithmetic.dataset import SparseModularDataset, GPUBatchIterator
from experiments.mod_arithmetic.loader import ModArithmeticLoader

from .config import ModPolynomialConfig
from .generator import ModPolynomialGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Depth probe for mod polynomial")
    parser.add_argument('--layers', type=int, nargs='+', required=True,
                        help="Layer counts to compare (e.g., 2 4 6 8)")
    parser.add_argument('--strategy', type=str, default='stride',
                        choices=['stride', 'target', 'random', 'fixed-random',
                                 'textbook', 'magnitude'],
                        help="Ordering strategy (default: stride)")
    parser.add_argument('--epochs', type=int, default=2000,
                        help="Number of epochs (default: 2000)")
    parser.add_argument('--eval-every', type=int, default=100,
                        help="Eval interval in epochs (default: 100)")
    parser.add_argument('--p', type=int, default=97, help="Prime modulus (default: 97)")
    parser.add_argument('--train-fraction', type=float, default=0.5,
                        help="Fraction of p^2 pairs for training (default: 0.5)")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument('--embed-dim', type=int, default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument('--num-heads', type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help="Weight decay (default: 0.1)")
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam'],
                        help="Optimizer type (default: adamw)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--stride', type=int, default=None,
                        help="Stride for stride ordering (default: floor(sqrt(p)))")
    return parser.parse_args()


def create_data(strategy, config, console):
    """Generate data and create train/test loaders for the given strategy."""
    generator = ModPolynomialGenerator(console=console)
    train_raw, test_raw = generator.generate(config)

    test_ds = SparseModularDataset(test_raw, mode='random', p=config.p)
    test_loader = GPUBatchIterator(test_ds, batch_size=config.batch_size * 8)

    if strategy == 'textbook':
        sorted_data = sorted(train_raw, key=lambda t: (t[0], t[1]))
        ds = SparseModularDataset(sorted_data, mode='preordered', p=config.p)
        train_loaders = [GPUBatchIterator(ds, batch_size=config.batch_size)]
    elif strategy == 'magnitude':
        sorted_data = sorted(train_raw, key=lambda t: (t[0] + t[1], t[0], t[1]))
        ds = SparseModularDataset(sorted_data, mode='preordered', p=config.p)
        train_loaders = [GPUBatchIterator(ds, batch_size=config.batch_size)]
    else:
        loader = ModArithmeticLoader(
            strategy=strategy, p=config.p,
            batch_size=config.batch_size, seed=config.seed,
            stride=config.stride,
        )
        train_loaders = loader.load(train_raw, config)

    return train_loaders, test_loader


def compute_accuracy(model, loader, device):
    """Compute classification accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch[:, :2])
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch[:, 2]).sum().item()
            total += batch.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def train_one_epoch(model, optimizer, criterion, loader, device):
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :2], batch[:, 2]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    args = parse_args()
    console = OLConsole()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModPolynomialConfig(
        p=args.p, train_fraction=args.train_fraction,
        batch_size=args.batch_size, lr=args.lr,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        weight_decay=args.weight_decay, optimizer=args.optimizer,
        seed=args.seed, stride=args.stride,
        epochs=args.epochs, eval_every=args.eval_every,
    )

    console.rule(f"[bold]Depth Probe: x³ + xy² + y (mod {config.p})[/bold]")
    console.print(f"[label]Strategy:[/label] {args.strategy}")
    console.print(f"[label]Depths:[/label] {args.layers}")
    console.print(f"[label]Epochs:[/label] {args.epochs}")
    console.print(f"[label]Device:[/label] {device}")
    console.print()

    # Generate data once — shared across all depths
    train_loaders, test_loader = create_data(args.strategy, config, console)
    console.print()

    # Create one model + optimizer per depth
    chance = 100.0 / config.p
    models = {}
    optimizers = {}
    criterion = nn.CrossEntropyLoss()

    for n_layers in args.layers:
        model = GrokkingTransformer(
            config.p, config.embed_dim, config.num_heads, n_layers,
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        console.print(
            f"[label]{n_layers}L:[/label] [value.count]{param_count:,}[/value.count] params"
        )

        optimizer_cls = optim.Adam if config.optimizer == 'adam' else optim.AdamW
        opt = optimizer_cls(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

        models[n_layers] = model
        optimizers[n_layers] = opt

    total_params = sum(
        sum(p.numel() for p in m.parameters()) for m in models.values()
    )
    total_mb = total_params * 4 / (1024 * 1024)  # float32
    console.print(
        f"\n[label]Total model memory:[/label] [value.count]{total_mb:.1f} MB[/value.count] "
        f"(×3 with Adam state ≈ [value.count]{total_mb * 3:.1f} MB[/value.count])"
    )
    console.print()

    # Track best val accuracy per depth
    best_val = {n: 0.0 for n in args.layers}
    start_time = time.time()
    depths_str = ", ".join(f"{n}L" for n in args.layers)

    from framework import display

    display.epoch_progress_start(f"depth probe [{depths_str}]", args.epochs)

    for epoch in range(1, args.epochs + 1):
        loader = train_loaders[epoch % len(train_loaders)]
        if hasattr(loader, 'seed_epoch'):
            loader.seed_epoch(config.seed + epoch)

        # Train all depths on the same epoch's data
        losses = {}
        for n_layers in args.layers:
            losses[n_layers] = train_one_epoch(
                models[n_layers], optimizers[n_layers],
                criterion, loader, device,
            )

        # Update progress bar
        avg_loss = sum(losses.values()) / len(losses)
        best_so_far = max(best_val.values()) if any(best_val.values()) else 0.0
        display.epoch_progress_update(epoch, args.epochs, avg_loss)

        # Eval at intervals
        if epoch % args.eval_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            eps = epoch / elapsed if elapsed > 0 else 0

            table = Table(
                box=box.ROUNDED, show_header=True,
                header_style="table.header",
                title=f"[bold]Epoch {epoch:,}[/bold] [detail]| {args.strategy} | "
                      f"{elapsed:.0f}s ({eps:.0f} ep/s)[/detail]",
                title_style="",
            )
            table.add_column("Depth", justify="center")
            table.add_column("Loss", justify="right")
            table.add_column("Train Acc", justify="right")
            table.add_column("Val Acc", justify="right")
            table.add_column("vs Chance", justify="right")
            table.add_column("Best Val", justify="right")

            for n_layers in args.layers:
                model = models[n_layers]
                train_acc = compute_accuracy(model, loader, device)
                val_acc = compute_accuracy(model, test_loader, device)
                best_val[n_layers] = max(best_val[n_layers], val_acc)

                diff = val_acc - chance
                if diff > 1.0:
                    diff_str = f"[metric.improved]+{diff:.2f}pp[/metric.improved]"
                elif diff > 0:
                    diff_str = f"[detail]+{diff:.2f}pp[/detail]"
                else:
                    diff_str = f"[metric.degraded]{diff:+.2f}pp[/metric.degraded]"

                best_diff = best_val[n_layers] - chance
                if best_diff > 1.0:
                    best_str = f"[metric.improved]{best_val[n_layers]:.2f}%[/metric.improved]"
                else:
                    best_str = f"{best_val[n_layers]:.2f}%"

                table.add_row(
                    f"[bold]{n_layers}L[/bold]",
                    f"[metric.value]{losses[n_layers]:.4f}[/metric.value]",
                    f"{train_acc:.2f}%",
                    f"{val_acc:.2f}%",
                    diff_str,
                    best_str,
                )

            console.print(table)

    display.epoch_progress_end()

    # Final summary
    console.rule("[bold]Depth Probe Summary[/bold]")
    summary = Table(box=box.ROUNDED, show_header=True, title="Final Results")
    summary.add_column("Depth", justify="center")
    summary.add_column("Final Loss", justify="right")
    summary.add_column("Best Val Acc", justify="right")
    summary.add_column("vs Chance", justify="right")

    for n_layers in args.layers:
        diff = best_val[n_layers] - chance
        if diff > 1.0:
            diff_str = f"[metric.improved]+{diff:.2f}pp[/metric.improved]"
        elif diff > 0:
            diff_str = f"[detail]+{diff:.2f}pp[/detail]"
        else:
            diff_str = f"[metric.degraded]{diff:+.2f}pp[/metric.degraded]"

        summary.add_row(
            f"[bold]{n_layers}L[/bold]",
            f"[metric.value]{losses[n_layers]:.4f}[/metric.value]",
            f"{best_val[n_layers]:.2f}%",
            diff_str,
        )

    console.print(summary)
    elapsed = time.time() - start_time
    console.print(f"\n[label]Total time:[/label] {elapsed:.1f}s")


if __name__ == '__main__':
    main()
