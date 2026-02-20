"""Dataset generator for modular arithmetic experiments."""

import random

from framework import DatasetGenerator


class ModArithmeticGenerator(DatasetGenerator):
    """Generates disjoint train/test datasets of (a, b, c) where c = (a+b) mod p."""

    def __init__(self, console=None):
        self.console = console

    def _print(self, msg):
        if self.console is not None:
            self.console.print(msg)

    def generate(self, config, **kwargs):
        """Return (train_data_raw, test_data_raw) as lists of (a, b, c) tuples."""
        from rich.table import Table
        from rich import box

        p = config.p
        train_size = config.train_size
        test_size = config.test_size

        self._print(f"[status]Generating disjoint datasets...[/status]")
        random.seed(config.seed)

        train_pairs = set()
        while len(train_pairs) < train_size:
            a = random.randint(0, p - 1)
            b = random.randint(0, p - 1)
            train_pairs.add((a, b))

        test_data = []
        while len(test_data) < test_size:
            a = random.randint(0, p - 1)
            b = random.randint(0, p - 1)
            if (a, b) not in train_pairs:
                test_data.append((a, b, (a + b) % p))

        train_list = [(a, b, (a + b) % p) for a, b in train_pairs]

        # Display dataset summary
        total_possible = p * p
        train_pct = 100 * len(train_pairs) / total_possible
        test_pct = 100 * len(test_data) / total_possible

        data_table = Table(show_header=True, box=box.ROUNDED, title="Dataset Summary")
        data_table.add_column("Split", style="trigger")
        data_table.add_column("Pairs", justify="right", style="value.count")
        data_table.add_column("Coverage", justify="right", style="detail")
        data_table.add_row("Training", f"{len(train_pairs):,}", f"{train_pct:.2f}%")
        data_table.add_row("Test", f"{len(test_data):,}", f"{test_pct:.2f}%")
        data_table.add_row("Total Possible", f"{total_possible:,}", "100%", style="detail")
        self._print(data_table)
        if self.console is not None:
            self.console.print_complete("0% data leakage between splits")

        return train_list, test_data
