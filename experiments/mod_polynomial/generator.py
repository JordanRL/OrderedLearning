"""Dataset generator for modular polynomial experiments.

Generates (x, y, c) tuples where c = (x^3 + x*y^2 + y) mod p.
"""

import random

from framework import DatasetGenerator


class ModPolynomialGenerator(DatasetGenerator):
    """Generates disjoint train/test datasets of (x, y, c) where c = (x^3 + xy^2 + y) mod p."""

    def __init__(self, console=None):
        self.console = console

    def _print(self, msg):
        if self.console is not None:
            self.console.print(msg)

    @staticmethod
    def polynomial(x, y, p):
        """Compute (x^3 + x*y^2 + y) mod p."""
        return (x * x * x + x * y * y + y) % p

    def generate(self, config, **kwargs):
        """Return (train_data_raw, test_data_raw) as lists of (x, y, c) tuples."""
        from rich.table import Table
        from rich import box

        p = config.p
        total_possible = p * p
        train_size = int(total_possible * config.train_fraction)
        test_size = total_possible - train_size

        self._print(f"[status]Generating disjoint datasets for x³ + xy² + y (mod {p})...[/status]")
        random.seed(config.seed)

        # Sample train pairs
        train_pairs = set()
        while len(train_pairs) < train_size:
            x = random.randint(0, p - 1)
            y = random.randint(0, p - 1)
            train_pairs.add((x, y))

        # All remaining pairs become test
        test_data = []
        for x in range(p):
            for y in range(p):
                if (x, y) not in train_pairs:
                    test_data.append((x, y, self.polynomial(x, y, p)))

        train_list = [(x, y, self.polynomial(x, y, p)) for x, y in train_pairs]

        # Display dataset summary
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
