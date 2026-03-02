"""Resonant dataset builder for modular arithmetic experiments.

Co-designs dataset composition and ordering as a single object to maximize
coherent Hessian-gradient entanglement at the target Fourier frequency and
its harmonics. Each batch contains one complete stride residue class, with
b-values split into a coupling group (for cross-batch Hessian coupling) and
a harmonic group (with computed progressions that create specific output
spacings for target frequencies).

Batch ordering groups all cycles of the same residue class together:
  R_0(c0), R_0(c1), R_0(c2), R_1(c0), R_1(c1), R_1(c2), ...
This ensures consecutive batches share the same embedding rows (same
residue class, different b-values), creating within-class entanglement
that reinforces the stride-s pattern in the embedding table (frequency F).
If cycles were interleaved (R_0, R_1, R_2, ...), consecutive batches
would activate disjoint embedding rows and produce F=1 from the unit
shift between residue classes.
"""

import math
import random


class ResonantDatasetBuilder:
    """Builds a resonant dataset where composition and ordering are co-designed.

    The dataset is organized as sequential residue class batches. Within each
    batch, b-values are carefully chosen to produce specific output spacing
    patterns that reinforce the target Fourier frequency and its harmonics.

    Args:
        p: Prime modulus defining Z_p.
        stride: Stride value s. Residue classes are R_j = {a : a mod s = j}.
        n_harmonics: Number of harmonic frequencies to target (F, 2F, 4F, ...).
        overlap_frac: Fraction of each batch dedicated to coupling (shared b-values).
        n_cycles: Number of complete residue-class sweeps per epoch.
        seed: Random seed for deterministic b-value selection.
        console: Optional OLConsole for display output.
    """

    def __init__(self, p, stride, n_harmonics=4, overlap_frac=0.4,
                 n_cycles=3, seed=42, console=None):
        self.p = p
        self.stride = stride
        self.n_harmonics = n_harmonics
        self.overlap_frac = overlap_frac
        self.n_cycles = n_cycles
        self.seed = seed
        self.console = console

        if n_harmonics <= 0:
            raise ValueError(f"n_harmonics must be > 0, got {n_harmonics}")
        if not (0.0 <= overlap_frac <= 1.0):
            raise ValueError(f"overlap_frac must be in [0, 1], got {overlap_frac}")
        if n_cycles <= 0:
            raise ValueError(f"n_cycles must be > 0, got {n_cycles}")
        if n_harmonics > stride:
            raise ValueError(
                f"n_harmonics ({n_harmonics}) cannot exceed stride ({stride})"
            )

    def build(self):
        """Build the resonant training dataset.

        Returns:
            (train_data, batch_size) where:
            - train_data: list of (a, b, c) tuples in designed batch order
            - batch_size: int, the uniform batch size (max residue class size)
        """
        p = self.p
        s = self.stride
        F = round(p / s)

        # Build residue classes
        residue_classes = []
        for j in range(s):
            R_j = sorted([a for a in range(j, p, s)])
            residue_classes.append(R_j)

        max_class_size = max(len(R) for R in residue_classes)

        # Pad smaller classes to uniform size by duplicating last element
        for j in range(s):
            while len(residue_classes[j]) < max_class_size:
                residue_classes[j].append(residue_classes[j][-1])

        batch_size = max_class_size

        # Compute group sizes
        n_coupling = round(batch_size * self.overlap_frac)
        n_harmonic = batch_size - n_coupling

        # Compute harmonic targets
        harmonics = []
        freq = F
        for h in range(self.n_harmonics):
            # Fold frequency past Nyquist
            effective_freq = freq
            if effective_freq > p // 2:
                effective_freq = p - effective_freq

            d = round(p / effective_freq)
            b_step = (d - s) % p
            harmonics.append({
                'freq': freq,
                'effective_freq': effective_freq,
                'd': d,
                'b_step': b_step,
            })
            freq *= 2

        # Pre-generate coupling pools for each cycle
        coupling_pools = []
        for c in range(self.n_cycles):
            rng = random.Random(self.seed + c)
            coupling_pools.append(rng.sample(range(p), n_coupling))

        # Build dataset: group all cycles of the same residue class together.
        # Order: R_0(c0), R_0(c1), ..., R_0(cn), R_1(c0), R_1(c1), ...
        # This ensures consecutive batches share the same embedding rows,
        # creating within-class entanglement that reinforces the stride-s
        # pattern (frequency F) rather than the unit-shift pattern (F=1).
        train_data = []
        unique_pairs = set()

        for j in range(s):
            a_values = residue_classes[j]
            harmonic_idx = j % self.n_harmonics
            target = harmonics[harmonic_idx]

            for c in range(self.n_cycles):
                coupling_pool = coupling_pools[c]

                # Generate b_base for harmonic group, seeded per (j, c)
                batch_rng = random.Random(self.seed * 1000 + c * s + j)
                b_base = batch_rng.randint(0, p - 1)

                # Coupling group: first n_coupling elements
                for k in range(n_coupling):
                    a = a_values[k]
                    b = coupling_pool[k]
                    out = (a + b) % p
                    train_data.append((a, b, out))
                    unique_pairs.add((a, b))

                # Harmonic group: remaining elements
                for k in range(n_harmonic):
                    idx = n_coupling + k
                    a = a_values[idx]
                    b = (b_base + k * target['b_step']) % p
                    out = (a + b) % p
                    train_data.append((a, b, out))
                    unique_pairs.add((a, b))

        self._stats = {
            'p': p,
            'stride': s,
            'F': F,
            'n_residue_classes': s,
            'max_class_size': max_class_size,
            'batch_size': batch_size,
            'n_harmonics': self.n_harmonics,
            'harmonics': harmonics,
            'n_coupling': n_coupling,
            'n_harmonic': n_harmonic,
            'overlap_frac': self.overlap_frac,
            'n_cycles': self.n_cycles,
            'total_batches': s * self.n_cycles,
            'total_examples': len(train_data),
            'unique_pairs': len(unique_pairs),
            'coverage_pct': 100 * len(unique_pairs) / (p * p),
        }

        return train_data, batch_size

    def display_stats(self):
        """Display construction statistics using Rich tables."""
        if self.console is None:
            return

        stats = self._stats

        from rich.table import Table
        from rich import box

        # Main parameters table
        main = Table(
            show_header=False, box=box.ROUNDED,
            title="Resonant Dataset Construction",
        )
        main.add_column("Parameter", style="label")
        main.add_column("Value", justify="right", style="value.count")

        main.add_row("Modulus (p)", f"{stats['p']:,}")
        main.add_row("Stride (s)", str(stats['stride']))
        main.add_row("Target Frequency (F)", str(stats['F']))
        main.add_row("Residue Classes", str(stats['n_residue_classes']))
        main.add_row("Batch Size", str(stats['batch_size']))
        main.add_row("Coupling / Harmonic per Batch",
                      f"{stats['n_coupling']} / {stats['n_harmonic']}")
        main.add_row("Cycles per Epoch", str(stats['n_cycles']))
        within = stats['n_residue_classes'] * (stats['n_cycles'] - 1)
        between = stats['n_residue_classes'] - 1
        main.add_row("Within-class Transitions", str(within))
        main.add_row("Between-class Transitions", str(between))
        main.add_row("Total Batches", str(stats['total_batches']))
        main.add_row("Total Examples", f"{stats['total_examples']:,}")
        main.add_row("Unique (a,b) Pairs", f"{stats['unique_pairs']:,}")
        main.add_row("Coverage of p\u00b2",
                      f"{stats['coverage_pct']:.4f}%")

        self.console.print(main)

        # Harmonics table
        harm = Table(
            show_header=True, box=box.SIMPLE,
            title="Harmonic Targets",
        )
        harm.add_column("Harmonic", style="label")
        harm.add_column("Frequency", justify="right")
        harm.add_column("Output Spacing (d)", justify="right")
        harm.add_column("b-step", justify="right")
        harm.add_column("Batches/Cycle", justify="right")

        batches_per_harmonic = stats['n_residue_classes'] // stats['n_harmonics']
        remainder = stats['n_residue_classes'] % stats['n_harmonics']

        for i, h in enumerate(stats['harmonics']):
            label = f"F" if i == 0 else f"{2**i}F"
            count = batches_per_harmonic + (1 if i < remainder else 0)
            harm.add_row(
                label,
                str(h['freq']),
                str(h['d']),
                str(h['b_step']),
                str(count),
            )

        self.console.print(harm)
