"""DFT analysis of mod_arithmetic dataset orderings.

Examines the spectral properties of (a, b, c) sequences under different
data orderings to identify structure that might influence training dynamics.

Generates separate analysis figures for each component (input a, input b,
target c). The Z/pZ modular spectrum (row 6) uses the same frequency
indices as the fourier hook, enabling direct comparison.

Usage:
    python -m analysis_tools.dataset_dft_analysis [--p 9973] [--train-size 300000] [--seed 42]
"""

import argparse
import math
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_dataset(p, train_size, seed):
    """Generate (a, b, c) triples — mirrors ModArithmeticGenerator."""
    random.seed(seed)
    train_pairs = set()
    while len(train_pairs) < train_size:
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)
        train_pairs.add((a, b))
    return [(a, b, (a + b) % p) for a, b in train_pairs]


def apply_ordering(data, mode, p, stride=None):
    """Apply ordering — mirrors SparseModularDataset."""
    if mode == 'stride':
        stride = stride or int(math.sqrt(p))
        return sorted(data, key=lambda x: (x[0] % stride, x[0]))
    elif mode == 'target':
        return sorted(data, key=lambda x: x[2])
    else:
        return data  # random = generation order


def compute_power_spectrum(seq, norm=True):
    """Compute one-sided power spectrum of a real sequence."""
    n = len(seq)
    fft = np.fft.rfft(seq - np.mean(seq))
    ps = np.abs(fft) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    if norm:
        total = ps.sum()
        if total > 0:
            ps = ps / total
    return freqs, ps


def compute_autocorrelation(seq, max_lag=500):
    """Compute normalized autocorrelation for lags 1..max_lag."""
    seq = np.array(seq, dtype=np.float64)
    seq = seq - seq.mean()
    var = np.var(seq)
    if var < 1e-12:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([
        np.mean(seq[:-lag] * seq[lag:]) / var for lag in lags
    ])
    return lags, acf


def batch_entropy(values, batch_size, p):
    """Measure how uniformly values are distributed within each batch."""
    n_batches = len(values) // batch_size
    entropies = []
    max_entropy = np.log(min(batch_size, p))

    for i in range(n_batches):
        batch = values[i * batch_size:(i + 1) * batch_size]
        _, counts = np.unique(batch, return_counts=True)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs))
        entropies.append(ent / max_entropy)  # normalized [0, 1]

    return np.array(entropies)


def batch_coverage(values, batch_size, p):
    """Fraction of p covered by unique values in each batch."""
    n_batches = len(values) // batch_size
    coverages = []
    for i in range(n_batches):
        batch = values[i * batch_size:(i + 1) * batch_size]
        coverages.append(len(np.unique(batch)) / p)
    return np.array(coverages)


def consecutive_differences(values, p):
    """Compute consecutive differences (mod p, signed)."""
    diffs = np.diff(values) % p
    diffs = np.where(diffs > p // 2, diffs - p, diffs)
    return diffs


def compute_batch_modular_spectrum(values, batch_size, p, max_freq=500):
    """Compute mean per-batch power spectrum over Z/pZ Fourier modes.

    For each batch of consecutive samples, builds a histogram of values
    (length p), takes its DFT, and computes |X_k|²/batch_size.
    Averages over all batches.

    Frequency k here is the same as in the fourier hook: it means
    k oscillations around the cyclic group Z/pZ, dividing the circle
    into p/k buckets. Directly comparable to fourier hook output.

    Returns:
        modular_freqs: array of frequency indices 1..max_k
        mean_power: mean |X_k|²/B per batch (≈1.0 for uniform random)
    """
    n_batches = len(values) // batch_size
    max_k = min(p // 2, max_freq)

    power_accum = np.zeros(max_k)

    for i in range(n_batches):
        batch = values[i * batch_size:(i + 1) * batch_size]
        hist = np.bincount(batch, minlength=p).astype(np.float64)
        X = np.fft.fft(hist)
        power_accum += np.abs(X[1:max_k + 1]) ** 2 / batch_size

    mean_power = power_accum / n_batches
    modular_freqs = np.arange(1, max_k + 1)
    return modular_freqs, mean_power


def plot_component_analysis(orderings, colors, component_idx, component_label,
                            batch_size, p, stride, output_path):
    """Generate a 6-row analysis figure for one data component."""
    n_samples = len(next(iter(orderings.values())))

    fig = plt.figure(figsize=(26, 24))
    gs = GridSpec(6, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f'DFT Analysis: {component_label}\n'
        f'p={p}, N={n_samples:,}, batch_size={batch_size}, stride={stride}\n'
        f'fixed-random = set iteration order, shuffled = true uniform permutation\n'
        f'Row 6: Z/pZ modular spectrum — same frequency indices as fourier hook',
        fontsize=14, y=0.99,
    )

    for col, (name, data) in enumerate(orderings.items()):
        values = np.array([x[component_idx] for x in data])

        # ── Row 1: Sequential power spectrum ──
        ax = fig.add_subplot(gs[0, col])
        freqs, ps = compute_power_spectrum(values, norm=True)
        ax.semilogy(freqs[1:], ps[1:], color=colors[name], alpha=0.6,
                     linewidth=0.3)
        window = min(100, len(ps) // 10)
        if window > 1:
            smoothed = np.convolve(ps[1:], np.ones(window) / window,
                                   mode='valid')
            ax.semilogy(freqs[1:len(smoothed) + 1], smoothed,
                        color=colors[name], linewidth=2,
                        label=f'smoothed (w={window})')
        ax.set_title(f'{name}: {component_label} power spectrum')
        ax.set_xlabel('Normalized frequency')
        ax.set_ylabel('Power (normalized)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Row 2: Autocorrelation ──
        ax = fig.add_subplot(gs[1, col])
        max_lag = min(2000, len(values) // 10)
        lags, acf = compute_autocorrelation(values, max_lag=max_lag)
        ax.plot(lags, acf, color=colors[name], linewidth=0.5)
        for mult in range(1, 6):
            bl = batch_size * mult
            if bl < max_lag:
                ax.axvline(bl, color='gray', linestyle='--', alpha=0.3,
                           linewidth=0.5)
        sig = 2 / np.sqrt(len(values))
        ax.axhline(sig, color='black', linestyle=':', alpha=0.4,
                    label=f'2/\u221aN={sig:.4f}')
        ax.axhline(-sig, color='black', linestyle=':', alpha=0.4)
        ax.axhline(0, color='black', alpha=0.2)
        ax.set_title(f'{name}: {component_label} autocorrelation')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Row 3: Consecutive differences histogram ──
        ax = fig.add_subplot(gs[2, col])
        diffs = consecutive_differences(values, p)
        bins = min(200, p // 10)
        ax.hist(diffs, bins=bins, color=colors[name], alpha=0.7, density=True)
        ax.set_title(f'{name}: consecutive \u0394{component_label} distribution')
        ax.set_xlabel('\u0394 (mod p, signed)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.95,
                f'mean={np.mean(diffs):.1f}\nstd={np.std(diffs):.1f}\n'
                f'|\u0394|<{p//20}: {np.mean(np.abs(diffs) < p//20)*100:.1f}%',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ── Row 4: Per-batch entropy ──
        ax = fig.add_subplot(gs[3, col])
        entropies = batch_entropy(values, batch_size, p)
        ax.plot(entropies, color=colors[name], linewidth=0.5, alpha=0.7)
        ax.axhline(np.mean(entropies), color='black', linestyle='--',
                    label=f'mean={np.mean(entropies):.4f}')
        ax.set_title(f'{name}: per-batch {component_label} entropy (norm.)')
        ax.set_xlabel('Batch index')
        ax.set_ylabel('Entropy / max_entropy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Row 5: Per-batch coverage ──
        ax = fig.add_subplot(gs[4, col])
        cov = batch_coverage(values, batch_size, p)
        ax.plot(cov, color=colors[name], linewidth=0.5, alpha=0.7)
        ax.axhline(np.mean(cov), color='black', linestyle='--',
                    label=f'mean={np.mean(cov):.4f}')
        expected = (1 - (1 - 1/p) ** batch_size)
        ax.axhline(expected, color='gray', linestyle=':',
                    label=f'expected uniform={expected:.4f}')
        ax.set_title(f'{name}: per-batch {component_label} coverage')
        ax.set_xlabel('Batch index')
        ax.set_ylabel('Unique / p')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Row 6: Per-batch Z/pZ modular spectrum ──
        ax = fig.add_subplot(gs[5, col])
        mod_freqs, mod_power = compute_batch_modular_spectrum(
            values, batch_size, p,
        )
        ax.semilogy(mod_freqs, mod_power, color=colors[name], alpha=0.6,
                     linewidth=0.3)
        smooth_w = 20
        if len(mod_power) > smooth_w:
            smoothed = np.convolve(
                mod_power, np.ones(smooth_w) / smooth_w, mode='valid',
            )
            ax.semilogy(mod_freqs[:len(smoothed)], smoothed,
                        color=colors[name], linewidth=2,
                        label=f'smoothed (w={smooth_w})')
        ax.axhline(1.0, color='black', linestyle=':', alpha=0.4,
                    label='random baseline')
        for h in range(1, 6):
            hf = stride * h
            if hf <= mod_freqs[-1]:
                ax.axvline(hf, color='gray', linestyle='--', alpha=0.3,
                           linewidth=0.5)
        ax.text(stride, ax.get_ylim()[1], f' s={stride}', fontsize=6,
                va='top', ha='left', color='gray')
        top5 = np.argsort(mod_power)[-5:]
        for pi in top5:
            if mod_power[pi] > np.median(mod_power) * 10:
                k = mod_freqs[pi]
                buckets = p / k
                ax.annotate(
                    f'k={k} ({buckets:.0f} buckets)',
                    (k, mod_power[pi]), fontsize=6,
                    textcoords='offset points', xytext=(5, 5),
                )
        ax.set_title(f'{name}: per-batch Z/pZ spectrum ({component_label})')
        ax.set_xlabel('Modular frequency k  (k osc. around Z/pZ)')
        ax.set_ylabel('|X_k|\u00b2/B  (random \u2248 1.0)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


def print_component_summary(orderings, component_idx, component_label,
                            batch_size, p, stride):
    """Print summary statistics for one component across all orderings."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {component_label}")
    print(f"{'=' * 70}")

    for name, data in orderings.items():
        values = np.array([x[component_idx] for x in data])
        diffs = consecutive_differences(values, p)
        freqs, ps = compute_power_spectrum(values, norm=True)

        # Spectral flatness
        ps_nz = ps[1:][ps[1:] > 0]
        spectral_flatness = np.exp(np.mean(np.log(ps_nz))) / np.mean(ps_nz)

        # Peak frequency
        peak_bin = np.argmax(ps[1:]) + 1
        peak_freq = freqs[peak_bin]
        peak_period = 1 / peak_freq if peak_freq > 0 else float('inf')

        # Batch entropy
        entropies = batch_entropy(values, batch_size, p)

        print(f"\n  {name.upper()}")
        print(f"    {component_label}: mean={values.mean():.1f}, std={values.std():.1f}")
        print(f"    Consecutive \u0394: mean|\u0394|={np.mean(np.abs(diffs)):.1f}, "
              f"|\u0394|<{p//20}: {np.mean(np.abs(diffs) < p//20)*100:.1f}%")
        print(f"    Spectral flatness: {spectral_flatness:.6f} (1.0 = white noise)")
        print(f"    Peak seq. frequency: f={peak_freq:.6f} "
              f"(period={peak_period:.0f} samples)")
        print(f"    Batch entropy: mean={entropies.mean():.4f}, "
              f"std={entropies.std():.4f}, min={entropies.min():.4f}")
        lags, acf = compute_autocorrelation(values, max_lag=500)
        print(f"    ACF lag-1={acf[0]:.6f}, "
              f"lag-{batch_size}={acf[batch_size-1]:.6f}")

        # Z/pZ modular spectrum
        mod_freqs, mod_power = compute_batch_modular_spectrum(
            values, batch_size, p,
        )
        low_k = min(stride, len(mod_power))
        print(f"    Z/pZ modular spectrum (per-batch):")
        print(f"      mean power k<{low_k}: {mod_power[:low_k].mean():.2f}  "
              f"(random \u2248 1.0)")
        print(f"      mean power k>{low_k}: {mod_power[low_k:].mean():.2f}")
        harmonic_ks = [stride * h for h in range(1, 6)
                       if stride * h <= len(mod_power)]
        if harmonic_ks:
            h_power = [mod_power[k - 1] for k in harmonic_ks]
            print(f"      stride harmonics (k={harmonic_ks}): "
                  f"power={[f'{v:.2f}' for v in h_power]}")
        top_k_idx = np.argsort(mod_power)[-5:][::-1]
        top_info = [(mod_freqs[i], mod_power[i], p / mod_freqs[i])
                    for i in top_k_idx]
        print(f"      top 5 peaks: "
              + ", ".join(f"k={k}({bkt:.0f}bkt,P={pw:.1f})"
                          for k, pw, bkt in top_info))


def main():
    parser = argparse.ArgumentParser(
        description='DFT analysis of mod_arithmetic dataset orderings',
    )
    parser.add_argument('--p', type=int, default=9973)
    parser.add_argument('--train-size', type=int, default=300000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--output', type=str, default='dataset_dft_analysis.png')
    args = parser.parse_args()

    p = args.p
    stride = args.stride or int(math.sqrt(p))
    print(f"Generating dataset: p={p}, train_size={args.train_size}, seed={args.seed}")
    print(f"Stride value: {stride}")
    print(f"Total possible pairs: {p*p:,} ({args.train_size/(p*p)*100:.1f}% coverage)")

    raw_data = generate_dataset(p, args.train_size, args.seed)

    # True uniform shuffle for comparison baseline
    shuffled_data = list(raw_data)
    rng = random.Random(args.seed + 1)  # different seed to avoid correlation
    rng.shuffle(shuffled_data)

    orderings = {
        'fixed-random': apply_ordering(raw_data, 'random', p),
        'shuffled': shuffled_data,
        'stride': apply_ordering(raw_data, 'stride', p, stride=args.stride),
        'target': apply_ordering(raw_data, 'target', p),
    }
    colors = {
        'fixed-random': '#e74c3c', 'shuffled': '#9b59b6',
        'stride': '#3498db', 'target': '#2ecc71',
    }

    # Generate separate figures for each component
    base, ext = os.path.splitext(args.output)
    components = [
        (2, 'target (c)', f'{base}_targets{ext}'),
        (0, 'input (a)', f'{base}_input_a{ext}'),
        (1, 'input (b)', f'{base}_input_b{ext}'),
    ]

    print(f"\nGenerating {len(components)} analysis figures...")
    for comp_idx, comp_label, comp_output in components:
        plot_component_analysis(
            orderings, colors, comp_idx, comp_label,
            args.batch_size, p, stride, comp_output,
        )

    # Print summary statistics for each component
    for comp_idx, comp_label, _ in components:
        print_component_summary(
            orderings, comp_idx, comp_label,
            args.batch_size, p, stride,
        )


if __name__ == '__main__':
    main()
