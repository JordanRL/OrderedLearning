"""Fourier structure observer hook with per-frequency power tracking."""

import math

import torch
import numpy as np

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class FourierHook(TrainingHook):
    """Analyze Fourier structure emergence in embeddings, decoder, and MLP.

    Ported from analysis_tools/fourier.py — same computation, live data.
    Uses model parameters (not gradients).

    Tracks individual frequency powers over time: maintains a cumulative set
    of frequencies that have crossed the significance threshold. Reports power
    for all tracked frequencies each epoch, plus a list of newly acquired
    frequencies. This enables reconstruction of the frequency acquisition
    staircase at full resolution.

    Also analyzes:
    - Decoder weight matrix: DFT along output dimension to detect Fourier
      structure in the output mapping (does the model decode via Fourier basis?)
    - Neuron activations: composition of MLP linear1 weights with embedding
      to measure which Fourier components each neuron responds to.
    """

    name = "fourier"
    description = "Fourier structure emergence in embeddings, decoder, and MLP weights"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {'epoch': {HookPoint.POST_EPOCH}}
    needs_grads = False

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            # Embedding Fourier metrics
            MetricInfo('low_freq_power', 'Fraction of spectral power in lowest frequencies',
                       'sum(P[:cutoff]) / sum(P)', 'higher = more low-freq structure',
                       label='Low Freq Power'),
            MetricInfo('spectral_entropy', 'Normalized entropy of power spectrum',
                       'H(P) / log(p)', '0 = concentrated, 1 = uniform',
                       label='Embedding Spectral Entropy'),
            MetricInfo('peak_frequency', 'Dominant frequency index (excluding DC)', 'argmax(P[1:p/2]) + 1',
                       label='Peak Frequency'),
            MetricInfo('peak_power', 'Power at peak frequency', 'P[peak_freq]',
                       label='Peak Power'),
            MetricInfo('n_significant_freqs', 'Number of frequencies above significance threshold',
                       'count(P[k] > 10/p)',
                       label='Significant Freqs'),
            MetricInfo('stride_harmonic_power', 'Total power at stride harmonics',
                       'sum(P[k*stride]) for k=1..9', 'higher = clock-arithmetic structure',
                       label='Stride Harmonic Power'),
            MetricInfo('freq_powers', 'Power at each tracked frequency (dict)', 'P[k] for tracked k',
                       label='Freq Powers'),
            MetricInfo('n_tracked_freqs', 'Cumulative count of ever-significant frequencies',
                       label='Tracked Freqs'),
            MetricInfo('newly_acquired_freqs', 'Frequencies crossing threshold this epoch (list)',
                       label='New Freqs'),

            # Decoder Fourier metrics
            MetricInfo('decoder_spectral_entropy', 'Normalized entropy of decoder weight DFT spectrum',
                       'H(P_dec) / log(p)', '0 = concentrated, 1 = uniform',
                       label='Decoder Spectral Entropy'),
            MetricInfo('decoder_peak_frequency', 'Dominant frequency in decoder weights',
                       label='Decoder Peak Freq'),
            MetricInfo('decoder_n_significant_freqs', 'Significant frequencies in decoder weights',
                       label='Decoder Significant Freqs'),

            # Neuron Fourier metrics
            MetricInfo('neuron_fourier_top1', 'Mean top-1 frequency concentration across MLP neurons',
                       'mean(max(P_neuron) for each neuron)',
                       'higher = neurons are frequency-selective',
                       label='Neuron Freq Selectivity'),
            MetricInfo('neuron_fourier_entropy', 'Mean spectral entropy of MLP neuron responses',
                       'mean(H(P_neuron))',
                       'lower = neurons have sharper frequency tuning',
                       label='Neuron Spectral Entropy'),
        ]

    def __init__(self, significance_multiplier: float = 10.0):
        """
        Args:
            significance_multiplier: Multiplier over uniform baseline (1/p) to
                determine significance threshold. Default 10.0 matches the
                original analysis.
        """
        self._significance_multiplier = significance_multiplier
        self._tracked_freqs: set[int] = set()

    def compute(self, ctx) -> dict[str, float]:
        # Find embedding weights from live model
        emb = None
        decoder_w = None
        mlp_weights = []  # (weight, layer_idx) pairs

        for name, param in ctx.model.named_parameters():
            if 'embedding' in name.lower() and 'weight' in name.lower():
                emb = param.data
            elif 'decoder' in name.lower() and 'weight' in name.lower():
                decoder_w = param.data
            elif 'linear1' in name.lower() and 'weight' in name.lower():
                mlp_weights.append(param.data)

        if emb is None:
            return {}

        p = emb.shape[0]
        metrics = {}

        # === Embedding Fourier analysis ===
        metrics.update(self._analyze_embedding(emb, p))

        # === Decoder Fourier analysis ===
        if decoder_w is not None and decoder_w.shape[0] == p:
            metrics.update(self._analyze_decoder(decoder_w, p))

        # === Neuron Fourier analysis (MLP weights composed with embedding) ===
        if mlp_weights and emb is not None:
            metrics.update(self._analyze_neurons(mlp_weights, emb, p))

        return metrics

    def _analyze_embedding(self, emb, p) -> dict:
        """Standard embedding Fourier analysis."""
        # DFT along token dimension
        fft_result = torch.fft.fft(emb.float(), dim=0)
        power_spectrum = (fft_result.abs() ** 2).mean(dim=1)

        # Normalize
        total_power = power_spectrum.sum() + 1e-8
        power_spectrum_norm = power_spectrum / total_power

        # 1. Low frequency concentration
        cutoff = max(p // 20, 10)
        low_freq_power = power_spectrum_norm[:cutoff].sum().item()

        # 2. Spectral entropy
        entropy = -(power_spectrum_norm * torch.log(power_spectrum_norm + 1e-12)).sum().item()
        max_entropy = np.log(p)
        normalized_entropy = entropy / max_entropy

        # 3. Peak frequency (skip DC)
        peak_freq = power_spectrum_norm[1:p // 2].argmax().item() + 1
        peak_power = power_spectrum_norm[peak_freq].item()

        # 4. Number of significant frequencies
        threshold = 1.0 / p * self._significance_multiplier
        significant_mask = power_spectrum_norm > threshold
        n_significant = significant_mask.sum().item()

        # 5. Power in harmonics of sqrt(p)
        stride = int(math.sqrt(p))
        harmonic_indices = [k * stride for k in range(1, min(10, p // stride))]
        harmonic_power = sum(
            power_spectrum_norm[idx].item() for idx in harmonic_indices if idx < p
        )

        # 6. Per-frequency power tracking
        current_significant = set()
        for i in range(1, p // 2 + 1):
            if power_spectrum_norm[i].item() > threshold:
                current_significant.add(i)

        newly_acquired = sorted(current_significant - self._tracked_freqs)
        self._tracked_freqs |= current_significant

        freq_powers = {
            k: round(power_spectrum_norm[k].item(), 8)
            for k in sorted(self._tracked_freqs)
        }

        metrics = {
            'low_freq_power': low_freq_power,
            'spectral_entropy': normalized_entropy,
            'peak_frequency': peak_freq,
            'peak_power': peak_power,
            'n_significant_freqs': n_significant,
            'stride_harmonic_power': harmonic_power,
            'freq_powers': freq_powers,
            'n_tracked_freqs': len(self._tracked_freqs),
        }

        if newly_acquired:
            metrics['newly_acquired_freqs'] = newly_acquired

        return metrics

    def _analyze_decoder(self, decoder_w, p) -> dict:
        """Fourier analysis of decoder (output) weight matrix.

        The decoder maps d_model -> p. DFT along the output dimension
        (dim 0, corresponding to tokens 0..p-1) reveals whether the
        output layer uses Fourier structure.
        """
        # decoder_w shape: (p, d_model)
        fft_result = torch.fft.fft(decoder_w.float(), dim=0)
        power_spectrum = (fft_result.abs() ** 2).mean(dim=1)

        total_power = power_spectrum.sum() + 1e-8
        ps_norm = power_spectrum / total_power

        # Spectral entropy
        entropy = -(ps_norm * torch.log(ps_norm + 1e-12)).sum().item()
        max_entropy = np.log(p)

        # Peak frequency (skip DC)
        peak_freq = ps_norm[1:p // 2].argmax().item() + 1

        # Significant frequencies
        threshold = 1.0 / p * self._significance_multiplier
        n_significant = (ps_norm > threshold).sum().item()

        return {
            'decoder_spectral_entropy': entropy / max_entropy,
            'decoder_peak_frequency': peak_freq,
            'decoder_n_significant_freqs': n_significant,
        }

    def _analyze_neurons(self, mlp_weights, emb, p) -> dict:
        """Fourier analysis of MLP neurons via composition with embedding.

        Computes W_mlp @ W_embed^T to get a (d_ff, p) matrix showing each
        neuron's response to each token. DFT along the token dimension
        reveals which Fourier components each neuron is tuned to.
        """
        all_top1 = []
        all_entropy = []
        max_entropy = np.log(max(p // 2, 1))

        emb_float = emb.float()  # (p, d_model)

        for w in mlp_weights:
            # w shape: (d_ff, d_model)
            # Compose: neuron_responses = w @ emb^T -> (d_ff, p)
            neuron_responses = w.float() @ emb_float.T

            # DFT along token dimension (dim 1)
            fft_result = torch.fft.fft(neuron_responses, dim=1)
            # Power per neuron per frequency: (d_ff, p)
            power = fft_result.abs() ** 2

            # Normalize per neuron (along frequency dim), skip DC
            # Use only positive frequencies: 1..p//2
            power_pos = power[:, 1:p // 2 + 1]  # (d_ff, p//2)
            total_per_neuron = power_pos.sum(dim=1, keepdim=True) + 1e-8
            ps_norm = power_pos / total_per_neuron

            # Top-1 concentration per neuron
            top1 = ps_norm.max(dim=1).values  # (d_ff,)
            all_top1.append(top1.mean().item())

            # Spectral entropy per neuron
            entropy = -(ps_norm * torch.log(ps_norm + 1e-12)).sum(dim=1)  # (d_ff,)
            if max_entropy > 0:
                all_entropy.append((entropy / max_entropy).mean().item())

        metrics = {}
        if all_top1:
            metrics['neuron_fourier_top1'] = sum(all_top1) / len(all_top1)
        if all_entropy:
            metrics['neuron_fourier_entropy'] = sum(all_entropy) / len(all_entropy)

        return metrics

    def reset(self):
        self._tracked_freqs = set()
