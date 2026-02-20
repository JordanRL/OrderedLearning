# Fourier

> Fourier structure emergence in embeddings, decoder, and MLP weights.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The Fourier hook analyzes the frequency-domain structure of learned representations, primarily in the embedding layer. For modular arithmetic, the theory predicts that networks learn to represent numbers using Fourier modes — periodic functions of the input tokens. This hook tracks the emergence of these Fourier modes during training.

It computes the DFT of the embedding matrix rows, measures spectral entropy (how concentrated the frequency spectrum is), identifies peak frequencies, tracks which frequencies cross a significance threshold over time, and specifically measures power at stride-harmonic frequencies (multiples of the stride value). It also analyzes the decoder weight matrix and MLP neuron weights for Fourier structure.

The hook maintains a set of "acquired frequencies" — frequencies that have exceeded a significance threshold — to track the progressive emergence of Fourier structure during training.

## Computational Cost

Low. Performs 1D DFTs on embedding vectors (length p, typically ~10K), which is fast. No additional forward or backward passes required. Operates directly on model weight tensors.

## Assumptions and Compatibility

- Designed for the modular arithmetic experiment's embedding structure
- Expects an embedding matrix where rows correspond to tokens 0..p-1
- Stride-harmonic metrics require a stride value in the run context
- Epoch-loop only

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `significance_multiplier` | `10.0` | A frequency is "significant" if its power exceeds this multiple of the mean spectral power |

## Metrics

### `low_freq_power`
- **Formula:** Fraction of total spectral power in the lowest 10% of frequency bins
- **Range:** [0, 1]
- **Interpretation:** High values indicate the embedding is dominated by low-frequency (smooth) structure. Low values indicate high-frequency or complex periodic structure.

### `spectral_entropy`
- **Formula:** Shannon entropy of the normalized power spectrum: `-Σ p_i log(p_i)`
- **Range:** [0, log(N)]
- **Interpretation:** Low entropy means power is concentrated in a few frequencies (structured representation). High entropy means power is spread across many frequencies (noise-like or complex representation).

### `peak_frequency`
- **Formula:** Frequency bin with the highest spectral power (excluding DC)
- **Range:** [1, p/2]
- **Interpretation:** The dominant periodic structure in the embedding. For stride-ordered training, this often corresponds to a stride harmonic.

### `peak_power`
- **Formula:** Spectral power at the peak frequency, normalized by total power
- **Range:** [0, 1]
- **Interpretation:** How dominant the peak frequency is. Values near 1 mean one frequency dominates the representation.

### `n_significant_freqs`
- **Formula:** Number of frequencies with power exceeding `significance_multiplier × mean_power`
- **Range:** [0, p/2]
- **Interpretation:** How many discrete frequencies the embedding has learned. Tracks the progressive acquisition of Fourier modes.

### `stride_harmonic_power`
- **Formula:** Total spectral power at stride harmonics (frequencies that are multiples of the stride) as a fraction of total power
- **Range:** [0, 1]
- **Interpretation:** How much of the embedding's frequency content aligns with the stride-imposed structure. Higher values indicate the data ordering is shaping the learned representation.

### `freq_powers`
- **Formula:** Power at each individual tracked frequency (logged as a list)
- **Interpretation:** Detailed per-frequency power values for post-hoc analysis.

### `n_tracked_freqs`
- **Formula:** Cumulative count of frequencies that have ever crossed the significance threshold
- **Range:** [0, p/2]
- **Interpretation:** Total number of Fourier modes acquired during training. Monotonically increasing.

### `newly_acquired_freqs`
- **Formula:** Number of new frequencies crossing the significance threshold at this epoch
- **Range:** [0, p/2]
- **Interpretation:** Rate of Fourier mode acquisition. Bursts of new frequencies may correspond to phase transitions.

### `decoder_spectral_entropy`
- **Formula:** Spectral entropy of the decoder (output projection) weight matrix
- **Range:** [0, log(N)]
- **Interpretation:** Whether the decoder has also developed Fourier structure, mirroring the embedding.

### `decoder_peak_frequency`
- **Formula:** Peak frequency in the decoder weight spectrum
- **Range:** [1, p/2]
- **Interpretation:** Dominant frequency in the decoder. Should align with embedding frequencies if the network has learned a consistent Fourier representation.

### `decoder_n_significant_freqs`
- **Formula:** Number of significant frequencies in the decoder spectrum
- **Range:** [0, p/2]
- **Interpretation:** Fourier mode count in the decoder, analogous to embedding `n_significant_freqs`.

### `neuron_fourier_top1`
- **Formula:** Median across MLP neurons of the fraction of spectral power in each neuron's top frequency
- **Range:** [0, 1]
- **Interpretation:** Whether individual MLP neurons are tuned to specific frequencies. High values indicate frequency-selective neurons.

### `neuron_fourier_entropy`
- **Formula:** Median spectral entropy across MLP neurons
- **Range:** [0, log(N)]
- **Interpretation:** Whether MLP neurons have concentrated or diffuse frequency content. Low entropy indicates frequency-selective neurons.
