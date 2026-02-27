"""Predictive coding model hierarchy.

Implements a hierarchical predictive coding network where each layer
predicts the layer below via top-down generative weights. Learning is
driven by local prediction errors — no global backpropagation.

Architecture:
    PredictiveCodingNetwork
    └── layers: nn.ModuleList[PCLayer]
        ├── generative: nn.Linear (top-down prediction, always present)
        ├── recognition: nn.Linear (bottom-up, optional non-symmetric)
        ├── lateral: nn.Linear (within-layer recurrence, optional)
        └── log_precision: nn.Parameter (learnable error weighting, optional)

The model exposes a settling API for iterative inference:
    1. initialize_activations(x) — forward-pass initialization
    2. inference_step(activations, clamped, lr) — one settling iteration
    3. compute_weight_gradients(activations) — local Hebbian weight updates

Experiments clamp observations at arbitrary layers via dict[int, Tensor].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

def _tanh_deriv(output: torch.Tensor) -> torch.Tensor:
    """Derivative of tanh given the *output* (not pre-activation)."""
    return 1.0 - output ** 2


def _relu_deriv(output: torch.Tensor) -> torch.Tensor:
    """Derivative of relu given the output."""
    return (output > 0).float()


def _linear_deriv(output: torch.Tensor) -> torch.Tensor:
    """Derivative of identity activation."""
    return torch.ones_like(output)


_ACTIVATIONS: dict[str, tuple[Any, Any]] = {
    'tanh': (torch.tanh, _tanh_deriv),
    'relu': (F.relu, _relu_deriv),
    'linear': (lambda x: x, _linear_deriv),
}


# ---------------------------------------------------------------------------
# PCLayerConfig
# ---------------------------------------------------------------------------

@dataclass
class PCLayerConfig:
    """Configuration for a single predictive coding layer.

    Each layer defines a generative mapping from the layer above to this
    layer's representation space.

    Args:
        input_size: Dimensionality of this layer (the layer being predicted).
        output_size: Dimensionality of the layer above (the predictor).
        activation: Activation function name ('tanh', 'relu', 'linear').
        symmetric: If True, bottom-up feedback uses the transpose of the
            generative weights. If False, a separate recognition linear is
            created for bottom-up signals.
        lateral: If True, include lateral (within-layer) connections.
        learnable_precision: If True, include a per-unit log-precision
            parameter that weights prediction errors.
    """
    input_size: int
    output_size: int
    activation: str = 'tanh'
    symmetric: bool = True
    lateral: bool = False
    learnable_precision: bool = False


# ---------------------------------------------------------------------------
# PCLayer
# ---------------------------------------------------------------------------

class PCLayer(nn.Module):
    """Single layer in a predictive coding hierarchy.

    Implements the generative model for one level: given activity in the
    layer above, predict activity at this level. Optionally includes
    non-symmetric recognition weights, lateral connections, and learnable
    precision.
    """

    def __init__(self, config: PCLayerConfig) -> None:
        super().__init__()
        self.config = config

        # Top-down generative weights: maps from layer above -> this layer
        self.generative = nn.Linear(config.output_size, config.input_size)

        # Optional non-symmetric recognition weights (bottom-up feedback)
        # No bias — only generative weights use bias in standard PC
        self.recognition: nn.Linear | None = None
        if not config.symmetric:
            self.recognition = nn.Linear(config.input_size, config.output_size, bias=False)

        # Optional lateral connections (within-layer)
        # No bias — lateral dynamics are zero-centered
        self.lateral_conn: nn.Linear | None = None
        if config.lateral:
            self.lateral_conn = nn.Linear(config.input_size, config.input_size, bias=False)

        # Optional learnable precision (log-space for positivity)
        self.log_precision: nn.Parameter | None = None
        if config.learnable_precision:
            self.log_precision = nn.Parameter(torch.zeros(config.input_size))

        # Resolve activation function and its derivative
        if config.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{config.activation}'. "
                f"Available: {list(_ACTIVATIONS.keys())}"
            )
        self._activation_fn, self._activation_deriv_fn = _ACTIVATIONS[config.activation]

    @property
    def precision(self) -> torch.Tensor | None:
        """Precision weights (exp of log_precision), or None."""
        if self.log_precision is not None:
            return torch.exp(self.log_precision)
        return None

    def predict(self, activity_above: torch.Tensor) -> torch.Tensor:
        """Top-down prediction: x_hat = f(W @ x_above).

        Args:
            activity_above: Activation at the layer above, shape (batch, output_size).

        Returns:
            Predicted activation at this layer, shape (batch, input_size).
        """
        return self._activation_fn(self.generative(activity_above))

    def prediction_error(
        self,
        activity: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Prediction error, optionally precision-weighted.

        Args:
            activity: Actual activation at this layer, shape (batch, input_size).
            prediction: Top-down prediction, shape (batch, input_size).

        Returns:
            Error tensor, shape (batch, input_size). If learnable precision
            is enabled, the error is scaled by the precision.
        """
        error = activity - prediction
        precision = self.precision
        if precision is not None:
            error = precision * error
        return error

    def activation_derivative(self, output: torch.Tensor) -> torch.Tensor:
        """Derivative of the activation function given the *output* value.

        For tanh: f'(a) = 1 - tanh(a)^2 = 1 - output^2
        For relu: f'(a) = (output > 0).float()
        For linear: f'(a) = 1

        Args:
            output: Post-activation values, shape (batch, input_size).

        Returns:
            Derivative values, same shape.
        """
        return self._activation_deriv_fn(output)

    def feedback(self, error: torch.Tensor) -> torch.Tensor:
        """Bottom-up feedback from this layer's error to the layer above.

        Uses recognition weights if non-symmetric, otherwise the transpose
        of the generative weights.

        Args:
            error: Prediction error at this layer, shape (batch, input_size).

        Returns:
            Feedback signal to the layer above, shape (batch, output_size).
        """
        if self.recognition is not None:
            return self.recognition(error)
        # Symmetric: use transpose of generative weights
        return F.linear(error, self.generative.weight.t())


# ---------------------------------------------------------------------------
# PredictiveCodingNetwork
# ---------------------------------------------------------------------------

class PredictiveCodingNetwork(nn.Module):
    """Hierarchical predictive coding network.

    A stack of PCLayer modules forming a generative hierarchy. Predictions
    flow top-down; prediction errors drive both inference (settling) and
    learning (local weight updates).

    Layer indexing convention:
        Layer 0 = bottom (input/observation space)
        Layer L = top (highest-level representation)

    Layers[i] maps from layer i+1 -> layer i (top-down).
    So len(layers) == len(layer_sizes) - 1.

    Args:
        layer_sizes: List of layer dimensions [input_dim, hidden1, ..., top_dim].
            Must have at least 2 elements.
        layer_configs: Optional per-layer configs. If None, all layers use
            defaults from default_* arguments.
        default_activation: Default activation for all layers.
        default_symmetric: Default symmetric setting for all layers.
        default_lateral: Default lateral setting for all layers.
        default_learnable_precision: Default precision setting for all layers.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        layer_configs: list[PCLayerConfig] | None = None,
        default_activation: str = 'tanh',
        default_symmetric: bool = True,
        default_lateral: bool = False,
        default_learnable_precision: bool = False,
    ) -> None:
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("Need at least 2 layer sizes (input + 1 hidden).")

        self._layer_sizes = list(layer_sizes)

        # Build layer configs if not provided
        if layer_configs is not None:
            if len(layer_configs) != len(layer_sizes) - 1:
                raise ValueError(
                    f"Expected {len(layer_sizes) - 1} layer configs, "
                    f"got {len(layer_configs)}."
                )
            configs = layer_configs
        else:
            configs = [
                PCLayerConfig(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    activation=default_activation,
                    symmetric=default_symmetric,
                    lateral=default_lateral,
                    learnable_precision=default_learnable_precision,
                )
                for i in range(len(layer_sizes) - 1)
            ]

        self.layers = nn.ModuleList([PCLayer(cfg) for cfg in configs])

    @property
    def num_layers(self) -> int:
        """Number of PC layers (generative mappings, not including input)."""
        return len(self.layers)

    @property
    def layer_sizes(self) -> list[int]:
        """Dimensions of all layers including the input layer."""
        return list(self._layer_sizes)

    def initialize_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Initialize latent activations via a bottom-up forward pass.

        Uses the recognition path (or generative transpose if symmetric)
        to propagate the input upward through the hierarchy.

        Args:
            x: Input observation, shape (batch, layer_sizes[0]).

        Returns:
            List of activation tensors [x_0, x_1, ..., x_L] where x_0 = x.
            All tensors are detached.
        """
        activations = [x.detach().clone()]
        current = x.detach()
        for layer in self.layers:
            # Bottom-up: use recognition weights or generative transpose
            current = layer.feedback(current)
            current = layer._activation_fn(current)
            activations.append(current.clone())
        return activations

    def inference_step(
        self,
        activations: list[torch.Tensor],
        clamped: dict[int, torch.Tensor],
        inference_lr: float = 0.1,
    ) -> list[torch.Tensor]:
        """One step of iterative inference (settling dynamics).

        Updates free (unclamped) layer activations to reduce variational
        free energy. Clamped layers are held fixed.

        Update rule for free layer l:
            e[l] = x[l] - predict_l(x[l+1])
            feedback_below = layers[l-1].feedback(f'(predict_pre[l-1]) * e[l-1])
            x[l] -= inference_lr * (e[l] - feedback_below)

        This is the gradient of free energy w.r.t. x[l], driving activations
        toward an equilibrium that minimizes prediction errors.

        Args:
            activations: Current activations [x_0, ..., x_L], modified in-place.
            clamped: {layer_index: tensor} for layers held fixed.
            inference_lr: Step size for activation updates.

        Returns:
            The same activations list (modified in-place).
        """
        n_levels = len(activations)  # L + 1 (including input)

        # Compute predictions and errors at each layer
        predictions = []
        errors = []
        for i, layer in enumerate(self.layers):
            pred = layer.predict(activations[i + 1])
            err = layer.prediction_error(activations[i], pred)
            predictions.append(pred)
            errors.append(err)

        # Compute activation updates for all free layers, then apply
        # (synchronous update: compute all deltas first, then apply)
        deltas = {}
        for l in range(n_levels):
            if l in clamped:
                continue

            delta = torch.zeros_like(activations[l])

            # Term 1: prediction error at this layer (pushes toward prediction from above)
            # Only layers 0..L-1 have predictions from above
            if l < len(self.layers):
                delta = delta + errors[l]

            # Term 2: feedback from the layer below (propagated error)
            # Layer l receives feedback from errors[l-1] (error at layer l-1)
            # weighted by the derivative of the activation at the prediction
            if l > 0 and (l - 1) < len(self.layers):
                layer_below = self.layers[l - 1]
                deriv = layer_below.activation_derivative(predictions[l - 1])
                weighted_error = deriv * errors[l - 1]
                # Feedback projects from layer l-1 space to layer l space
                feedback_signal = layer_below.feedback(weighted_error)
                delta = delta - feedback_signal

            # Optional lateral contribution
            if l < len(self.layers) and self.layers[l].lateral_conn is not None:
                delta = delta + self.layers[l].lateral_conn(activations[l])

            deltas[l] = delta

        # Apply updates synchronously
        for l, delta in deltas.items():
            activations[l] = activations[l] - inference_lr * delta

        return activations

    def compute_layer_errors(
        self, activations: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Compute prediction errors at each generative layer.

        errors[i] = activations[i] - layers[i].predict(activations[i+1])
        (optionally precision-weighted)

        Args:
            activations: Current activations [x_0, ..., x_L].

        Returns:
            List of error tensors, one per generative layer (length = num_layers).
        """
        errors = []
        for i, layer in enumerate(self.layers):
            pred = layer.predict(activations[i + 1])
            err = layer.prediction_error(activations[i], pred)
            errors.append(err)
        return errors

    def free_energy(self, activations: list[torch.Tensor]) -> torch.Tensor:
        """Total variational free energy.

        F = 0.5 * sum_l ||errors[l]||^2

        When precision is learnable, the energy also includes a log-det term:
        F_l = 0.5 * (precision * ||e||^2 - log(precision).sum())
        but the precision weighting is already folded into prediction_error(),
        so we just sum squared (precision-weighted) errors here. The log-det
        term is handled in compute_weight_gradients for precision updates.

        Args:
            activations: Settled activations [x_0, ..., x_L].

        Returns:
            Scalar free energy tensor.
        """
        errors = self.compute_layer_errors(activations)
        energy = torch.tensor(0.0, device=activations[0].device)
        for err in errors:
            energy = energy + 0.5 * err.pow(2).sum() / err.shape[0]
        return energy

    def compute_weight_gradients(self, activations: list[torch.Tensor]) -> None:
        """Populate .grad on all parameters using local Hebbian learning rules.

        This does NOT call backward(). It writes .grad directly using the
        analytically derived local update rules:

        For generative weights at layer l:
            W.grad = -(f'(a[l]) * e[l]) @ x[l+1].T / batch_size
            b.grad = -(f'(a[l]) * e[l]).mean(0)  (if bias exists)

        For recognition weights (if non-symmetric):
            Uses error at the layer above for the recognition update.

        For log_precision (if learnable):
            grad = 0.5 * (e_raw^2 - 1).mean(0)
            where e_raw is the un-precision-weighted error.

        Args:
            activations: Settled activations [x_0, ..., x_L].
        """
        batch_size = activations[0].shape[0]

        for i, layer in enumerate(self.layers):
            x_above = activations[i + 1]  # activity at layer above
            pred = layer.predict(x_above)  # prediction at this layer

            # Raw (unweighted) prediction error for precision gradient
            raw_error = activations[i] - pred

            # Precision-weighted error
            error = layer.prediction_error(activations[i], pred)

            # Activation derivative at prediction
            deriv = layer.activation_derivative(pred)

            # Modulated error: f'(prediction) * error
            modulated = deriv * error

            # ---- Generative weight gradient ----
            # W.grad = -(modulated.T @ x_above) / batch_size
            w_grad = -(modulated.t() @ x_above) / batch_size
            layer.generative.weight.grad = w_grad

            if layer.generative.bias is not None:
                layer.generative.bias.grad = -modulated.mean(dim=0)

            # ---- Recognition weight gradient (if non-symmetric) ----
            if layer.recognition is not None:
                # Recognition maps input_size -> output_size, so
                # weight shape is (output_size, input_size).
                # Hebbian rule: correlate layer-above activity with this-layer error.
                r_grad = -(x_above.t() @ error) / batch_size
                layer.recognition.weight.grad = r_grad

            # ---- Lateral weight gradient (if present) ----
            if layer.lateral_conn is not None:
                l_grad = -(error.t() @ activations[i]) / batch_size
                layer.lateral_conn.weight.grad = l_grad

            # ---- Precision gradient (if learnable) ----
            if layer.log_precision is not None:
                # d(F)/d(log_precision) = 0.5 * precision * (e_raw^2 - 1/precision)
                # In log-space: = 0.5 * (precision * e_raw^2 - 1)
                precision = layer.precision
                prec_grad = 0.5 * (precision * raw_error.pow(2) - 1.0).mean(dim=0)
                layer.log_precision.grad = prec_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (bottom-up, no settling).

        Useful for evaluation or compatibility with code that expects
        model(input) -> output. Propagates through recognition/feedback
        path to produce top-layer activations.

        Args:
            x: Input tensor, shape (batch, layer_sizes[0]).

        Returns:
            Top-layer activations, shape (batch, layer_sizes[-1]).
        """
        current = x
        for layer in self.layers:
            current = layer.feedback(current)
            current = layer._activation_fn(current)
        return current

    def parameter_groups(
        self, base_lr: float, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Build per-layer parameter groups for an optimizer.

        Each PCLayer gets its own parameter group, enabling per-layer
        learning rate tuning.

        Args:
            base_lr: Base learning rate applied to all groups.
            **kwargs: Additional optimizer kwargs (weight_decay, etc.)
                applied to all groups.

        Returns:
            List of parameter group dicts suitable for torch.optim constructors.
        """
        groups = []
        for i, layer in enumerate(self.layers):
            groups.append({
                'params': list(layer.parameters()),
                'lr': base_lr,
                'name': f'layer_{i}',
                **kwargs,
            })
        return groups
