"""Tests for framework/models/predictive_coding.py — PC model hierarchy."""

import pytest
import torch
import torch.nn as nn

from framework.models.predictive_coding import (
    _tanh_deriv, _relu_deriv, _linear_deriv, _ACTIVATIONS,
    PCLayerConfig, PCLayer, PredictiveCodingNetwork,
)


class TestActivationDerivatives:

    def test_tanh_deriv(self):
        """tanh derivative: 1 - output^2."""
        output = torch.tensor([0.0, 0.5, -0.5, 1.0])
        expected = 1.0 - output ** 2
        result = _tanh_deriv(output)
        assert torch.allclose(result, expected)

    def test_relu_deriv(self):
        """relu derivative: 1 where output > 0, 0 elsewhere."""
        output = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        result = _relu_deriv(output)
        assert torch.allclose(result, expected)

    def test_linear_deriv(self):
        """linear derivative: all ones."""
        output = torch.tensor([1.0, -1.0, 5.0])
        result = _linear_deriv(output)
        assert torch.allclose(result, torch.ones_like(output))

    def test_activations_registry_keys(self):
        """Registry has tanh, relu, linear."""
        assert 'tanh' in _ACTIVATIONS
        assert 'relu' in _ACTIVATIONS
        assert 'linear' in _ACTIVATIONS

    def test_activations_registry_values_are_callable(self):
        """Each entry is (activation_fn, derivative_fn)."""
        for name, (fn, deriv) in _ACTIVATIONS.items():
            assert callable(fn)
            assert callable(deriv)


class TestPCLayerConfig:

    def test_defaults(self):
        cfg = PCLayerConfig(input_size=8, output_size=4)
        assert cfg.activation == 'tanh'
        assert cfg.symmetric is True
        assert cfg.lateral is False
        assert cfg.learnable_precision is False

    def test_custom_values(self):
        cfg = PCLayerConfig(
            input_size=8, output_size=4, activation='relu',
            symmetric=False, lateral=True, learnable_precision=True,
        )
        assert cfg.activation == 'relu'
        assert cfg.symmetric is False
        assert cfg.lateral is True
        assert cfg.learnable_precision is True


class TestPCLayer:

    def test_basic_construction(self):
        """Default config creates generative only."""
        cfg = PCLayerConfig(input_size=8, output_size=4)
        layer = PCLayer(cfg)
        assert layer.generative is not None
        assert layer.recognition is None
        assert layer.lateral_conn is None
        assert layer.log_precision is None

    def test_non_symmetric_has_recognition(self):
        """Non-symmetric creates separate recognition weights."""
        cfg = PCLayerConfig(input_size=8, output_size=4, symmetric=False)
        layer = PCLayer(cfg)
        assert layer.recognition is not None
        assert layer.recognition.weight.shape == (4, 8)

    def test_lateral_has_connection(self):
        """Lateral=True creates lateral connections."""
        cfg = PCLayerConfig(input_size=8, output_size=4, lateral=True)
        layer = PCLayer(cfg)
        assert layer.lateral_conn is not None
        assert layer.lateral_conn.weight.shape == (8, 8)

    def test_learnable_precision(self):
        """Learnable precision creates log_precision parameter."""
        cfg = PCLayerConfig(input_size=8, output_size=4, learnable_precision=True)
        layer = PCLayer(cfg)
        assert layer.log_precision is not None
        assert layer.log_precision.shape == (8,)

    def test_precision_property(self):
        """Precision returns exp(log_precision)."""
        cfg = PCLayerConfig(input_size=4, output_size=2, learnable_precision=True)
        layer = PCLayer(cfg)
        precision = layer.precision
        assert precision is not None
        assert torch.allclose(precision, torch.exp(layer.log_precision))

    def test_precision_none_when_not_learnable(self):
        cfg = PCLayerConfig(input_size=4, output_size=2)
        layer = PCLayer(cfg)
        assert layer.precision is None

    def test_unknown_activation_raises(self):
        """Unknown activation raises ValueError."""
        cfg = PCLayerConfig(input_size=4, output_size=2, activation='unknown')
        with pytest.raises(ValueError, match="Unknown activation"):
            PCLayer(cfg)

    def test_predict_shape(self):
        """predict returns correct shape."""
        cfg = PCLayerConfig(input_size=8, output_size=4)
        layer = PCLayer(cfg)
        x_above = torch.randn(3, 4)  # batch=3, output_size=4
        pred = layer.predict(x_above)
        assert pred.shape == (3, 8)  # batch=3, input_size=8

    def test_prediction_error_no_precision(self):
        """prediction_error = activity - prediction (no precision)."""
        cfg = PCLayerConfig(input_size=4, output_size=2)
        layer = PCLayer(cfg)
        activity = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        prediction = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
        error = layer.prediction_error(activity, prediction)
        expected = activity - prediction
        assert torch.allclose(error, expected)

    def test_prediction_error_with_precision(self):
        """prediction_error is precision-weighted when learnable."""
        cfg = PCLayerConfig(input_size=4, output_size=2, learnable_precision=True)
        layer = PCLayer(cfg)
        activity = torch.ones(1, 4)
        prediction = torch.zeros(1, 4)
        error = layer.prediction_error(activity, prediction)
        # error = precision * (activity - prediction)
        expected = layer.precision * (activity - prediction)
        assert torch.allclose(error, expected)

    def test_activation_derivative(self):
        """activation_derivative returns correct derivative."""
        cfg = PCLayerConfig(input_size=4, output_size=2, activation='tanh')
        layer = PCLayer(cfg)
        output = torch.tensor([[0.5, -0.5, 0.0, 0.9]])
        deriv = layer.activation_derivative(output)
        expected = _tanh_deriv(output)
        assert torch.allclose(deriv, expected)

    def test_feedback_symmetric(self):
        """Symmetric feedback uses generative weight transpose."""
        cfg = PCLayerConfig(input_size=4, output_size=2, symmetric=True)
        layer = PCLayer(cfg)
        error = torch.randn(3, 4)
        feedback = layer.feedback(error)
        assert feedback.shape == (3, 2)  # projects to output_size

    def test_feedback_non_symmetric(self):
        """Non-symmetric feedback uses recognition weights."""
        cfg = PCLayerConfig(input_size=4, output_size=2, symmetric=False)
        layer = PCLayer(cfg)
        error = torch.randn(3, 4)
        feedback = layer.feedback(error)
        assert feedback.shape == (3, 2)


class TestPredictiveCodingNetwork:

    def test_minimum_layers(self):
        """Needs at least 2 layer sizes."""
        with pytest.raises(ValueError, match="at least 2"):
            PredictiveCodingNetwork([4])

    def test_wrong_config_count(self):
        """Wrong number of layer configs raises."""
        configs = [PCLayerConfig(4, 2)]
        with pytest.raises(ValueError, match="Expected 2"):
            PredictiveCodingNetwork([4, 8, 2], layer_configs=configs)

    def test_basic_construction(self):
        """3-layer network with defaults."""
        net = PredictiveCodingNetwork([8, 4, 2])
        assert net.num_layers == 2
        assert net.layer_sizes == [8, 4, 2]
        assert len(net.layers) == 2

    def test_custom_configs(self):
        """Build with explicit PCLayerConfigs."""
        configs = [
            PCLayerConfig(8, 4, activation='relu'),
            PCLayerConfig(4, 2, activation='linear'),
        ]
        net = PredictiveCodingNetwork([8, 4, 2], layer_configs=configs)
        assert net.num_layers == 2

    def test_initialize_activations_shapes(self):
        """initialize_activations returns correct number and shapes."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        assert len(activations) == 3  # input + 2 hidden
        assert activations[0].shape == (3, 8)
        assert activations[1].shape == (3, 4)
        assert activations[2].shape == (3, 2)

    def test_initialize_activations_input_detached(self):
        """Input activation (layer 0) is detached from the original tensor."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8, requires_grad=True)
        activations = net.initialize_activations(x)
        # Layer 0 is explicitly detached + cloned from x
        assert not activations[0].requires_grad
        # Clone ensures separate storage from x
        assert activations[0].data_ptr() != x.data_ptr()

    def test_inference_step_preserves_clamped(self):
        """inference_step does not modify clamped layers."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        clamped_value = activations[0].clone()

        activations = net.inference_step(activations, clamped={0: x}, inference_lr=0.1)
        assert torch.allclose(activations[0], clamped_value)

    def test_inference_step_updates_free(self):
        """inference_step modifies unclamped layers."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        act1_before = activations[1].clone()

        activations = net.inference_step(activations, clamped={0: x}, inference_lr=0.5)
        # Layer 1 should change (it's free)
        assert not torch.allclose(activations[1], act1_before)

    def test_compute_layer_errors_count(self):
        """compute_layer_errors returns one error per layer."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        errors = net.compute_layer_errors(activations)
        assert len(errors) == 2
        assert errors[0].shape == (3, 8)
        assert errors[1].shape == (3, 4)

    def test_free_energy_scalar(self):
        """free_energy returns a scalar tensor."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        energy = net.free_energy(activations)
        assert energy.dim() == 0  # scalar
        assert energy.item() >= 0  # sum of squared errors

    def test_free_energy_decreases_with_settling(self):
        """Free energy should decrease after settling iterations."""
        torch.manual_seed(42)
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)

        energy_before = net.free_energy(activations).item()
        for _ in range(20):
            activations = net.inference_step(activations, clamped={0: x}, inference_lr=0.05)
        energy_after = net.free_energy(activations).item()

        assert energy_after <= energy_before

    def test_compute_weight_gradients_populates_grad(self):
        """compute_weight_gradients sets .grad on all generative parameters."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)

        # Ensure no grads initially
        for layer in net.layers:
            assert layer.generative.weight.grad is None

        net.compute_weight_gradients(activations)

        for layer in net.layers:
            assert layer.generative.weight.grad is not None
            assert layer.generative.bias.grad is not None

    def test_compute_weight_gradients_recognition(self):
        """Recognition weight grads are set for non-symmetric layers."""
        net = PredictiveCodingNetwork([8, 4, 2], default_symmetric=False)
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        net.compute_weight_gradients(activations)

        for layer in net.layers:
            assert layer.recognition.weight.grad is not None

    def test_compute_weight_gradients_lateral(self):
        """Lateral weight grads are set when lateral=True."""
        net = PredictiveCodingNetwork([8, 4, 2], default_lateral=True)
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        net.compute_weight_gradients(activations)

        for layer in net.layers:
            assert layer.lateral_conn.weight.grad is not None

    def test_compute_weight_gradients_precision(self):
        """Precision grads are set when learnable_precision=True."""
        net = PredictiveCodingNetwork([8, 4, 2], default_learnable_precision=True)
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        net.compute_weight_gradients(activations)

        for layer in net.layers:
            assert layer.log_precision.grad is not None

    def test_forward_shape(self):
        """forward returns top-layer activations."""
        net = PredictiveCodingNetwork([8, 4, 2])
        x = torch.randn(3, 8)
        out = net.forward(x)
        assert out.shape == (3, 2)

    def test_parameter_groups(self):
        """parameter_groups returns per-layer groups."""
        net = PredictiveCodingNetwork([8, 4, 2])
        groups = net.parameter_groups(base_lr=0.01)
        assert len(groups) == 2
        for i, g in enumerate(groups):
            assert g['lr'] == 0.01
            assert g['name'] == f'layer_{i}'
            assert len(g['params']) > 0

    def test_inference_step_with_lateral(self):
        """inference_step includes lateral contribution."""
        net = PredictiveCodingNetwork([8, 4, 2], default_lateral=True)
        x = torch.randn(3, 8)
        activations = net.initialize_activations(x)
        activations = net.inference_step(activations, clamped={0: x})
        # Just verify it runs without error
        assert len(activations) == 3
