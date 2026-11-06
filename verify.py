"""Meaningful numerical checks, independent of downloading MNIST."""
import json
import tempfile
from pathlib import Path
import numpy as np
from network import NeuralNetwork, softmax, cross_entropy


def run_checks():
    rng = np.random.default_rng(7)
    model = NeuralNetwork(4, 5, 3, seed=9, dtype=np.float64)
    inputs = rng.normal(size=(3, 4))
    labels = np.array([0, 2, 1])
    logits, cache = model.forward(inputs)
    assert logits.shape == (3, 3)
    probabilities = softmax(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)
    assert np.all(probabilities >= 0)
    assert np.min(np.abs(cache[1])) > 1e-4
    loss, gradients = model.loss_and_gradients(inputs, labels)
    epsilon = 1e-5
    max_absolute = 0.0
    max_scaled = 0.0
    checked = 0
    for name, parameter in model.parameters.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus_logits, plus_cache = model.forward(inputs)
            parameter[index] = original - epsilon
            minus_logits, minus_cache = model.forward(inputs)
            parameter[index] = original
            # A finite difference must stay on the same smooth ReLU branch.
            assert np.array_equal(plus_cache[1] > 0, minus_cache[1] > 0)
            numeric = (cross_entropy(plus_logits, labels) - cross_entropy(minus_logits, labels)) / (2 * epsilon)
            analytic = gradients[name][index]
            absolute = abs(numeric - analytic)
            scaled = absolute / max(1e-8, abs(numeric) + abs(analytic))
            max_absolute = max(max_absolute, absolute)
            max_scaled = max(max_scaled, scaled)
            np.testing.assert_allclose(numeric, analytic, atol=1e-8, rtol=1e-5)
            checked += 1
    extreme = np.array([[10000., -10000., 0.], [-10000., 10000., 0.]])
    assert np.isfinite(cross_entropy(extreme, np.array([1, 0])))
    np.testing.assert_allclose(softmax(extreme).sum(axis=1), 1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.npz'
        model.save(path)
        loaded = NeuralNetwork.load(path)
        np.testing.assert_array_equal(model.forward(inputs)[0], loaded.forward(inputs)[0])
        np.testing.assert_array_equal(model.predict(inputs), loaded.predict(inputs))
    report = dict(gradient_parameters_checked=checked, max_absolute_error=max_absolute,
                  max_scaled_error=max_scaled, initial_loss=loss,
                  minimum_absolute_relu_input=float(np.min(np.abs(cache[1]))),
                  shapes_probability_sums_extreme_logits_reload='passed')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run_checks()
