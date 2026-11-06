"""Two-layer classifier. Every derivative is calculated explicitly."""
import numpy as np


def require_finite(name, values):
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f'{name} contains NaN or infinity')


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    # log-sum-exp computes log probabilities without first rounding them to zero.
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    loss = np.mean(log_normalizer - shifted[np.arange(len(labels)), labels])
    require_finite('loss', loss)
    return float(loss)


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 seed=42, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.parameters = {
            'W1': (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2 / input_size)).astype(dtype),
            'b1': np.zeros(hidden_size, dtype=dtype),
            'W2': (rng.standard_normal((hidden_size, output_size)) * np.sqrt(1 / hidden_size)).astype(dtype),
            'b2': np.zeros(output_size, dtype=dtype),
        }

    def forward(self, inputs):
        p = self.parameters
        if inputs.ndim != 2 or inputs.shape[1] != p['W1'].shape[0] or len(inputs) == 0:
            raise ValueError('Expected nonempty input rows matching W1')
        hidden_logits = inputs @ p['W1'] + p['b1']
        hidden = np.maximum(0, hidden_logits)
        logits = hidden @ p['W2'] + p['b2']
        require_finite('logits', logits)
        return logits, (inputs, hidden_logits, hidden)

    def loss_and_gradients(self, inputs, labels):
        logits, (inputs, hidden_logits, hidden) = self.forward(inputs)
        loss = cross_entropy(logits, labels)
        output_gradient = softmax(logits)
        output_gradient[np.arange(len(labels)), labels] -= 1
        output_gradient /= len(labels)  # Average exactly once, here.
        gradients = {}
        gradients['W2'] = hidden.T @ output_gradient
        gradients['b2'] = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ self.parameters['W2'].T
        hidden_gradient *= hidden_logits > 0
        gradients['W1'] = inputs.T @ hidden_gradient
        gradients['b1'] = hidden_gradient.sum(axis=0)
        for name, gradient in gradients.items():
            assert gradient.shape == self.parameters[name].shape
            require_finite(name + ' gradient', gradient)
        return loss, gradients

    def update(self, gradients, learning_rate):
        for name in self.parameters:
            self.parameters[name] -= learning_rate * gradients[name]
            require_finite(name, self.parameters[name])

    def predict(self, inputs):
        return self.forward(inputs)[0].argmax(axis=1)

    def save(self, path):
        np.savez_compressed(path, **self.parameters)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            p = {name: saved[name].copy() for name in ('W1', 'b1', 'W2', 'b2')}
        if (p['W1'].ndim != 2 or p['W2'].ndim != 2 or
            p['b1'].shape != (p['W1'].shape[1],) or
            p['W2'].shape[0] != p['W1'].shape[1] or p['b2'].shape != (p['W2'].shape[1],)):
            raise ValueError('Incompatible saved parameter shapes')
        model = cls(p['W1'].shape[0], p['W1'].shape[1], p['W2'].shape[1])
        model.parameters = p
        for name, value in p.items():
            require_finite(name, value)
        return model


def evaluate(model, inputs, labels, batch_size=1024):
    total_loss = 0.0
    predictions = []
    for start in range(0, len(labels), batch_size):
        logits, _ = model.forward(inputs[start:start + batch_size])
        batch_labels = labels[start:start + batch_size]
        total_loss += cross_entropy(logits, batch_labels) * len(batch_labels)
        predictions.append(logits.argmax(axis=1))
    predictions = np.concatenate(predictions)
    return total_loss / len(labels), float(np.mean(predictions == labels)), predictions
