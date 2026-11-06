"""Reproduce the tutorial's one-example forward/backward/update trace."""
import numpy as np
from network import NeuralNetwork, softmax

np.set_printoptions(precision=6, suppress=True)
model = NeuralNetwork(2, 2, 2, dtype=np.float64)
model.parameters = dict(W1=np.array([[.1, -.2], [.3, .4]]), b1=np.zeros(2),
                        W2=np.array([[.2, -.1], [-.3, .4]]), b2=np.zeros(2))
inputs, labels = np.array([[1., 2.]]), np.array([1])
logits, (_, hidden_logits, hidden) = model.forward(inputs)
loss, gradients = model.loss_and_gradients(inputs, labels)
print('Z1:', hidden_logits, '\nH:', hidden, '\nZ2:', logits)
print('Probabilities:', softmax(logits), '\nLoss:', loss)
for name, gradient in gradients.items():
    print('Gradient', name, gradient.shape, '\n', gradient)
model.update(gradients, .1)
for name, parameter in model.parameters.items():
    print('Updated', name, '\n', parameter)
print('New probabilities:', softmax(model.forward(inputs)[0]))
print('New loss:', model.loss_and_gradients(inputs, labels)[0])
