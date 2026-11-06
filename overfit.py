"""Can the complete 784-128-10 model memorize 32 real training examples?"""
import argparse
import numpy as np
from data import load_data
from network import NeuralNetwork, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    (inputs, labels), _, _ = load_data(args.data, download=not args.no_download)
    inputs, labels = inputs[:32], labels[:32]
    model = NeuralNetwork(seed=42)
    initial, _, _ = evaluate(model, inputs, labels)
    for step in range(1000):
        _, gradients = model.loss_and_gradients(inputs, labels)
        model.update(gradients, 0.1)
    loss, accuracy, _ = evaluate(model, inputs, labels)
    print(f'32-example memorization: initial loss={initial:.6f}, final loss={loss:.6f}, accuracy={accuracy:.2%}')
    assert accuracy == 1.0 and loss < 0.05


if __name__ == '__main__':
    main()
