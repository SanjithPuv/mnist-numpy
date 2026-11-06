import argparse
from pathlib import Path
import numpy as np
from data import load_data
from network import NeuralNetwork, softmax
from plots import image_grid


def main():
    parser = argparse.ArgumentParser(description='Load weights and predict unused test images')
    parser.add_argument('--model', default='results/model.npz')
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/demo.png')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    _, _, (inputs, labels) = load_data(args.data, download=not args.no_download)
    inputs, labels = inputs[:20], labels[:20]
    model = NeuralNetwork.load(args.model)
    probabilities = softmax(model.forward(inputs)[0])
    predictions = probabilities.argmax(axis=1)
    for index, (actual, predicted) in enumerate(zip(labels, predictions)):
        print(f'Test image {index}: predicted {predicted}, actual {actual}, probability {probabilities[index, predicted]:.3f}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image_grid(inputs, labels, predictions, args.output, 'Inference: no gradient or update')


if __name__ == '__main__':
    main()
