import argparse
import json
from pathlib import Path
import numpy as np
from data import load_data
from network import NeuralNetwork, evaluate
from plots import confusion_plot, image_grid


def main():
    parser = argparse.ArgumentParser(description='Evaluate the held-out MNIST test set')
    parser.add_argument('--data', default='data')
    parser.add_argument('--model', default='results/model.npz')
    parser.add_argument('--output', default='results')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    _, _, (inputs, labels) = load_data(args.data, download=not args.no_download)
    model = NeuralNetwork.load(args.model)
    loss, accuracy, predictions = evaluate(model, inputs, labels)
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    confusion_plot(matrix, output / 'confusion_matrix.png')
    np.savetxt(output / 'confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
    # Reserve the first 20 test images exclusively for the prediction demo.
    image_grid(inputs[20:40], labels[20:40], predictions[20:40], output / 'predictions.png', 'Held-out test predictions')
    errors = np.flatnonzero((predictions != labels) & (np.arange(len(labels)) >= 20))[:20]
    image_grid(inputs[errors], labels[errors], predictions[errors], output / 'errors.png', 'Incorrect test predictions')
    result = dict(test_loss=loss, test_accuracy=accuracy, test_count=len(labels),
                  error_examples=[dict(index=int(i), actual=int(labels[i]), predicted=int(predictions[i])) for i in errors])
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
