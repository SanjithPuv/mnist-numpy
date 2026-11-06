import argparse
import json
from pathlib import Path
import numpy as np
from data import load_data
from network import NeuralNetwork, evaluate
from plots import curves, image_grid


def main():
    parser = argparse.ArgumentParser(description='Train a NumPy MNIST network')
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('epochs, batch-size, and learning-rate must be positive')
    training, validation, test = load_data(args.data, args.seed, download=not args.no_download)
    inputs, labels = training
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Train {inputs.shape}, validation {validation[0].shape}, test {test[0].shape}')
    print(f'Train pixel range [{inputs.min()}, {inputs.max()}], label range [{labels.min()}, {labels.max()}]')
    image_grid(inputs, labels, None, output / 'dataset_samples.png', 'Training samples')
    model = NeuralNetwork(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), args.batch_size):
            indices = order[start:start + args.batch_size]
            _, gradients = model.loss_and_gradients(inputs[indices], labels[indices])
            model.update(gradients, args.learning_rate)
        train_loss, train_accuracy, _ = evaluate(model, *training)
        validation_loss, validation_accuracy, _ = evaluate(model, *validation)
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                   validation_loss=validation_loss, validation_accuracy=validation_accuracy)
        history.append(row)
        print(f'Epoch {epoch:2}: train loss {train_loss:.4f}, accuracy {train_accuracy:.2%}; '
              f'validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.2%}', flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save(output / 'model.npz')
    model.save(output / 'last_model.npz')
    (output / 'history.json').write_text(json.dumps(history, indent=2))
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    curves(history, output / 'learning_curves.png')
    print('Saved model.npz selected by validation loss. Test labels were not used for selection.')


if __name__ == '__main__':
    main()
