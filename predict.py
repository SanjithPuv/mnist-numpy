import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from data import load_data
from network import NeuralNetwork, softmax
from plots import image_grid


def load_digit_image(path):
    """Turn one digit on a plain background into an MNIST-style input row."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        grayscale = Image.alpha_composite(background, rgba).convert('L')
    pixels = np.asarray(grayscale, dtype=np.float32)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background_level = float(np.median(border))
    ink = background_level - pixels if background_level >= 128 else pixels - background_level
    ink = np.maximum(ink, 0)
    peak = float(ink.max())
    if peak < 16:
        raise ValueError('No clear digit found. Use one high-contrast digit with a plain border.')

    # Suppress faint background variation before finding the digit bounds.
    ink[ink < 0.20 * peak] = 0
    rows, columns = np.nonzero(ink)
    crop = ink[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    digit = Image.fromarray(np.rint(crop * (255 / peak)).astype(np.uint8))
    scale = 20 / max(digit.size)
    size = tuple(max(1, round(length * scale)) for length in digit.size)
    digit = digit.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28))
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    # Center by ink mass, as in MNIST. A blank margin leaves room for the shift.
    values = np.asarray(canvas, dtype=np.float32)
    row_mass, column_mass = values.sum(axis=1), values.sum(axis=0)
    mass = row_mass.sum()
    if mass == 0:
        raise ValueError('The digit disappeared after resizing. Crop the image closer to the digit.')
    coordinates = np.arange(28)
    dx = np.clip(13.5 - coordinates @ column_mass / mass, -4, 4)
    dy = np.clip(13.5 - coordinates @ row_mass / mass, -4, 4)
    centered = canvas.transform((28, 28), Image.Transform.AFFINE,
                                (1, 0, -dx, 0, 1, -dy), Image.Resampling.BILINEAR)
    return np.asarray(centered, dtype=np.float32).reshape(1, 784) / 255.0


def main():
    parser = argparse.ArgumentParser(description='Predict a handwritten digit photo or run the MNIST demo')
    parser.add_argument('--model', default='results/model.npz')
    parser.add_argument('--image', type=Path, help='Path to an image of one handwritten digit')
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/demo.png')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f'Model not found at {args.model}. Run python train.py first or pass --model.')
    model = NeuralNetwork.load(args.model)
    if args.image is not None:
        try:
            inputs = load_digit_image(args.image)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            parser.error(f'Could not read digit image {args.image}: {error}')
        probabilities = softmax(model.forward(inputs)[0])[0]
        digit = int(probabilities.argmax())
        print(f'Predicted digit: {digit} (model score: {probabilities[digit]:.1%})')
        return

    _, _, (inputs, labels) = load_data(args.data, download=not args.no_download)
    inputs, labels = inputs[:20], labels[:20]
    probabilities = softmax(model.forward(inputs)[0])
    predictions = probabilities.argmax(axis=1)
    for index, (actual, predicted) in enumerate(zip(labels, predictions)):
        print(f'Test image {index}: predicted {predicted}, actual {actual}, probability {probabilities[index, predicted]:.3f}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image_grid(inputs, labels, predictions, args.output, 'Inference: no gradient or update')


if __name__ == '__main__':
    main()
