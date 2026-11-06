"""Headless plots work on laptops and servers alike."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def image_grid(inputs, labels, predictions, path, title):
    count = min(20, len(labels))
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index < count:
            axis.imshow(inputs[index].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            text = f'Actual: {labels[index]}'
            if predictions is not None:
                text += f' / Pred: {predictions[index]}'
            axis.set_title(text, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'validation'):
            axis.plot(np.arange(1, len(history) + 1), [row[f'{split}_{metric}'] for row in history], label=split)
        axis.set(xlabel='Epoch', ylabel=metric)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion_plot(matrix, path):
    fig, axis = plt.subplots(figsize=(8, 7))
    picture = axis.imshow(matrix, cmap='Blues')
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha='center', va='center', fontsize=8,
                      color='white' if matrix[row, column] > matrix.max()/2 else 'black')
    axis.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='Actual digit')
    fig.colorbar(picture, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
