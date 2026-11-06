"""Download, validate, and parse the original MNIST IDX files."""
import gzip
from pathlib import Path
import struct
import urllib.request
import numpy as np

BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


def read_idx(path, images):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        raw = stream.read()
    header_size = 16 if images else 8
    if len(raw) < header_size:
        raise ValueError(f'Truncated IDX header: {path}')
    if images:
        magic, count, rows, columns = struct.unpack('>IIII', raw[:16])
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(f'Invalid image header: {path}')
        shape = (count, rows * columns)
    else:
        magic, count = struct.unpack('>II', raw[:8])
        if magic != 2049:
            raise ValueError(f'Invalid label header: {path}')
        shape = (count,)
    values = np.frombuffer(raw, dtype=np.uint8, offset=header_size)
    if values.size != int(np.prod(shape)):
        raise ValueError(f'IDX payload length mismatch: {path}')
    return values.reshape(shape)


def load_data(directory='data', seed=42, validation_size=10000, download=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        path = directory / name
        plain = directory / name.removesuffix('.gz')
        if not path.exists() and plain.exists():
            path = plain
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'Place {name} (or its uncompressed version) in {directory}')
            temporary = directory / (name + '.part.gz')
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                    temporary.write_bytes(response.read())
                # Validate gzip/IDX before promoting the download to the cache.
                read_idx(temporary, images='images' in name)
                temporary.replace(path)
            except Exception as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f'MNIST download failed: {error}. Place the four original '
                                   f'IDX files in {directory} and use --no-download.') from error
        paths.append(path)
    train_images = read_idx(paths[0], True).astype(np.float32) / 255.0
    train_labels = read_idx(paths[1], False).astype(np.int64)
    test_images = read_idx(paths[2], True).astype(np.float32) / 255.0
    test_labels = read_idx(paths[3], False).astype(np.int64)
    for images, labels in ((train_images, train_labels), (test_images, test_labels)):
        if len(images) != len(labels) or np.any(labels > 9):
            raise ValueError('Image/label count mismatch or invalid digit label')
    if not 0 < validation_size < len(train_images):
        raise ValueError('validation_size must leave nonempty training and validation sets')
    order = np.random.default_rng(seed).permutation(len(train_images))
    validation, training = order[:validation_size], order[validation_size:]
    return ((train_images[training], train_labels[training]),
            (train_images[validation], train_labels[validation]), (test_images, test_labels))
