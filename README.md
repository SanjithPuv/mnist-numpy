# MNIST from scratch in NumPy

A readable 784 → 128 (ReLU) → 10 neural network with manual backpropagation and mini-batch SGD. No automatic differentiation or machine learning framework. Start with this README, then follow `TUTORIAL.md` beside the source.

## Setup

Clone the repository and run:

```bash
git clone https://github.com/SanjithPuv/mnist-numpy.git
cd mnist-numpy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -m venv .venv` and `.venv\Scripts\activate` instead. The implementation targets Python 3.10 or later. This run used Python with NumPy 2.3.5 and Matplotlib 3.10.8.

## Run in this order

```bash
python verify.py
python worked_example.py
python overfit.py
python train.py --epochs 15 --batch-size 128 --learning-rate 0.1 --seed 42
python evaluate.py --model results/model.npz
python predict.py --model results/model.npz
```

This repository includes source, tutorial, and measured results. Run training to generate weights and plots locally. The loader downloads missing files from Google's public MNIST mirror at https://storage.googleapis.com/cvdf-datasets/mnist/. You may instead supply the four original compressed or uncompressed IDX files in `data/`. Add `--no-download` to overfit, training, evaluation, and prediction commands to forbid network access. Download failures produce a clear error rather than substituting data.

If CPU execution is slow, on macOS/Linux try:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py --output results_new
```

On Windows PowerShell set `$env:OPENBLAS_NUM_THREADS="1"` before running Python. Training time depends on CPU and NumPy's BLAS library. Training holds the normalized dataset in memory and makes a shuffled split, so allow several hundred MB of RAM. A virtual environment is recommended; do not include it in source control.

Use `--output results_experiment` to preserve the supplied run when experimenting. Evaluation also has `--output`; prediction takes an output image filename. `--help` shows each script's options. Only the training loader's seed affects the train/validation split; test data remains unchanged.

## File map

| File | Purpose |
|---|---|
| requirements.txt | NumPy and Matplotlib dependencies |
| data.py | Download/cache, IDX parsing, normalization, fixed split |
| network.py | Initialization, forward, stable softmax/loss, gradients, SGD update, save/load, metrics |
| train.py | Shuffled mini-batches, metrics, validation checkpoint selection |
| evaluate.py | Held-out metrics, confusion matrix, examples and mistakes |
| predict.py | Saved-model inference on 20 test images unused in training |
| plots.py | Sample grids, learning curves, confusion matrix |
| verify.py | float64 finite-difference check, shapes, stability, reload check |
| overfit.py | Memorize 32 real training images |
| worked_example.py | Print every stage of the tutorial's numerical example |
| TUTORIAL.md | Concepts, equations, shapes, code-block walkthrough, diagnoses, experiments |
| RESULTS.md | Measured results and interpretation |
| data/ (generated) | Downloaded MNIST files |
| results/ (generated) | Weights, configuration, metrics, plots, numerical verification |


## Actual results

The supplied 15-epoch run achieved **97.08% test accuracy** (9,708 / 10,000), with test cross-entropy **0.104279**. The 90% target was exceeded. Training accuracy was 98.13%, validation accuracy 96.90%. The selected checkpoint was epoch 15, based on validation loss, not test accuracy.

All 43 tiny-network parameter derivatives passed finite-difference checking; maximum absolute error was 1.84e-11. Shapes, probability sums, extreme logits, and exact reload predictions passed. The full network memorized 32 training examples with 100% accuracy and loss 0.001847. See RESULTS.md for limitations and TUTORIAL.md for explanations.

## Common mistakes

Check that images are normalized, batches are rows, labels stay paired with images, and `@` is used for matrix products. Average gradients once, use the pre-update W2 when propagating hidden gradients, and use stable logits-based loss. If learning fails, run `verify.py` followed by `overfit.py`. If training improves while validation worsens, use the validation-selected checkpoint. The tutorial gives a fuller troubleshooting table.

## Three experiments

1. Compare learning rates 0.01 and 0.5: expect slower learning versus larger, potentially unstable changes.
2. Compare batch sizes 32 and 512: expect noisier and more frequent updates versus smoother and fewer updates per epoch.
3. Change hidden size to 32 or 256 in network.py: expect a capacity/speed tradeoff and potentially more overfitting with the larger model.

Use validation results to compare experiments. Keep the test set for a final assessment.
