# Live Hand Number Recognition

Recognize the number of raised fingers using your MacBook camera. The live preview displays a total from **0 to 10** across up to two hands, with hand landmarks and colored fingertips. A closed fist counts as zero; an empty frame displays "No hand detected."

## Run the camera on macOS

Use Python 3.10–3.12 (Python 3.12 recommended). In Terminal:

```bash
git clone https://github.com/SanjithPuv/mnist-numpy.git
cd mnist-numpy
python3.12 -m venv .venv-camera
source .venv-camera/bin/activate
python -m pip install mediapipe opencv-python
python camera.py
```

If you already cloned the repository, run `git pull` inside it, then follow the environment/install/run steps above. If your `python3` is already version 3.10–3.12, you can use it instead of `python3.12`.

Allow the camera permission when macOS asks. If needed, enable Terminal (or the editor running Python) under **System Settings → Privacy & Security → Camera**, then restart that app. Close FaceTime or other apps using the camera. If the wrong camera opens, try `python camera.py --camera 1`.

Face your palms toward the camera, keep fingers separated, and use good lighting. Show 0–5 fingers on one hand or combine both hands for 0–10. Hold each pose briefly for the total to stabilize. Press **Q**, **Esc**, or close the preview window to stop and release the camera.

The first run downloads Google's pretrained Hand Landmarker model into `models/hand_landmarker.task`. Later runs reuse the cached model. Frames are processed locally, without saving or uploading video. No MNIST training or weights are needed for camera mode.

## How camera recognition works

OpenCV reads the webcam. [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) estimates 21 landmarks per hand. The program uses joint angles and distances to estimate which fingers are extended, adds their counts, and requires a consistent total for 0.25 seconds before displaying it.

This is **finger counting**, not ASL numeral recognition. The counting rules are a heuristic, so bent fingers, overlapping hands, sideways views, or poor lighting can give incorrect results. Camera accuracy has not been measured. The original MNIST accuracy below applies only to handwritten digits.

## Original handwritten digit project

The earlier image classifier and its tutorial remain available as a separate learning example. The webcam feature uses a pretrained hand model; it does not use the NumPy digit network.

### MNIST from scratch in NumPy

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

## Predict a digit from your own image

After installing the dependencies and running `python train.py` to create `results/model.npz`, run:

```bash
python predict.py --image path/to/my_digit.png
```

Use a PNG or JPEG of one digit from 0 to 9, with clear contrast and a plain margin around it. Crop away other writing and shadows. The script crops, resizes, and centers the digit for the model, then prints its prediction and model score. A high score does not guarantee the prediction is correct.

Use `--model path/to/model.npz` if your trained weights are saved elsewhere. Run `python predict.py` without `--image` for the original MNIST demo.

## File map

| File | Purpose |
|---|---|
| requirements.txt | NumPy, Matplotlib, and Pillow dependencies |
| data.py | Download/cache, IDX parsing, normalization, fixed split |
| network.py | Initialization, forward, stable softmax/loss, gradients, SGD update, save/load, metrics |
| train.py | Shuffled mini-batches, metrics, validation checkpoint selection |
| evaluate.py | Held-out metrics, confusion matrix, examples and mistakes |
| predict.py | Classify a local image or run the MNIST prediction demo |
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
