# Learning handwritten digits, one operation at a time

Read this beside `network.py`. Everything below describes the implementation in this project. No differentiation or neural network library is involved.

## 1. What the task means

An image is a grid of brightness measurements. MNIST images have 28 rows and 28 columns, so there are 784 measurements per image. Its label is an integer from 0 to 9, supplied by the dataset. Flattening changes the arrangement in memory, not the pixel information: pixel at row r, column c becomes feature 28*r+c. This dense network does not explicitly know which pixels are neighbors.

In `data.py`, `FILES` names the four official MNIST files. `urlopen` downloads their bytes; `.part` prevents an interrupted transfer from becoming a completed cache entry. A gzip decompression check precedes the rename. `read_idx` then validates the IDX header and exact payload length. Existing compressed or uncompressed local files are accepted. No alternative dataset is used on failure.

IDX integers are big-endian, which means the most significant byte comes first. `struct.unpack('>IIII', ...)` reads four unsigned 32-bit integers: magic number 2051, image count, row count, column count. Image pixels follow as unsigned bytes. Labels use magic 2049 and a count, followed by one byte per label. Gzip wraps this format with compression. `np.frombuffer` interprets the payload; `reshape` supplies its dimensions.

`astype(np.float32) / 255.0` maps black 0 to 0.0 and white 255 to 1.0. A gray value of 128 becomes about 0.502. Smaller, consistently scaled inputs help keep updates manageable. float32 saves memory and computation; numerical gradient checks use float64 for greater precision.

The official 60,000 training examples are shuffled deterministically and split into 50,000 training and 10,000 validation examples. The official 10,000 test examples stay separate. Training examples determine gradients. Validation examples select the saved checkpoint with the lowest validation loss. Test examples measure performance after that selection. Looking at test results repeatedly to choose learning rates would indirectly tune to the test set, weakening that measurement.

`load_data` returns three `(images, labels)` pairs. Image shapes are `(50000,784)`, `(10000,784)`, `(10000,784)`; corresponding label shapes are `(50000,)`, `(10000,)`, `(10000,)`. `train.py` prints ranges and writes `dataset_samples.png` so you can check that parsing produced actual digits.

## 2. Neurons, weights, and shapes

A neuron computes a weighted sum and adds a bias. For one hidden neuron j:

    z_j = sum_i(x_i * W1_ij) + b1_j

Here i identifies an input pixel, j identifies a hidden neuron, x_i is pixel brightness, W1_ij is its learned influence, and b1_j is an adjustable offset. A negative weight can suppress a neuron when a pixel is bright. A bias shifts the threshold even when all inputs are zero. The hidden neurons learn useful combinations of pixels rather than receiving hand-written digit rules.

For B images at once, the forward equations are:

    Z1 = X W1 + b1
    H  = ReLU(Z1)
    Z2 = H W2 + b2
    P  = softmax(Z2)

| Symbol | Meaning | Shape |
|---|---|---|
| X | Batch of input rows | (B,784) |
| W1 | Input-to-hidden weights | (784,128) |
| b1 | Hidden offsets | (128,) |
| Z1 | Hidden values before activation | (B,128) |
| H | Hidden values after activation | (B,128) |
| W2 | Hidden-to-output weights | (128,10) |
| b2 | Output offsets | (10,) |
| Z2 | Output logits, or raw scores | (B,10) |
| P | Class probabilities | (B,10) |

Matrix multiplication adds products over matching inner dimensions. For example, `[2,3] @ [[1,4],[5,6]]` gives `[17,26]`, because 2*1+3*5=17 and 2*4+3*6=26. NumPy's `@` means matrix multiplication, while `*` means elementwise multiplication. Broadcasting adds the bias row to each image's row without storing B copies.

`forward` checks the input shape, calculates these three stages, and returns logits plus a cache. The cache stores X, Z1, and H because backward calculations need them. ReLU is `max(0,z)` per entry: `[-2,0,3]` becomes `[0,0,3]`. Without a nonlinear activation, two affine layers collapse to a single affine layer, limiting the patterns they can represent.

`__init__` draws independent random weights. Hidden weights use standard deviation sqrt(2/784), called He initialization; this helps compensate for ReLU discarding negative values. Output weights use sqrt(1/128), a moderate variance for raw class scores. Biases begin at zero. Initializing all hidden weights identically would make hidden neurons behave identically and receive matching updates. With all-zero weights and our zero derivative at ReLU zero, hidden learning can also stall.

There are 101,770 trainable numbers: 784*128 + 128 + 128*10 + 10.

## 3. Stable probabilities and loss

For one image with logits z_k, class k has probability:

    p_k = exp(z_k - m) / sum_j exp(z_j - m),   m = max_j z_j

Indices k and j range over classes. Subtracting the same m leaves the ratios unchanged. It makes every exponent's input nonpositive, preventing overflow. For logits `[1001,1002]`, use `[-1,0]`; the resulting probabilities are about `[0.269,0.731]`. `keepdims=True` preserves shape `(B,1)` for correct row broadcasting. Summing over axis 1 adds classes within each image.

Cross-entropy for a batch is:

    L = (1/B) sum_n [log(sum_k exp(z_nk - m_n)) - (z_n,y_n - m_n)]

B is the number of images, n identifies an image, y_n is its true digit, z_nk is its score for class k, and m_n is that row's maximum score. This equals the average `-log(p_correct)`, but `cross_entropy` calculates it directly from logits. A very small probability can round to zero; taking its logarithm would give negative infinity. For scores `[10000,-10000]` and true class 1, the direct formula gives a finite loss of 20000 even if the incorrect class probability rounds to zero.

The code subtracts row maxima, computes the log normalizer, selects each correct-class score with `np.arange` and `labels`, and averages. A correct probability of 0.9 gives loss about 0.105; 0.1 gives about 2.303. Loss measures confidence as well as correctness. Accuracy only asks whether the largest score belongs to the right class. `argmax(axis=1)` picks that class without needing softmax.

## 4. Calculus and the chain rule

A derivative says how quickly one quantity changes when another changes. If f(w)=w*w, then f'(w)=2w. At w=3, increasing w by 0.001 changes f by about 6*0.001.

For a chain, let u=2w+1 and L=u*u. Then dL/du=2u and du/dw=2, so dL/dw=4u. At w=3, u=7 and the derivative is 28. Backpropagation repeats this local rule backward through the network. A gradient is a collection of derivatives, one for each entry in an array. It points toward increasing loss; SGD subtracts it.

For one example, L=-z_y+log(sum_j exp(z_j)). Differentiating with respect to z_k gives:

    dL/dz_k = -1[k=y] + exp(z_k)/sum_j exp(z_j)
            = p_k - 1[k=y]

The indicator 1[k=y] is 1 for the true class and 0 otherwise. This is the combined softmax/cross-entropy derivative; no full softmax Jacobian is needed. In a batch:

    D2 = (P - Y) / B

Y is the conceptual one-hot label matrix `(B,10)`. The implementation avoids allocating it: copy the probabilities, subtract 1 at the true-label entries, then divide by B. This is the ONLY batch averaging in backward propagation.

## 5. Walking backward through the actual code

| Code/equation | Shape | Explanation |
|---|---|---|
| `D2 = (P-Y)/B` | (B,10) | Loss sensitivity to output scores |
| `dW2 = H.T @ D2` | (128,10) | Add each hidden activation times its output sensitivity |
| `db2 = D2.sum(axis=0)` | (10,) | Each output bias affects every row equally |
| `dH = D2 @ W2.T` | (B,128) | Send output sensitivities back to hidden neurons |
| `D1 = dH * (Z1 > 0)` | (B,128) | ReLU passes gradients only where its input is positive |
| `dW1 = X.T @ D1` | (784,128) | Add input pixel times hidden sensitivity |
| `db1 = D1.sum(axis=0)` | (128,) | Add hidden sensitivities across images |

A transpose exchanges rows and columns. H.T has shape `(128,B)`, so multiplying by `(B,10)` sums over images and leaves one gradient for each hidden-to-output connection. W2.T has shape `(10,128)`, so multiplying D2 by it sums over output classes and leaves hidden sensitivities for each image. X.T has shape `(784,B)`, allowing the same batch summation for first-layer weights.

Bias gradients sum because the same bias is reused for all examples: the derivative of each `z_nk` with respect to `b_k` is 1. Do not divide these sums by B again, because D2 already contains that factor.

ReLU's derivative is 1 when z>0 and 0 when z<0. It is undefined at zero; this implementation chooses 0 there. `hidden_gradient *= hidden_logits > 0` applies that mask elementwise.

`loss_and_gradients` computes derivatives without changing parameters. `update` separately performs `parameter -= learning_rate * gradient`. This separation ensures backward propagation uses the same weights that produced the forward values, especially when calculating dH using W2. Every gradient's shape is checked, and non-finite losses, logits, gradients, or updated parameters raise an error.

## 6. A complete small numerical example

Use two input features, two hidden neurons, and two classes to keep the arithmetic visible. The same operations scale to 784,128,10. Let B=1:

    X = [[1, 2]]                    shape (1,2)
    W1 = [[0.1, -0.2], [0.3, 0.4]]  shape (2,2)
    b1 = [0, 0]                     shape (2,)
    W2 = [[0.2, -0.1], [-0.3, 0.4]] shape (2,2)
    b2 = [0, 0]                     shape (2,)
    y = [1]

Forward:

    Z1 = [[0.7, 0.6]]
    H  = [[0.7, 0.6]]
    Z2 = [[-0.04, 0.17]]
    shifted scores = [[-0.21, 0]]
    P ≈ [[0.447692, 0.552308]]
    L = -log(0.552308) ≈ 0.593650

Backward, rounded:

    D2 = [[0.447692, -0.447692]]
    dW2 = [[0.313384, -0.313384],
           [0.268615, -0.268615]]
    db2 = [0.447692, -0.447692]
    dH = [[0.134308, -0.313384]]
    D1 = dH  # both hidden inputs are positive
    dW1 = [[0.134308, -0.313384],
           [0.268615, -0.626769]]
    db1 = [0.134308, -0.313384]

With learning rate 0.1, subtract 0.1 times each gradient:

    W1_new ≈ [[0.086569, -0.168662], [0.273138, 0.462677]]
    b1_new ≈ [-0.013431, 0.031338]
    W2_new ≈ [[0.168662, -0.068662], [-0.326862, 0.426862]]
    b2_new ≈ [-0.044769, 0.044769]

The updated network increases the correct-class probability to about 0.6310 and decreases loss to about 0.46046. `worked_example.py` prints the unrounded calculation so you can compare every array. One useful update does not prove a whole training run will converge.

## 7. Training and checkpoint selection

An epoch is one pass through the training set. A batch is a small group used for one update. A batch size of 128 means about 391 updates per epoch for 50,000 images, including the final partial batch. Smaller batches provide noisier estimates; they also make more updates per epoch.

In `train.py`, `rng.permutation` creates a new random order each epoch. Slicing that order chooses batch indices without changing image-label correspondence. The gradient uses those images only, then `update` changes the weights. After the epoch, `evaluate` measures both training and validation metrics with the current weights. These are full end-of-epoch measurements, not averages of losses from changing intermediate models.

Learning rate controls step size. Too large can make loss jump or diverge; too small can make progress slow. Defaults are 15 epochs, batch size 128, learning rate 0.1, seed 42. A fixed seed controls initialization and shuffling; differing numerical libraries or hardware can still cause small differences.

`history.json` stores the measurements. `curves` plots loss and accuracy. Underfitting means the model still fits training data poorly. Overfitting means training performance improves while validation performance worsens. More epochs do not necessarily improve new-image performance.

The lowest validation-loss checkpoint becomes `model.npz`; the last epoch is also saved as `last_model.npz`. `np.savez_compressed` stores named NumPy arrays with their shapes and dtypes. `np.load(..., allow_pickle=False)` loads plain numerical arrays without Python object unpickling. Saving weights is enough for inference, but this project does not implement exact mid-training resume with RNG state.

## 8. Evaluation and inference

`evaluate.py` loads the selected checkpoint and measures the official test set. It creates a confusion matrix using `np.add.at(matrix, (labels, predictions), 1)`. Rows are actual digits; columns are predictions. Entry (4,9), for example, counts actual 4s classified as 9s. Diagonal entries are correct predictions. CSV and plotted versions are saved.

`predictions.png` shows test images 20–39. `errors.png` shows up to 20 mistakes after index 19; `evaluation.json` records their indices and labels. Inspect the images before explaining a specific mistake. Possible causes include ambiguous handwriting, faint or unusual strokes, and the dense model's limited treatment of spatial structure. A mistake alone does not reveal its exact cause, and a high softmax probability is not a guarantee of correctness or calibrated confidence.

`predict.py` reserves the first 20 test images for its demonstration grid. They were never used in training or validation, although the final evaluation includes them in its aggregate score. They are not a new independent test set. The script loads weights, runs forward propagation, chooses the largest score, and prints probabilities. It never computes gradients or updates weights. That is inference.

This model expects centered, MNIST-like images with light strokes on a dark background. Arbitrary phone photos need additional preprocessing that this demo does not implement.

## 9. Why the checks matter

`verify.py` uses a 4→5→3 float64 network and three input rows. For each parameter θ, it estimates:

    numerical derivative ≈ [L(θ+ε)-L(θ-ε)] / (2ε), ε=0.00001

θ is one weight or bias. The code compares this with the manually calculated gradient for all 43 parameters. It requires hidden inputs to be away from zero and checks that both perturbations preserve the ReLU masks. Otherwise a finite difference could cross a nondifferentiable kink and mislead us. It reports absolute error and scaled error `abs(numeric-analytic)/max(1e-8,abs(numeric)+abs(analytic))`. Absolute tolerance is also used because relative errors are unstable near zero.

This supports the backward implementation at the checked points; it does not prove every possible input works. Probability-sum and shape checks catch dimension and normalization mistakes, but cannot establish useful learning. Extreme-score checks exercise numerical stability, but not every float limit. Non-finite guards stop obvious numerical failures, but finite values can still be wrong.

`overfit.py` makes the full network memorize 32 actual training images. Reaching 100% accuracy with low loss supports the connection between data, gradients, and updates. It says nothing about generalization. The save/reload check requires identical logits and predictions, supporting persistence correctness for that model. The final held-out evaluation addresses generalization on MNIST, not all handwriting in the world.

Run checks before trusting training. A gradient check can pass even if labels are shuffled relative to images; this is why data inspection and tiny-subset memorization are also useful.

## 10. Common problems

| Symptom | Likely checks |
|---|---|
| Download fails | Read the actual error; place all four original gzip or raw IDX files in `data/`, then use `--no-download` |
| Shape mismatch | Confirm batches are rows and use `@`; compare the shape table above |
| Probabilities do not sum to 1 | Sum over axis 1, retaining a column dimension for division |
| Loss NaN or infinite | Normalize pixels, subtract maximum logits, lower learning rate, inspect finite checks |
| Training stays near 10% | Check labels match images, weights are random, ReLU mask is correct, and updates are called |
| Learning unexpectedly slow | Check batch averaging happens exactly once, then try a larger learning rate |
| Training improves but validation worsens | Use the validation-selected model; try fewer epochs |
| Gradient check fails only near zero | Ensure both perturbations stay on the same ReLU branch |
| Saving fails | Check directory permissions and available disk space |
| CPU run unusually slow | Try limiting BLAS threads as shown in README; small matrix products can suffer from excess threading |

## 11. Three experiments

1. Change `--learning-rate` to 0.01 and then 0.5, keeping other settings fixed. Predict slower early learning at 0.01 and larger, possibly unstable jumps at 0.5. Compare validation curves rather than choosing from test scores.
2. Change `--batch-size` to 32 and 512. Predict noisier, more frequent updates at 32 and smoother, fewer updates at 512. With fixed epochs these runs make different numbers of updates, so this is not a controlled comparison of noise alone.
3. Edit the default hidden size in `network.py` from 128 to 32, then 256. Predict lower capacity and faster steps with 32; higher capacity and greater overfitting risk with 256. Keep separate output directories and compare validation loss before final test evaluation.
