# Measured execution results

The actual run used the original MNIST dataset, seed 42, float32 training, architecture 784→128→10, batch size 128, learning rate 0.1, and 15 epochs. NumPy 2.3.5 and Matplotlib 3.10.8 were installed. BLAS used one thread. These are observed results, not promised performance on other runs.

| Measurement | Observed value |
|---|---:|
| Training images | 50,000 |
| Validation images | 10,000 |
| Test images | 10,000 |
| Pixel range | 0.0–1.0 |
| Labels | 0–9 |
| Final training loss | 0.0714 |
| Final training accuracy | 98.13% |
| Selected validation loss | 0.1069 |
| Selected validation accuracy | 96.90% |
| Selected epoch (lowest validation loss) | 15 |
| Test loss | 0.1042788004 |
| Test accuracy | 97.08% |
| Correct test predictions | 9,708 |
| Incorrect test predictions | 292 |
| Checked finite-difference derivatives | 43 / 43 passed |
| Largest absolute derivative error | 1.84044e-11 |
| Largest scaled derivative error | 1.55389e-9 |
| Smallest absolute checked hidden preactivation | 0.0492275 |
| Tiny-subset initial loss | 2.420897 |
| Tiny-subset final loss (1,000 updates) | 0.001847 |
| Tiny-subset accuracy | 100% (32/32) |

Probability sums, shapes, finite extreme-logit loss, and identical logits/predictions after saving and loading passed. The training run completed with finite-value guards enabled. The included run_history.json and test_evaluation.json record the original measurements. Run the scripts to generate fresh weights and plots in results/.

## Interpreting the figures

Training and validation loss decrease substantially. The modest gap between their accuracies suggests some generalization gap, but validation loss still improved at the final epoch. We did not tune further using the test score. A higher test accuracy than validation accuracy is possible because these are different finite samples.

In the inspected errors.png grid:

- Test image 259 is a 6 predicted as 0. Its prominent closed loop resembles a zero, although its upper stroke distinguishes it. The model may have underweighted that stroke.
- Test image 707 is a 4 predicted as 9. The narrow upper structure approaches a closed loop, a feature shared with some handwritten nines.
- Test image 591 is an 8 predicted as 3. Its upper loop is faint and partly open, which may make its two-curve structure resemble a three.

These are plausible visual explanations, not proven causal attributions. The grid also contains mistakes that are quite readable to a person. A dense model can miss distinctions that humans find easy.

The prediction demo got 19/20 correct. At test index 8 it predicted 6 for a 5 with probability about 0.952. This is a concrete example of a confident mistake; softmax outputs should not be read as guaranteed reliability.

## Verification limits

Finite differences check a tiny network at chosen smooth points, not all possible parameters. Tiny-subset memorization supports the end-to-end update pipeline, not generalization. Save/load equality checks persistence, not prediction quality. The held-out score measures MNIST performance, not phone photos or every handwriting style. Fixed seeds improve reproducibility but do not guarantee bitwise-identical training across hardware and BLAS implementations.
