# Baseline for Uncertainty Estimation (On going)

## Supports
- Gaussian Process Classification (1D toy example)
    - [Gaussian Process for Machine Learning] 
- Deep Deterministic Uncertainty: A Simple Baseline (DDU)
    - not with residual connection and spectral normalization
    - [Paper](https://arxiv.org/abs/2102.11582)
- Mixture Logit Network
    - [Paper](https://arxiv.org/abs/2111.01632)
- Maxsoftmax, Softmax Entropy
    - [Paper](https://arxiv.org/abs/1610.02136)
- Mahalanobis Distance from feature space
    - [Paper](https://arxiv.org/abs/2106.09022)
- Dropout Uncertainty
    - [Paper](https://www.nature.com/articles/s41598-017-17876-z)
- Outlier Exposure
    - [Paper](https://arxiv.org/pdf/1812.04606)
- Energy-based OOD
    - [Paper](https://arxiv.org/pdf/2010.03759)

### Out-of-Distribution Detection

In-Distribution Dataset: MNIST
Near Out-of-Distribution: EMNIST
Far Out-of-Distribution: FMNIST

### Calibration
- Histogram Binning
- Expected Calibration Error

### TODO
- SNGP
- Virtual Outlier Exposure