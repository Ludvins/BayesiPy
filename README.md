# BayesiPy

**BayesiPy** is a Python library for *post-hoc uncertainty estimation* in pre-trained neural networks. In other words, given a deep neural network (DNN) that was trained via standard back-propagation, BayesiPy can enhance it with calibrated confidence estimates (error bars on the predictions) without altering the model’s original accuracy. This is crucial in applications where knowing the model’s uncertainty is as important as the prediction itself (e.g., in medical diagnosis or autonomous driving).

BayesiPy provides a suite of state-of-the-art techniques to quantify uncertainty post-hoc, each balancing fidelity and computational cost in different ways. These include:

- A full range of **Linearized Laplace (LLA)** methods (from [aleximmer/Laplace](https://github.com/aleximmer/Laplace)):  
  - **Full Laplace**  
  - **Layerwise Laplace**  
  - **Subnetwork Laplace**  
  - **Last-Layer Laplace** 
  - Various curvature approximations (Exact, Kron, Diagonal, KFAC, etc.)
- **Empirical Last-Layer Laplace Approximation (ELLA)**
- **Variational Last-Layer Laplace Approximation (VaLLA)**
- **Mean-Field Variational Inference (MFVI)**
- **Spectral-normalized Gaussian Process (SNGP)**
- **Fixed-Mean Gaussian Process (FM-GP)**

In the sections below, we explain each technique and cite their original references. We then compare the methods, highlighting trade-offs in calibration, scalability, and computational cost. Finally, we show how to install BayesiPy, give a simple usage example, and outline how you can contribute to the project.

---

## Uncertainty Estimation Techniques in BayesiPy

### 1. Linearized Laplace Methods

BayesiPy integrates the full suite of [Linearized Laplace approximations](https://github.com/aleximmer/Laplace) proposed by Immer et al. (2021) and subsequent works. The idea is to treat a pre-trained neural network $f(\cdot; \theta)$ at its MAP estimate $\theta_\text{MAP}$ and approximate the posterior over $\theta$ locally by a Gaussian whose mean is $\theta_\text{MAP}$ and whose covariance is derived from the (generalized) Gauss-Newton or Hessian of the negative log-likelihood. By “linearizing” the network’s parameters around the MAP solution, we obtain:

- **Full Laplace**: Uses the full Hessian matrix. Most accurate but extremely memory-intensive for large models.  
- **Layerwise Laplace**: Applies the Laplace approximation separately (or in block structures) for different layers. Balances accuracy and memory usage.  
- **Subnetwork Laplace**: Selects a subset of the network’s parameters (e.g., a subnetwork or certain layers) for the Laplace approximation, reducing complexity.  
- **Last-Layer Laplace (LLA)**: Approximates only the last layer’s weights by a Gaussian, holding the rest of the network fixed as a deterministic feature extractor.  
- **Curvature Approximations**: You can choose from diagonal, KFAC, Kron, or exact Hessian approximations to balance computational cost with approximation accuracy.

**References**:  
- Immer et al. (2021) – *“Improving Predictions of Neural Networks via Monte Carlo Methods, the Laplace Approximation, and Bayesian Neural Networks.”* [[arXiv](https://arxiv.org/abs/2106.14806)]  
- Kristiadi et al. (2020) – *“Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks.”* [[ICML](http://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf)] (for the original LLA insight)

**Empirical Last-Layer Laplace Approximation (ELLA)** specifically accelerates and approximates the last-layer Laplace by using subsets of data, Nyström approximations, or other low-rank techniques to handle larger models and datasets. While it focuses on the last layer, its emphasis on scalability and memory efficiency makes it appealing for modern architectures.

- *Source:* Deng et al. (2022) proposed an accelerated linearized Laplace method (ELLA) with a Nyström approximation to the network’s tangent kernel for improved scalability.  
  [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d826cc6b2b713e3b9aad0b67c3b0f79-Abstract-Conference.html)]

**Variational Last-Layer Laplace Approximation (VaLLA)** is another variant of LLA that leverages sparse Gaussian processes in function space. Rather than computing Hessians directly, VaLLA uses variational inference to fit a GP whose mean is anchored at the DNN output. In practice, VaLLA can yield high-quality calibration with sub-linear complexity in the dataset size.

- *Source:* Ortega et al. (2024a) – *“Variational Linearized Laplace Approximation for Bayesian Deep Learning.”* [[ICML](https://proceedings.mlr.press/v235/ortega24a.html)]

---

### 2. Mean-Field Variational Inference (MFVI)

**Mean-Field Variational Inference (MFVI)** (a.k.a. “Bayes by Backprop”) is a classic approach where we assume a factorized Gaussian over neural network weights and optimize its mean/variance parameters via the ELBO. While it can be applied to all layers, it is also possible to do post-hoc MFVI on only the final layers. This method can correct overconfidence, but often underestimates uncertainty due to the independence assumption.

- *Source:* Blundell et al. (2015) – *“Weight Uncertainty in Neural Networks.”* [[ICML](http://proceedings.mlr.press/v37/blundell15.html)]

---

### 3. Spectral-normalized Gaussian Process (SNGP)

**Spectral-normalized Gaussian Process (SNGP)** integrates a GP layer at the network’s output and enforces a distance-preserving feature space via spectral normalization. This makes the network’s predictions *distance-aware*, which helps with out-of-distribution (OOD) detection. SNGP typically involves a re-training or fine-tuning step to incorporate the spectral normalization in earlier layers.

- *Source:* Liu et al. (2020) – *“Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.”* [[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/4089e94f74c885f1bfb1e77f2711b67b-Abstract.html)]

---

### 4. Fixed-Mean Gaussian Process (FM-GP)

**Fixed-Mean Gaussian Process (FM-GP)** is a method where we overlay a GP whose mean is fixed to the pre-trained DNN output. This GP focuses on learning the uncertainty (variance) around the DNN’s predictions. FM-GP can be trained post-hoc with a sparse variational approach, scaling to large datasets and architectures. Empirically, it often outperforms other methods in calibration quality with moderate computational overhead.

- *Source:* Ortega et al. (2024b) – *“Fixed-Mean Gaussian Processes for Post-hoc Bayesian Deep Learning.”* [[arXiv](https://arxiv.org/abs/2412.04177)]

---

## Comparison of Techniques

Below is a brief summary comparing these techniques, with insights drawn from both the original Laplace approximation library [Immer et al., 2021] and recent works like [Ortega et al., 2024b]:

| **Method**            | **Pros**                                                                                                         | **Cons**                                                                                                                                                                                                                                                                         | **Use Case**                                                                                                                                              |
|-----------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Full Laplace**      | Most faithful local Gaussian approximation to MAP parameters; captures correlations in all parameters            | Extremely memory- and computation-heavy; not feasible for large networks                                                                                                                                                                                                         | Good for small-to-medium networks or when maximizing fidelity is paramount                                                                                 |
| **Layerwise/Subnet**  | Balances coverage (whole network or a subset) with computational feasibility; can approximate key layers          | More complex to configure (which layers to include? which blocks?). Some approximation needed for Hessian.                                                                                                                                                                       | For medium-to-large networks if you need more coverage than last-layer but can’t afford full Laplace                                                      |
| **Last-Layer Laplace** (LLA) | Very fast post-hoc correction; no retraining, just Hessian-based Gaussian around final layer                | Focuses only on last-layer uncertainty; can miss uncertainties originating in earlier layers                                                                                                                                                | A quick Bayesian “upgrade” that often fixes overconfidence on moderate tasks                                                                              |
| **ELLA**              | Scalable variant of last-layer Laplace using low-rank or Nyström approximations                                  | Still limited to last-layer uncertainty; hyperparameters for kernel approximation need tuning                                                                                                                                                                                   | Large-scale scenarios where even standard LLA is too costly                                                                                                |
| **VaLLA**             | Excellent calibration (function-space GP perspective) with sub-linear complexity in data size                    | Requires iterative variational optimization; can be slower to converge; more complicated inference step                                                                                                                                                                         | High-fidelity uncertainty for large datasets or critical applications                                                                                     |
| **MFVI**              | Classic fully Bayesian approach over weights; easy to implement (Bayes by Backprop)                              | Mean-field assumption often underestimates uncertainty; can be very slow or memory-heavy for large networks                                                                                                                                                                     | For those wanting a “full BNN” approach or partial Bayesian layers with factorized posteriors                                                             |
| **SNGP**              | Distance-aware; single forward-pass at inference; strong OOD detection                                           | Typically requires training from scratch or at least heavy fine-tuning with spectral normalization; not purely post-hoc                                                                                                                  | Production-friendly if you can integrate spectral norms and a GP head early on                                                                            |
| **FM-GP**             | High-quality calibration; scalable to large data; easy to wrap any pre-trained model (fixed mean)                | Extra training to fit the GP’s variational parameters; number of inducing points is a hyperparameter that can affect memory usage                                                                                                                                               | Post-hoc method offering advanced Bayesian-quality uncertainty for large-scale tasks without heavy retraining                                             |

---

## Installation

BayesiPy is built with Python (>=3.8) and PyTorch as the core deep learning library. You’ll need to have PyTorch installed (see [pytorch.org](https://pytorch.org) for instructions) plus some additional packages like [BackPACK](https://github.com/f-dangel/backpack) for efficient Hessian approximations in Laplace.

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ludvins/BayesiPy.git
   cd BayesiPy
   ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

This will install PyTorch, Backpack, and other required libraries. It’s recommended to use a virtual environment (e.g., venv or Conda) to isolate your setup.

## Usage Example

Below is a short snippet demonstrating how to apply a Fixed-Mean GP (FMGP) to a pre-trained regression model. For other methods, the usage is similar, but you would import from the relevant module (e.g., bayesipy.laplace for the linearized Laplace methods).

```python
import copy
import torch
import numpy as np
from bayesipy.fmgp import FMGP

# Suppose 'f' is your pre-trained PyTorch model, e.g., an nn.Module for regression.
fmgp = FMGP(
    model=copy.deepcopy(f),        # copy of the MAP-trained model
    likelihood="regression",       # 'regression' or 'classification'
    kernel="RBF",                  # kernel for the GP (e.g., Radial Basis Function)
    inducing_locations="kmeans",   # how to initialize inducing points (k-means on the training data)
    num_inducing=50,               # number of inducing points (scales the GP complexity)
    noise_variance=np.exp(-5),     # initial noise variance for regression
    subrogate_regularizer=True,    # stability trick during training
    y_mean=0.0,                    # target mean if data was normalized
    y_std=1.0                      # target std if data was normalized
)

# Train the FMGP to learn the GP variance parameters (the base model 'f' is not changed):
loss = fmgp.fit(
    iterations=3000,
    lr=1e-3,
    train_loader=train_loader,  # PyTorch DataLoader with (X, y)
    verbose=True
)
print("Finished FMGP training with final loss:", loss)

# Get predictions with uncertainty:
X_test = ...  # some test inputs (NumPy array or torch.Tensor)
mean_pred, var_pred = fmgp.predict(torch.tensor(X_test, dtype=torch.float32))
print("Predictive mean:", mean_pred)
print("Predictive variance:", var_pred)
```

For Linearized Laplace methods (including Full, Subnetwork, Last-layer, etc.), you would typically import from bayesipy.laplace. For instance:

```python
from bayesipy.laplace import Laplace

# Suppose 'f' is your pre-trained model.
laplace_model = Laplace(
    model=f,
    approximation="full",    # 'full', 'kron', 'diag', 'kfac', etc.
    subset_of_weights="all", # 'all', 'last_layer', 'subnetwork', etc.
    likelihood="classification"
)

laplace_model.fit(train_loader)
preds, preds_std = laplace_model.predict(test_loader)
```

Consult the repository’s examples folder for more usage details and advanced configurations.
## Contributing

Contributions are welcome! If you’d like to improve BayesiPy, follow these steps:

1. Open an Issue: Report bugs, request new features, or ask questions. We track all changes and discussions in GitHub issues.

2. Fork the Repo & Create a Branch: For code changes, fork BayesiPy and create a feature branch for your work.

3. Pull Request: When you’re ready, open a pull request describing your changes. Make sure to include relevant tests or update examples/documentation. Please follow PEP8 style guidelines.

4. Testing: We encourage adding or updating tests in tests/. Ensure your changes don’t break existing functionality.

5. Discussion: For major proposals, start a discussion via an issue so we can share feedback.

By contributing, you help advance accessible Bayesian deep learning methods for the community.

## Bibliography

1. **Immer et al. (2021)** – *“Improving Predictions of Neural Networks via Monte Carlo Methods, the Laplace Approximation, and Bayesian Neural Networks.”*  
   [[arXiv:2106.14806]](https://arxiv.org/abs/2106.14806)

2. **Kristiadi et al. (2020)** – *“Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks.”*  
   ICML 37:5436–5446, [Link](http://proceedings.mlr.press/v119/kristiadi20a/kristiadi20a.pdf)

3. **Deng et al. (2022)** – *“Accelerated Linearized Laplace Approximation for Bayesian Deep Learning.”*  
   NeurIPS 35, [Link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d826cc6b2b713e3b9aad0b67c3b0f79-Abstract-Conference.html)

4. **Ortega et al. (2024a)** – *“Variational Linearized Laplace Approximation for Bayesian Deep Learning.”*  
   ICML 41, [Link](https://proceedings.mlr.press/v235/ortega24a.html)

5. **Blundell et al. (2015)** – *“Weight Uncertainty in Neural Networks.”*  
   ICML 32, [Link](http://proceedings.mlr.press/v37/blundell15.html)

6. **Liu et al. (2020)** – *“Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.”*  
   NeurIPS 33, [Link](https://proceedings.neurips.cc/paper/2020/hash/4089e94f74c885f1bfb1e77f2711b67b-Abstract.html)

7. **Ortega et al. (2024b)** – *“Fixed-Mean Gaussian Processes for Post-hoc Bayesian Deep Learning.”*  
   [[arXiv:2412.04177]](https://arxiv.org/abs/2412.04177)
