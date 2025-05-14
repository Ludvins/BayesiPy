#!/usr/bin/env python3
import numpy as np
import scipy as sp
import torch
from properscoring import crps_gaussian
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from bayesipy.utils import gaussian_logdensity, safe_cholesky


class Metrics:
    def __init__(self):
        """Defines a class that encapsulates all considered metrics."""
        # Device and dtype are initialized with the first "update" call.
        self._set_device_dtype = True

    def set_device_dtype(self, device, dtype):
        """Sets the Metrics device and dtype."""
        self.device = device
        self.dtype = dtype

    def reset(self):
        """Ressets all the metrics to zero."""
        self.n_data = 0

    def update(self, y, Fmean, Fvar):
        """Updates the considered metrics. If this is the first call to the
        method, device and dtype are initialized taking that of "Fmean".

        Parameters
        ----------
        y : torch.tensor
            Contains the target values
        Fmean : torch.tensor
            Constains the posterior mean
        Fvar : torch.tensor
            Constains the posterior covariances
        """
        # If this is the first call to "update", set device and dtype.
        if self._set_device_dtype:
            self.set_device_dtype(Fmean.device, Fmean.dtype)
            # Create new metric variables
            self.reset()
            # Set to False
            self._set_device_dtype = False
        # Update total number of "seen" points. Used to average.
        self.n_data += y.shape[0]


class Regression(Metrics):
    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()

        # Mean Squared Error
        self.mse = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # Continuous Ranked Probability Score
        self.crps = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # Negative Log-Likelihood
        self.nll = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Centered Quantiles from 10% to 90%
        self.q_10 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_20 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_30 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_40 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_50 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_60 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_70 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_80 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.q_90 = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def update(self, y, Fmean, Fvar):
        """Updates the considered metrics. If this is the first call to the
        method, device and dtype are initialized taking that of "Fmean".

        Parameters
        ----------
        y : torch.tensor
            Contains the target values
        Fmean : torch.tensor
            Constains the posterior mean
        Fvar : torch.tensor
            Constains the posterior covariances
        """
        super().update(y, Fmean, Fvar)
        # For regression NLL is Gaussian log density.
        self.nll -= torch.sum(
            gaussian_logdensity(Fmean.squeeze(), Fvar.squeeze(), y.squeeze())
        )
        self.mse += torch.nn.functional.mse_loss(Fmean, y, reduction="sum")

        # Compute standard deviation instead of variance.
        Fstd = torch.sqrt(Fvar)

        # Compute the rest of metrics
        self.crps += self.compute_crps(y, Fmean, Fstd)
        self.q_10 += self.compute_quantile_error(y, Fmean, Fstd, 0.10)
        self.q_20 += self.compute_quantile_error(y, Fmean, Fstd, 0.20)
        self.q_30 += self.compute_quantile_error(y, Fmean, Fstd, 0.30)
        self.q_40 += self.compute_quantile_error(y, Fmean, Fstd, 0.40)
        self.q_50 += self.compute_quantile_error(y, Fmean, Fstd, 0.50)
        self.q_60 += self.compute_quantile_error(y, Fmean, Fstd, 0.60)
        self.q_70 += self.compute_quantile_error(y, Fmean, Fstd, 0.70)
        self.q_80 += self.compute_quantile_error(y, Fmean, Fstd, 0.80)
        self.q_90 += self.compute_quantile_error(y, Fmean, Fstd, 0.90)

    def compute_quantile_error(self, y, Fmean, Fstd, alpha):
        """Computes the ammount of target values inside the centered quantile
        of alpha percent of the density. That is target values inside:

            (F_mean - d * F_std, F_mean + d * F_std)

        where d is given by \(\alpha\).

        Parameters
        ----------
        y : torch.tensor
            Contains the target values
        Fmean : torch.tensor
            Constains the posterior mean
        Fstd : torch.tensor
            Constains the posterior standard deviation

        Returns
        -------
        inside : int
            The ammount of target values in

        """
        # Compute "d"
        deviation = norm.ppf(0.5 + alpha / 2)
        # Compute upper and lower itnerval values
        upper = Fmean + deviation * Fstd
        lower = Fmean - deviation * Fstd
        # Create True-False mask of tarets inside
        inside = ((y < upper) * (y > lower)).to(self.dtype)

        # Aggregate True values.
        return torch.sum(inside)

    def compute_crps(self, y, mean_pred, std_pred):
        crps = crps_gaussian(
            y.detach().cpu(), mean_pred.detach().cpu(), std_pred.detach().cpu()
        )
        return np.sum(crps)

    def get_dict(self):
        """Scale metrics and return them in a dictionary."""
        # Scale Quantiles from ammount to proportion
        q_10 = self.q_10.item() / self.n_data
        q_20 = self.q_20.item() / self.n_data
        q_30 = self.q_30.item() / self.n_data
        q_40 = self.q_40.item() / self.n_data
        q_50 = self.q_50.item() / self.n_data
        q_60 = self.q_60.item() / self.n_data
        q_70 = self.q_70.item() / self.n_data
        q_80 = self.q_80.item() / self.n_data
        q_90 = self.q_90.item() / self.n_data
        # Create an array with the absolute value of the differences
        # between the estimated quantiles and the optimal ones.
        #  q_0 and q_100 are 0 and 1 by definition for any model.
        Q = np.array(
            [
                0.0,
                np.abs(q_10 - 0.1),
                np.abs(q_20 - 0.2),
                np.abs(q_30 - 0.3),
                np.abs(q_40 - 0.4),
                np.abs(q_50 - 0.5),
                np.abs(q_60 - 0.6),
                np.abs(q_70 - 0.7),
                np.abs(q_80 - 0.8),
                np.abs(q_90 - 0.9),
                0.0,
            ]
        ).T
        # Compute the CQM metric as the integral of this function
        CQM = sp.integrate.trapezoid(x=np.arange(0, 1.1, 0.1), y=Q)
        # Return scaled metrics.
        return {
            "RMSE": np.sqrt(self.mse.item() / self.n_data),
            "NLL": float(self.nll.item()) / self.n_data,
            "Q-10": float(q_10),
            "Q-20": float(q_20),
            "Q-30": float(q_30),
            "Q-40": float(q_40),
            "Q-50": float(q_50),
            "Q-60": float(q_60),
            "Q-70": float(q_70),
            "Q-80": float(q_80),
            "Q-90": float(q_90),
            "CQM": float(CQM),
            "CRPS": float(self.crps) / self.n_data,
        }


class ECE(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()

        self.confidences = []
        self.predictions = []
        self.labels = []

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]

    def reset(self):
        self.confidences = []
        self.predictions = []
        self.labels = []

    def update(self, labels, F):
        probs = F.softmax(-1)
        conf, pred = torch.max(probs, -1)
        self.confidences.append(conf)
        self.predictions.append(pred)
        self.labels.append(labels.squeeze(-1))

    def compute(self):
        self.predictions = torch.cat(self.predictions, -1)
        self.labels = torch.cat(self.labels, -1)
        self.confidences = torch.cat(self.confidences, -1)

        accuracies = self.predictions.eq(self.labels)
        ece = torch.zeros(1, device=self.confidences.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = self.confidences.gt(bin_lower.item()) * self.confidences.le(
                bin_upper.item()
            )
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = self.confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class SoftmaxClassification(Metrics):
    def set_device_dtype(self, device, dtype):
        """Sets the Metrics device and dtype."""
        self.generator = torch.Generator(device=device)
        super().set_device_dtype(device, dtype)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.nll = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.acc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.brier = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.ece = ECE()

        self.generator.manual_seed(2147483647)

    def update(self, y, Fmean, Fvar):
        """Updates the considered metrics. If this is the first call to the
        method, device and dtype are initialized taking that of "Fmean".

        Parameters
        ----------
        y : torch.tensor
            Contains the target values
        Fmean : torch.tensor
            Constains the posterior mean
        Fvar : torch.tensor
            Constains the posterior covariances
        """
        super().update(y, Fmean, Fvar)

        # Get samples from the posterior
        chol = safe_cholesky(Fvar)
        z = torch.randn(
            2048,
            Fmean.shape[0],
            Fvar.shape[-1],
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        # Use re-parameterization Trick
        logit_samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
        # Get probailities
        prob_samples = logit_samples.softmax(-1)
        # Average and compute logarithm to scale to logit again
        logits = prob_samples.mean(0).log()
        # Update metrics
        self.acc += self.compute_acc(y, logits)
        self.nll += self.compute_nll(y, logits)
        self.brier += self.compute_brier(y, logits)
        self.ece.update(y, logits)

    def compute_brier(self, y, F):
        # Compute Probabilities
        probs = F.softmax(-1)
        # Compute one_hot encoding of the targets
        oh_on = torch.nn.functional.one_hot(
            y.to(torch.long).squeeze(), num_classes=probs.shape[-1]
        )
        # Compute distance between probabilities and one hot encoding
        dist = (probs - oh_on) ** 2
        return torch.sum(dist)

    def compute_acc(self, y, F):
        return (torch.argmax(F, -1) == y.flatten()).float().sum()

    def compute_nll(self, y, F):
        return torch.nn.functional.cross_entropy(
            F, y.to(torch.long).squeeze(-1), reduction="sum"
        )

    def get_dict(self):
        return {
            "NLL": float(self.nll.item()) / self.n_data,
            "ACC": float(self.acc.item()) / self.n_data,
            "ECE": float(self.ece.compute()),
            "BRIER": float(self.brier.item()) / self.n_data,
        }



class SoftmaxClassificationSamples(Metrics):
    def set_device_dtype(self, device, dtype):
        """Sets the Metrics device and dtype."""
        self.generator = torch.Generator(device=device)
        super().set_device_dtype(device, dtype)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.nll = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.acc = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.brier = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.ece = ECE()

        self.generator.manual_seed(2147483647)

    def update(self, y, F):
        """Updates the considered metrics. If this is the first call to the
        method, device and dtype are initialized taking that of "Fmean".

        Parameters
        ----------
        y : torch.tensor
            Contains the target values
        Fmean : torch.tensor
            Constains the posterior mean
        Fvar : torch.tensor
            Constains the posterior covariances
        """
        super().update(y, F, None)
        # Get probailities
        prob_samples = F.softmax(-1)
        # Average and compute logarithm to scale to logit again
        logits = prob_samples.mean(0).log()
        # Update metrics
        self.acc += self.compute_acc(y, logits)
        self.nll += self.compute_nll(y, logits)
        self.brier += self.compute_brier(y, logits)
        self.ece.update(y, logits)

    def compute_brier(self, y, F):
        # Compute Probabilities
        probs = F.softmax(-1)
        # Compute one_hot encoding of the targets
        oh_on = torch.nn.functional.one_hot(
            y.to(torch.long).squeeze(), num_classes=probs.shape[-1]
        )
        # Compute distance between probabilities and one hot encoding
        dist = (probs - oh_on) ** 2
        return torch.sum(dist)

    def compute_acc(self, y, F):
        return (torch.argmax(F, -1) == y.flatten()).float().sum()

    def compute_nll(self, y, F):
        return torch.nn.functional.cross_entropy(
            F, y.to(torch.long).squeeze(-1), reduction="sum"
        )

    def get_dict(self):
        return {
            "NLL": float(self.nll.item()) / self.n_data,
            "ACC": float(self.acc.item()) / self.n_data,
            "ECE": float(self.ece.compute()),
            "BRIER": float(self.brier.item()) / self.n_data,
        }

class OOD(Metrics):
    def set_device_dtype(self, device, dtype):
        """Sets the Metrics device and dtype."""
        self.generator = torch.Generator(device=device)
        super().set_device_dtype(device, dtype)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.labels = []
        self.preds = []
        self.generator.manual_seed(2147483647)

    def update(self, y, Fmean, Fvar):
        """Updates all the metrics given the results in the parameters.

        Parameters
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
            Contains the loss value for the given batch.
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
        """
        super().update(y, Fmean, Fvar)

        chol = safe_cholesky(Fvar)
        z = torch.randn(
            2048,
            Fmean.shape[0],
            Fvar.shape[-1],
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)

        # Compute probabilities
        probs = samples.softmax(-1)
        # Average on sample dimension
        probs = probs.mean(0)
        # Compute Entropy
        H = -torch.sum(probs * probs.log(), -1)
        # Store Monte-Carlo entropy
        self.preds.append(H)

        # Store labels
        self.labels.append(y)

    def get_dict(self):
        return {
            "labels": torch.cat(self.labels).cpu().numpy(),
            "preds": torch.cat(self.preds).cpu().numpy(),
        }

class OOD_Samples(Metrics):
    def set_device_dtype(self, device, dtype):
        """Sets the Metrics device and dtype."""
        self.generator = torch.Generator(device=device)
        super().set_device_dtype(device, dtype)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.labels = []
        self.preds = []
        self.generator.manual_seed(2147483647)

    def update(self, y, F):
        """Updates all the metrics given the results in the parameters.

        Parameters
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
            Contains the loss value for the given batch.
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
        """
        super().update(y, F, None)

        # Compute probabilities
        probs = F.softmax(-1)
        # Average on sample dimension
        probs = probs.mean(0)
        # Compute Entropy
        H = -torch.sum(probs * probs.log(), -1)
        # Store Monte-Carlo entropy
        self.preds.append(H)

        # Store labels
        self.labels.append(y)

    def get_dict(self):
        return {
            "labels": torch.cat(self.labels).cpu().numpy(),
            "preds": torch.cat(self.preds).cpu().numpy(),
        }

def score(model, generator, metrics_cls, verbose=False):
    """
    Evaluates the given model using the arguments provided.

    Parameters
    ---------
    model : torch.nn.Module
        Torch model to train.
    generator : iterable
        Must return batches of pairs corresponding to the
        given inputs and target values.
    metrics_cls : bayesipy.utils.Metrics class
        The class to use of the metrics to use.

    Returns
    -------
    metrics : dictionary
        Contains pairs of (metric, value) averaged over the number of
        batches.
    """
    # Set model in evaluation mode
    # Initialize metrics

    metrics = metrics_cls()

    if verbose:
        # Initialize TQDM bar
        iters = tqdm(range(len(generator)), unit="iteration")
        iters.set_description("Evaluating ")
    else:
        iters = range(len(generator))
    data_iter = iter(generator)

    with torch.no_grad():
        # Batches evaluation
        for _ in iters:
            inputs, targets = next(data_iter)
            if not isinstance(inputs, list):
                inputs = inputs.to(model.device).to(model.dtype)
            targets = targets.to(model.device).to(model.dtype)
            Fmean, Fvar = model.predict(inputs)
            metrics.update(targets, Fmean, Fvar)
    # Return metrics as a dictionary
    return metrics.get_dict()



def score_samples(model, generator, metrics_cls, verbose=False):
    """
    Evaluates the given model using the arguments provided.

    Parameters
    ---------
    model : torch.nn.Module
        Torch model to train.
    generator : iterable
        Must return batches of pairs corresponding to the
        given inputs and target values.
    metrics_cls : bayesipy.utils.Metrics class
        The class to use of the metrics to use.

    Returns
    -------
    metrics : dictionary
        Contains pairs of (metric, value) averaged over the number of
        batches.
    """
    # Set model in evaluation mode
    # Initialize metrics

    metrics = metrics_cls()

    if verbose:
        # Initialize TQDM bar
        iters = tqdm(range(len(generator)), unit="iteration")
        iters.set_description("Evaluating ")
    else:
        iters = range(len(generator))
    data_iter = iter(generator)

    with torch.no_grad():
        # Batches evaluation
        for _ in iters:
            inputs, targets = next(data_iter)
            if not isinstance(inputs, list):
                inputs = inputs.to(model.device).to(model.dtype)
            targets = targets.to(model.device).to(model.dtype)
            Fsamples = model.predict(inputs)
            metrics.update(targets, Fsamples)
    # Return metrics as a dictionary
    return metrics.get_dict()
