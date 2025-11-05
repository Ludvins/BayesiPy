import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .utils.converter import converter
from .utils.linear import BayesLinearMF
from .utils.conv import BayesConv2dMF
from .utils.batchnorm import BayesBatchNorm2dMF

from bayesipy.utils import gaussian_logdensity

class MFVI(torch.nn.Module):
    def __init__(self,  model,likelihood, prior_precision, n_samples, seed, noise_variance = None, y_mean=0, y_std=1) -> None:
        super(MFVI, self).__init__()

        # Get a parameter
        p = list(model.parameters())[0]

        # Store device and dtype
        self.device = p.device
        self.dtype = p.dtype

        # Store model and parameters
        self.mfvi_model = converter(model.cpu(), seed=seed).to(self.device).to(self.dtype)
        self.prior_precision = prior_precision
        self.n_samples = n_samples
        self.likelihood = likelihood
        if self.likelihood == "regression":
            self.log_noise = torch.nn.Parameter(torch.tensor(0.5*np.log(noise_variance), device=self.device, dtype=self.dtype))
        self.y_mean = y_mean
        self.y_std = y_std

    def train(self):
        self.mfvi_model.train()
        self.update_mode("flipout")

    def eval(self):
        self.mfvi_model.eval()
        self.update_mode("reparam")


    def update_mode(self, mode):
        self.mode = mode
        for m in self.mfvi_model.modules():
            if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
                m.update_mode(mode)

    def _prior_regularization(self):
        for name, param in self.mfvi_model.named_parameters():
            if "_psi" in name:
                param.grad.data.add_((param * 2).exp(), alpha=self.alpha).sub_(
                    1.0 / self.num_data
                )
            else:
                param.grad.data.add_(param, alpha=self.alpha)

    def _compute_loss(self, y_pred, targets):
        if self.likelihood == "classification":
            return F.cross_entropy(y_pred, targets)
        elif self.likelihood == "regression":
            noise = self.log_noise.exp()
            gaussian_log_likelihood = gaussian_logdensity(y_pred, noise**2, targets)
            return -gaussian_log_likelihood.mean()

    def fit(self, train_loader, iterations, mean_lr = 1e-4, var_lr = 1e-3, mean_weight_decay = 2e-4, var_weight_decay = 0, momentum=0.9, verbose=False):
        self.train()

        self.num_data = len(train_loader.dataset)
        self.alpha = self.prior_precision / self.num_data

        mus, psis = [], []
        for name, param in self.mfvi_model.named_parameters():
            if "psi" in name:
                psis.append(param)
            else:
                mus.append(param)
        
        optimizer = torch.optim.SGD(
            [
                {"params": mus, "lr": mean_lr, "weight_decay": mean_weight_decay},
                {"params": psis, "lr": var_lr, "weight_decay": var_weight_decay},
            ],
            momentum=momentum,
            nesterov=True,
        )


        # Define the Adam optimizer specifically for 'self.log_noise'
        if self.likelihood == "regression":
            optimizer_adam = torch.optim.Adam(
                [{"params": self.log_noise, "lr": mean_lr}]
            )

        if verbose:
            iters = tqdm(range(iterations), unit=" iteration")
            iters.set_description("Training ")
        else:
            iters = range(iterations)
        data_iter = iter(train_loader)
        losses = []
        for i in iters:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)
            inputs = inputs.to(self.device).to(self.dtype)
            targets = targets.to(self.device)
            y_pred = self.mfvi_model(inputs)

            loss = self._compute_loss(y_pred, targets)
            losses.append(loss.item())
            optimizer.zero_grad()
            
            if self.likelihood == "regression":
                optimizer_adam.zero_grad()

            loss.backward()
            self._prior_regularization()

            if self.likelihood == "regression":
                optimizer_adam.step()
            
            optimizer.step()
        return losses

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = x.to(self.device).to(self.dtype)
        y_preds = [self.mfvi_model(x) for _ in range(self.n_samples)]
        ret = torch.stack(y_preds)

        if self.likelihood == "classification":
            return ret * self.y_std + self.y_mean
        elif self.likelihood == "regression":
            # In this case, return mean and variance of Gaussian mixture of samples
            mean = ret.mean(0) * self.y_std + self.y_mean
            variance = ret.var(0) + self.log_noise.exp()**2 * self.y_std**2
            return mean, variance
            

    def sample(self, x):
        self.mfvi_model.eval()
        self.update_mode("mc")
        with torch.no_grad():
            x = x.to(self.device).to(self.dtype)
            y_preds = [self.mfvi_model(x) for _ in range(self.n_samples)]
            ret = torch.stack(y_preds)


            return ret * self.y_std + self.y_mean
