from __future__ import annotations

import argparse
import copy
import sys
import time             
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------#
# Local modules                                                                #
# -----------------------------------------------------------------------------#
sys.path.append(".")
from bayesipy.laplace import Laplace         
from bayesipy.utils import assert_reproducibility 
from bayesipy.utils.datasets import SortedAirline_Dataset, Year_Dataset, Taxi_Dataset
from bayesipy.utils.metrics import Regression
from bayesipy.utils.pretrained_models import Airline_MLP, Year_MLP, Taxi_MLP    

# -----------------------------------------------------------------------------#
# Argument parsing                                                             #
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="CIFAR-10 Laplace experiment (timed)")

# *Experiment control
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--dataset", type=str, default="airline",
                    choices=["airline", "year", "taxi"],
                    help="Dataset to use for the experiment. "
                         "Options: 'airline', 'year', 'taxi'.")

# *Hardware / I/O
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--results_dir", type=str, default="results")


# *Optimisation / Laplace hyper-parameters
parser.add_argument(
    "--prior_opt_lr",
    type=float,
    default=0.1,
    help="Learning-rate used for prior precision optimisation",
)
parser.add_argument(
    "--prior_opt_iters",
    type=int,
    default=1_000,
    help="Iterations for prior precision optimisation",
)
parser.add_argument("--optimize_hyper_parameters", dest="optimize_hyper_parameters",
                     action="store_true")
parser.add_argument("--no_optimize_hyper_parameters", dest="optimize_hyper_parameters",
                     action="store_false")

parser.set_defaults(optimize_hyper_parameters=True)



ARGS = parser.parse_args()

# -----------------------------------------------------------------------------#
# Derived constants                                                            #
# -----------------------------------------------------------------------------#
DTYPE = {"float32": torch.float32, "float64": torch.float64}[ARGS.dtype]
DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
          if ARGS.device == "auto" else torch.device(ARGS.device))

RESULTS_DIR = Path(ARGS.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert_reproducibility(ARGS.seed)


# -----------------------------------------------------------------------------#
# Data                                                                         #
# -----------------------------------------------------------------------------#
if ARGS.dataset == "airline":
   ds = SortedAirline_Dataset()
elif ARGS.dataset == "year":
   ds = Year_Dataset()
elif ARGS.dataset == "taxi":
   ds = Taxi_Dataset()
else:
   raise ValueError(f"Unknown dataset: {ARGS.dataset}")
train_ds, test_ds = ds.train_test_splits()
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=ARGS.batch_size)


backbone = torch.nn.Sequential(
    nn.Linear(ds.input_dim, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, ds.output_dim)).to(DTYPE).to(DEVICE)

backbone_path = RESULTS_DIR / f"{ARGS.dataset}_backbone.pth"
# Load or initialize the backbone model
if backbone_path.exists():
    print(f"Loading pre-trained backbone from {backbone_path}")
    backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
    backbone.eval()


# Compute test rmse
with torch.no_grad():
    preds = []
    targets = []
    for x, y in test_loader:
        x = x.to(DEVICE).to(DTYPE)
        targets.append(y.cpu().numpy())
        pred = backbone(x)
        preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    test_rmse = np.sqrt(np.mean((preds - targets) ** 2))
print(f"Test RMSE of the backbone model: {test_rmse:.4f}")


lla = Laplace(model=backbone.to(DEVICE),
                  likelihood="regression",
                  y_mean=ds.y_mean,
                  y_std=ds.y_std)

# -----------------------------  TRAIN  (timed)  ------------------------------#
train_start = time.perf_counter()                    
lla.fit(
    train_loader=train_loader,
    progress_bar=True,
)

if ARGS.optimize_hyper_parameters:


    log_sigma = torch.zeros(1, requires_grad=True)
    log_prior = torch.zeros(1, requires_grad=True)

    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

    for i in range(ARGS.prior_opt_iters):
        hyper_optimizer.zero_grad()
        neg_marglik = -lla.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    prior_precision = log_prior.exp().item()
    sigma_noise = log_sigma.exp().item()
    print("Prior precision:", prior_precision)
    print("Noise sigma:", sigma_noise)
train_time = time.perf_counter() - train_start      





# -----------------------------  EVALUATE (timed) -----------------------------#
metrics_lla_test = Regression()
eval_start = time.perf_counter()                   

with torch.no_grad():
    for x, y in test_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        var = torch.diagonal(var).squeeze().unsqueeze(-1)
        metrics_lla_test.update(y.to(DEVICE), mean, var)
metrics_lla_test = metrics_lla_test.get_dict()

print("Test metrics:", metrics_lla_test)

eval_time = time.perf_counter() - eval_start        


class GaussianMultidimensionalDataset(torch.utils.data.Dataset):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.mean + self.std * np.random.normal()
        return torch.tensor(sample, dtype=torch.float64), torch.tensor(
            [1], dtype=torch.float64
        )

inputs = np.stack([sample[0] for sample in train_ds])

# Create the dataset
gaussian_multidimensional_dataset = GaussianMultidimensionalDataset(
    size=len(test_ds), 
    mean=np.mean(inputs, axis=0).flatten(),
    std=np.std(inputs, axis=0).flatten(),
)

# Create the DataLoader
gaussian_loader = DataLoader(
    gaussian_multidimensional_dataset, batch_size=ARGS.batch_size, shuffle=True
)


with torch.no_grad():
    preds = []
    for x, y in gaussian_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        # Compute Gaussian entropy
        entropy = 0.5 * torch.log(2 * np.pi * np.e * var)
        entropy = entropy.resize(entropy.size(0), 1)  # Reshape to ensure correct dimensions

        preds.append(entropy.cpu().numpy())

    for x, y in test_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        # Compute Gaussian entropy
        entropy = 0.5 * torch.log(2 * np.pi * np.e * var)
        entropy = entropy.resize(entropy.size(0), 1)  # Reshape to ensure correct dimensions

        preds.append(entropy.cpu().numpy())

# Concatenate all predictions
preds = np.concatenate(preds, axis=0)
# Create labels for OOD detection
ood_labels = np.concatenate(
    [np.ones(len(gaussian_multidimensional_dataset)), np.zeros(len(test_ds))]
)

auc_lla = roc_auc_score(ood_labels.flatten(), preds.flatten())
print(f"AUC for OOD detection: {auc_lla:.4f}")


# -----------------------------------------------------------------------------#
# Save results                                                                 #
# -----------------------------------------------------------------------------#
def build_filename(a: argparse.Namespace) -> str:
    bits = [f"s{a.seed}"]
    return ARGS.dataset + "_lla_" + "_".join(bits)

results: Dict[str, float | int | str] = {
    # core metrics
    "model": "Laplace",
    "dataset": ARGS.dataset,
    "test_rmse": metrics_lla_test["RMSE"],
    "test_nll": metrics_lla_test["NLL"],
    "test_crps": metrics_lla_test["CRPS"],
    "test_cqm": metrics_lla_test["CQM"],
    "test_auc": auc_lla,
    # hyper-parameters
    "seed": ARGS.seed,
    "dtype": ARGS.dtype,
    "device": DEVICE.type,
    "prior_opt_lr": ARGS.prior_opt_lr,
    "prior_opt_iters": ARGS.prior_opt_iters,
    "optimize_hyper_parameters": ARGS.optimize_hyper_parameters,
}
dir = build_filename(ARGS) + ".csv"
pd.DataFrame([results]).to_csv(RESULTS_DIR / dir, index=False)

