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
from bayesipy.laplace import TestLaplace         
from bayesipy.utils import assert_reproducibility 
from bayesipy.utils.datasets import Airline_Dataset, Year_Dataset, Taxi_Dataset
from bayesipy.utils.metrics import Regression
from bayesipy.utils.pretrained_models import Airline_MLP, Year_MLP, Taxi_MLP    

# -----------------------------------------------------------------------------#
# Argument parsing                                                             #
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="CIFAR-10 Laplace experiment (timed)")

# *Experiment control
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
parser.add_argument("--batch_size", type=int, default=50)

# *Hardware / I/O
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
parser.add_argument("--results_dir", type=str, default="results")

# *Model architecture
parser.add_argument("--features", type=int, default=500)

# *Optimisation / Laplace hyper-parameters
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--iterations", type=int, default=20_000)
parser.add_argument("--prior_opt_lr", type=float, default=0.1)
parser.add_argument("--prior_opt_iters", type=int, default=1_000)

parser.add_argument("--optimize_hyper_parameters", dest="optimize_hyper_parameters",
                     action="store_true")
parser.add_argument("--no_optimize_hyper_parameters", dest="optimize_hyper_parameters",
                     action="store_false")
parser.add_argument("--use_embedding", dest="use_embedding",
                     action="store_true", help="Use the embedding of the backbone as context points")
parser.add_argument("--no_use_embedding", dest="use_embedding",
                     action="store_false", help="Do not use the embedding of the backbone as context points")
parser.add_argument("--sqrt", dest="sqrt",
                     action="store_true", help="Use the embedding of the backbone as context points")
parser.add_argument("--no_sqrt", dest="sqrt",
                     action="store_false", help="Do not use the embedding of the backbone as context points")
parser.add_argument("--dataset", type=str, default="airline",
                    choices=["airline", "year", "taxi"],
                    help="Dataset to use for the experiment")

parser.set_defaults(optimize_hyper_parameters=True)
parser.set_defaults(use_embedding=True)
parser.set_defaults(sqrt=True)



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
# Models & transforms                                                          #
# -----------------------------------------------------------------------------#
if ARGS.dataset == "airline":
    backbone = Airline_MLP().to(DTYPE)
elif ARGS.dataset == "year":
    backbone = Year_MLP().to(DTYPE)
elif ARGS.dataset == "taxi":
    backbone = Taxi_MLP().to(DTYPE)
else:
    raise ValueError(f"Unknown dataset: {ARGS.dataset}")
# -----------------------------------------------------------------------------#
# Data                                                                         #
# -----------------------------------------------------------------------------#
if ARGS.dataset == "airline":
   ds = Airline_Dataset()
elif ARGS.dataset == "year":
   ds = Year_Dataset()
elif ARGS.dataset == "taxi":
   ds = Taxi_Dataset()
else:
   raise ValueError(f"Unknown dataset: {ARGS.dataset}")
train_ds, test_ds = ds.train_test_splits()
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=ARGS.batch_size)


class CustomModel(nn.Module):
    def __init__(self, scale_params, n_features):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(ds.input_dim, 200)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, n_features * 1)

        # Define a scalar parameter
        if ARGS.sqrt:
            self.scale = torch.sqrt(torch.tensor(scale_params / n_features, dtype=DTYPE))
        else:
            self.scale = torch.tensor(scale_params / n_features, dtype=DTYPE)
        self.n_features = n_features

    def forward(self, x):

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x).reshape(x.size(0), self.n_features, 1)
        return self.scale * x

model2 = CustomModel(
    scale_params=sum(p.numel() for p in backbone.parameters()),
                            n_features=ARGS.features).to(DTYPE)

lla = TestLaplace(model=copy.deepcopy(backbone).to(DTYPE).to(DEVICE),
                  model2=model2.to(DEVICE),
                  likelihood="regression",
                  y_mean=ds.y_mean,
                  y_std=ds.y_std,)

# make an uniform loader from the min to max of each data dimension
inputs = torch.stack([sample[0] for sample in train_ds]).numpy()
min = np.min(inputs, axis=0).flatten()
max = np.max(inputs, axis=0).flatten()
print(f"Min: {min.shape}, Max: {max.shape}")

class UniformMultidimensionalDataset(torch.utils.data.Dataset):
    def __init__(self, size, low, high):
        self.size = size
        self.low = low
        self.high = high

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = np.random.uniform(self.low, self.high)
        return torch.tensor(sample, dtype=torch.float64), torch.tensor(
            [1], dtype=torch.float64
        )


# Define the size of the dataset
multidimensional_dataset_size = 1000  # You can adjust this size as needed

# Create the dataset
uniform_multidimensional_dataset = UniformMultidimensionalDataset(
    size=multidimensional_dataset_size, low=min, high=max
)

# Create the DataLoader
uniform_loader = DataLoader(
    uniform_multidimensional_dataset, batch_size=ARGS.batch_size, shuffle=True
)

# -----------------------------  TRAIN  (timed)  ------------------------------#
train_start = time.perf_counter()                    
losses, losses_exact = lla.fit(
    iterations=ARGS.iterations,
    train_loader=train_loader,
    lr=ARGS.lr,
    context_points_loader=uniform_loader,
    optimize_hyper_parameters=ARGS.optimize_hyper_parameters,
    prior_opt_iterations=ARGS.prior_opt_iters,
    prior_opt_lr=ARGS.prior_opt_lr,
    verbose=False,
)
train_time = time.perf_counter() - train_start      



# Calculate the number of minibatches per epoch
minibatches_per_epoch = int(len(train_ds) / ARGS.batch_size)
print(f"Minibatches per epoch: {minibatches_per_epoch}")

# Calculate the average loss per epoch
average_losses_per_epoch = [
    np.mean(losses[i * minibatches_per_epoch : (i + 1) * minibatches_per_epoch])
    for i in range(len(losses) // minibatches_per_epoch)
]


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
        var = torch.diagonal(var).squeeze().unsqueeze(-1)

        # Compute Gaussian entropy
        entropy = 0.5 * torch.log(2 * np.pi * np.e * var)
        preds.append(entropy.cpu().numpy())

    for x, y in test_loader:
            mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
            var = torch.diagonal(var).squeeze().unsqueeze(-1)

            # Compute Gaussian entropy
            entropy = 0.5 * torch.log(2 * np.pi * np.e * var)
            preds.append(entropy.cpu().numpy())

# Concatenate all predictions
preds = np.concatenate(preds, axis=0)
# Create labels for OOD detection
ood_labels = np.concatenate(
    [np.ones(len(gaussian_multidimensional_dataset)), np.zeros(len(test_ds))]
)
auc_lla = roc_auc_score(ood_labels.flatten(), preds.flatten())


# -----------------------------------------------------------------------------#
# Save results                                                                 #
# -----------------------------------------------------------------------------#
def build_filename(a: argparse.Namespace) -> str:
    bits = [f"f{a.features}", f"bs{a.batch_size}",
            f"lr{a.lr}", f"it{a.iterations}", f"s{a.seed}", f"s{a.sqrt}"]
    return ARGS.dataset + "_alla_" + "_".join(bits)

results: Dict[str, float | int | str] = {
    # core metrics
    "model": "ALaplace",
    "dataset": ARGS.dataset,
    "test_rmse": metrics_lla_test["RMSE"],
    "test_nll": metrics_lla_test["NLL"],
    "test_crps": metrics_lla_test["CRPS"],
    "test_cqm": metrics_lla_test["CQM"],
    "test_auc": auc_lla,
    # timings
    "train_time_s": round(train_time, 2),            
    "eval_time_s": round(eval_time, 2),              
    # hyper-parameters
    "seed": ARGS.seed,
    "dtype": ARGS.dtype,
    "batch_size": ARGS.batch_size,
    "device": DEVICE.type,
    "features": ARGS.features,
    "lr": ARGS.lr,
    "laplace_iters": ARGS.iterations,
    "prior_opt_lr": ARGS.prior_opt_lr,
    "prior_opt_iters": ARGS.prior_opt_iters,
    "optimize_hyper_parameters": ARGS.optimize_hyper_parameters,
    "sqrt": ARGS.sqrt
}
dir = build_filename(ARGS) + ".csv"
pd.DataFrame([results]).to_csv(RESULTS_DIR / dir, index=False)


# Plot the averaged loss per epoch
plt.plot(average_losses_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Average Loss per Epoch")
plt.yscale("log")
dir = build_filename(ARGS) + ".png"
plt.savefig(RESULTS_DIR / dir)
plt.close()
