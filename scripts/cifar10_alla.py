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
from bayesipy.utils.datasets import CIFAR10_Dataset, CIFAR10_OOD_Dataset  
from bayesipy.utils.metrics import OOD, SoftmaxClassification             
from bayesipy.utils.pretrained_models import CIFAR10_Resnet             

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
parser.add_argument("--backbone", type=str, default="resnet20",
                    choices=["resnet20", "resnet32", "resnet44", "resnet56", "resnet110"])
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
backbone = CIFAR10_Resnet(ARGS.backbone).to(DTYPE)
embedding = CIFAR10_Resnet(ARGS.backbone, embedding=True).to(DTYPE)
transform = CIFAR10_Resnet(ARGS.backbone, get_transform=True)

# -----------------------------------------------------------------------------#
# Data                                                                         #
# -----------------------------------------------------------------------------#
cifar10 = CIFAR10_Dataset(transform=transform)
train_ds, test_ds = cifar10.train_test_splits()
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=ARGS.batch_size)

# -----------------------------------------------------------------------------#
# ConvNet head with BN                                                         #
# -----------------------------------------------------------------------------#
class ConvHead(nn.Module):
    def __init__(self, n_features: int, scale_params: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 200),
            #nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, n_features * 10),
        )
        # Define a scalar parameter
        if ARGS.sqrt:
            self.scale = torch.sqrt(torch.tensor(scale_params / n_features, dtype=DTYPE))
        else:
            self.scale = torch.tensor(scale_params / n_features, dtype=DTYPE)
        
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x).reshape(x.size(0), self.n_features, 10)
        return self.scale * x

# -----------------------------------------------------------------------------#

class CustomModel(nn.Module):
    def __init__(self, secondary_network, scale_params, n_features):
        super(CustomModel, self).__init__()
        self.secondary_network = secondary_network
        self.secondary_network.eval()
        self.fc1 = nn.Linear(64, 250)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, n_features * 10)

        for param in self.secondary_network.parameters():
            param.requires_grad = False

        # Define a scalar parameter
        if ARGS.sqrt:
            self.scale = torch.sqrt(torch.tensor(scale_params / n_features, dtype=DTYPE))
        else:
            self.scale = torch.tensor(scale_params / n_features, dtype=DTYPE)
        self.n_features = n_features

    def forward(self, x):
        with torch.no_grad():
            x = self.secondary_network(x)

        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x).reshape(x.size(0), self.n_features, 10)
        return self.scale * x

# -----------------------------------------------------------------------------#
# Laplace approximation                                                        #
# -----------------------------------------------------------------------------#
if ARGS.use_embedding:
    # Use the embedding of the backbone as context points
    conv_head = CustomModel(secondary_network=copy.deepcopy(embedding).to(DTYPE).to(DEVICE),
                            scale_params=sum(p.numel() for p in backbone.parameters()),
                            n_features=ARGS.features).to(DTYPE)
else:
    conv_head = ConvHead(ARGS.features, sum(p.numel() for p in backbone.parameters())).to(DTYPE)

lla = TestLaplace(model=copy.deepcopy(backbone).to(DTYPE).to(DEVICE),
                  model2=conv_head.to(DEVICE),
                  likelihood="classification")

svhn = datasets.SVHN(root="data", split="train", download=True, transform=transform)
context_loader = DataLoader(svhn, batch_size=ARGS.batch_size, shuffle=True)

# -----------------------------  TRAIN  (timed)  ------------------------------#
train_start = time.perf_counter()                    
losses, losses_exact = lla.fit(
    iterations=ARGS.iterations,
    train_loader=train_loader,
    lr=ARGS.lr,
    context_points_loader=context_loader,
    optimize_hyper_parameters=ARGS.optimize_hyper_parameters,
    prior_opt_iterations=ARGS.prior_opt_iters,
    prior_opt_lr=ARGS.prior_opt_lr,
    verbose=True,
)
train_time = time.perf_counter() - train_start      



# Calculate the number of minibatches per epoch
minibatches_per_epoch = int(len(train_ds) / ARGS.batch_size)

# Calculate the average loss per epoch
average_losses_per_epoch = [
    np.mean(losses[i * minibatches_per_epoch : (i + 1) * minibatches_per_epoch])
    for i in range(len(losses) // minibatches_per_epoch)
]


# -----------------------------  EVALUATE (timed) -----------------------------#
eval_start = time.perf_counter()                   

metrics_lla_test = SoftmaxClassification()
with torch.no_grad():
    for x, y in test_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        var = torch.diagonal(var).permute(2, 0, 1)
        metrics_lla_test.update(y.to(DEVICE), mean, var)
metrics_lla_test = metrics_lla_test.get_dict()

eval_time = time.perf_counter() - eval_start        

ood_ds = CIFAR10_OOD_Dataset(transform=transform)
_, ood_test = ood_ds.get_splits()
ood_loader = DataLoader(ood_test, batch_size=ARGS.batch_size, shuffle=True)

ood_metrics = OOD()
with torch.no_grad():
    for x, y in ood_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        var = torch.diagonal(var).permute(2, 0, 1)
        ood_metrics.update(y.to(DEVICE), mean, var)

auc_lla = roc_auc_score(ood_metrics.get_dict()["labels"],
                        ood_metrics.get_dict()["preds"])



# -----------------------------------------------------------------------------#
# Save results                                                                 #
# -----------------------------------------------------------------------------#
def build_filename(a: argparse.Namespace) -> str:
    bits = [f"bb{a.backbone}", f"emb{a.use_embedding}", f"f{a.features}", f"bs{a.batch_size}",
            f"lr{a.lr}", f"it{a.iterations}", f"s{a.seed}", f"s{a.sqrt}"]
    return "cifar10_alla_" + "_".join(bits)

results: Dict[str, float | int | str] = {
    # core metrics
    "model": "ALaplace",
    "test_acc": metrics_lla_test["ACC"],
    "test_nll": metrics_lla_test["NLL"],
    "test_ece": metrics_lla_test["ECE"],
    "test_brier": metrics_lla_test["BRIER"],
    "test_auc": auc_lla,
    # timings
    "train_time_s": round(train_time, 2),            
    "eval_time_s": round(eval_time, 2),              
    # hyper-parameters
    "seed": ARGS.seed,
    "dtype": ARGS.dtype,
    "batch_size": ARGS.batch_size,
    "device": DEVICE.type,
    "backbone": ARGS.backbone,
    "features": ARGS.features,
    "lr": ARGS.lr,
    "laplace_iters": ARGS.iterations,
    "prior_opt_lr": ARGS.prior_opt_lr,
    "prior_opt_iters": ARGS.prior_opt_iters,
    "optimize_hyper_parameters": ARGS.optimize_hyper_parameters,
    "use_embedding": ARGS.use_embedding,
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
