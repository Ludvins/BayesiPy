from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets

# -----------------------------------------------------------------------------#
# Local modules (make sure your PYTHONPATH includes the project root)          #
# -----------------------------------------------------------------------------#
sys.path.append(".")
from bayesipy.laplace import Laplace          # noqa: E402
from bayesipy.utils import assert_reproducibility  # noqa: E402
from bayesipy.utils.datasets import (              # noqa: E402
    CIFAR10_Dataset,
    CIFAR10_OOD_Dataset,
)
from bayesipy.utils.metrics import OOD, SoftmaxClassification  # noqa: E402
from bayesipy.utils.pretrained_models import CIFAR10_Resnet    # noqa: E402

# -----------------------------------------------------------------------------#
# Argument parsing                                                             #
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="CIFAR-10 Laplace experiment")

# *Experiment control
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument(
    "--dtype",
    type=str,
    default="float32",
    choices=["float32", "float64"],
    help="Torch floating-point precision",
)
parser.add_argument("--batch_size", type=int, default=50, help="Mini-batch size")

# *Hardware / I/O
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda"],
    help='Device to run on; "auto" picks CUDA if available',
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results",
    help="Directory in which to save CSV files",
)

# *Model architecture
parser.add_argument(
    "--backbone",
    type=str,
    default="resnet20",
    choices=["resnet20", "resnet32", "resnet44", "resnet56", "resnet110"],
    help="WideResNet backbone for CIFAR-10",
)

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
opt_group = parser.add_mutually_exclusive_group()
opt_group.add_argument(
    "--optimize_hyper_parameters",
    dest="optimize_hyper_parameters",
    action="store_true",
    help="Enable hierarchical optimisation of prior precision (default)",
)
opt_group.add_argument(
    "--no_optimize_hyper_parameters",
    dest="optimize_hyper_parameters",
    action="store_false",
    help="Disable hierarchical optimisation of prior precision",
)
parser.set_defaults(optimize_hyper_parameters=True)

ARGS = parser.parse_args()

# -----------------------------------------------------------------------------#
# Derived constants                                                            #
# -----------------------------------------------------------------------------#
DTYPE: torch.dtype = {"float32": torch.float32, "float64": torch.float64}[ARGS.dtype]
DEVICE: torch.device
if ARGS.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(ARGS.device)

RESULTS_DIR = Path(ARGS.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert_reproducibility(ARGS.seed)

# -----------------------------------------------------------------------------#
# Models & transforms                                                          #
# -----------------------------------------------------------------------------#
backbone = CIFAR10_Resnet(ARGS.backbone).to(DTYPE)
embedding_net = CIFAR10_Resnet(ARGS.backbone, embedding=True).to(DTYPE)
transform = CIFAR10_Resnet(ARGS.backbone, get_transform=True)

# -----------------------------------------------------------------------------#
# Data                                                                         #
# -----------------------------------------------------------------------------#
cifar10 = CIFAR10_Dataset(transform=transform)
train_ds, test_ds = cifar10.train_test_splits()
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=ARGS.batch_size)


# -----------------------------------------------------------------------------#
# Laplace approximation                                                        #
# -----------------------------------------------------------------------------#
backbone_n_params = sum(p.numel() for p in backbone.parameters())

lla = Laplace(
    model=copy.deepcopy(backbone).to(DTYPE).to(DEVICE),
    likelihood="classification",
)

svhn = datasets.SVHN(
    root=Path("data"),
    split="train",
    download=True,
    transform=transform,
)
context_loader = DataLoader(svhn, batch_size=ARGS.batch_size, shuffle=True)

# Actual Laplace fit
lla.fit(
    train_loader=train_loader,
    progress_bar=True,
)

if ARGS.optimize_hyper_parameters:
    # Define log prior and log noise for LLA hyperparameters
    log_prior = torch.zeros(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior], lr=ARGS.prior_opt_lr)

    # Optimize LLA hyperparameters over 100 iterations
    for i in range(ARGS.prior_opt_iters):
        hyper_optimizer.zero_grad()
        neg_marglik = -lla.log_marginal_likelihood(log_prior.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    # Print optimized prior precision and noise sigma
    prior_precision = log_prior.exp().item()
    print("Prior precision:", prior_precision)

# -----------------------------------------------------------------------------#
# Evaluation                                                                   #
# -----------------------------------------------------------------------------#
metrics_lla_test = SoftmaxClassification()
with torch.no_grad():
    for x, y in test_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        metrics_lla_test.update(y.to(DEVICE), mean, var)
metrics_lla_test = metrics_lla_test.get_dict()

# OOD evaluation (SVHN → CIFAR-10 shift)
ood_ds = CIFAR10_OOD_Dataset(transform=transform)
_, ood_test = ood_ds.get_splits()
ood_loader = DataLoader(ood_test, batch_size=ARGS.batch_size, shuffle=True)

ood_metrics = OOD()
with torch.no_grad():
    for x, y in ood_loader:
        mean, var = lla.predict(x.to(DEVICE).to(DTYPE))
        ood_metrics.update(y.to(DEVICE), mean, var)

auc_lla = roc_auc_score(
    ood_metrics.get_dict()["labels"],
    ood_metrics.get_dict()["preds"],
)

# -----------------------------------------------------------------------------#
# Save results                                                                 #
# -----------------------------------------------------------------------------#
def build_filename(args: argparse.Namespace) -> str:
    """Create a short but descriptive file-name that encodes the run’s HPs."""
    bits = [
        f"bb{args.backbone}",
        f"s{args.seed}",
    ]
    return "cifar10_lla_" + "_".join(bits) + ".csv"


results: Dict[str, float | int | str] = {
    # core metrics
    "model": "Laplace",
    "test_acc": metrics_lla_test["ACC"],
    "test_nll": metrics_lla_test["NLL"],
    "test_ece": metrics_lla_test["ECE"],
    "test_brier": metrics_lla_test["BRIER"],
    "test_auc": auc_lla,
    # hyper-parameters
    "seed": ARGS.seed,
    "dtype": ARGS.dtype,
    "device": DEVICE.type,
    "backbone": ARGS.backbone,
    "prior_opt_lr": ARGS.prior_opt_lr,
    "prior_opt_iters": ARGS.prior_opt_iters,
    "optimize_hyper_parameters": ARGS.optimize_hyper_parameters,
}

pd.DataFrame([results]).to_csv(RESULTS_DIR / build_filename(ARGS), index=False)
