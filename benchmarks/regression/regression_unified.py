"""Unified regression experiment runner for large tabular/time-series datasets
(Airline, Taxi, Year) using BayesiPy models.

This completes and cleans up the provided starter script by:
- Encapsulating setup, training, evaluation, and saving into functions.
- Adding full argparse arguments for each method (FMGP, MAP, LLA, ELLA, MFVI, SNGP, VaLLA).
- Ensuring compatibility with `bayesipy.utils.metrics.Regression` + `score`.
- Adding basic config saving + MLflow experiment tracking.

Outputs
-------
- Prints a one-row CSV-style table of metrics.
- Saves the same to `results/<dataset>/<method>_<seed>.csv`.
- Logs params, metrics, and artifacts to MLflow.

Notes
-----
- We assume BayesiPy classes expose `.fit(...)` (when needed) and `.predict(x)`
  returning `(Fmean, Fvar)` for Regression metrics. For MAP, we wrap the model to
  return a predictive variance estimated from training residuals (homoscedastic).
"""

from __future__ import annotations

import sys
import os
import copy
import argparse
from time import time
import json
import numbers
from typing import Any

import numpy as np
import pandas as pd
from pyparsing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow

# Local path
sys.path.append(".")

# BayesiPy imports
from bayesipy.laplace import ella
from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression, score
from bayesipy.utils.pretrained_models import Airline_MLP, Taxi_MLP, Year_MLP

torch.backends.cudnn.benchmark = True


# ----------------------------
# Utilities
# ----------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(args: argparse.Namespace, outdir: str) -> str:
    """Save the run configuration to a JSON file and return its path."""
    cfg = vars(args).copy()
    cfg_path = os.path.join(
        outdir,
        f"config_{args.dataset}_{args.method}_seed{args.seed}.json",
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg_path


# ----------------------------
# Data & Model
# ----------------------------


def load_dataset_and_model(
    dataset: str,
    device: torch.device,
    dtype: torch.dtype,
):
    dataset = dataset.lower()
    if dataset == "airline":
        ds = Airline_Dataset()
        model = Airline_MLP().to(device).to(dtype)
    elif dataset == "year":
        ds = Year_Dataset()
        model = Year_MLP().to(device).to(dtype)
    elif dataset == "taxi":
        ds = Taxi_Dataset()
        model = Taxi_MLP().to(device).to(dtype)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_dataset, test_dataset = ds.train_test_splits()
    val_dataset = ds.validation_split()

    # Save target normalization for potential post-processing (if needed)
    y_mean = float(ds.y_mean.item())
    y_std = float(ds.y_std.item())

    return (train_dataset, test_dataset, val_dataset), model, y_mean, y_std


def make_loaders(
    train_ds,
    test_ds,
    val_ds,
    batch_size: int,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


class MapWrapper(nn.Module):
    """Wrap a deterministic regressor to provide (Fmean, Fvar) via homoscedastic noise.
    The variance is estimated on-the-fly from training residuals.
    """

    def __init__(
        self,
        base: nn.Module,
        sigma2: float,
        y_mean: float,
        y_std: float,
    ):
        super().__init__()
        self.base = base
        self.register_buffer("sigma2", torch.tensor(float(sigma2)))
        self.device = next(base.parameters()).device
        self.y_mean = y_mean
        self.y_std = y_std
        self.dtype = next(base.parameters()).dtype

    def predict(self, x: torch.Tensor):
        mean = self.base(x)
        var = torch.full_like(mean, self.sigma2.item())
        return mean * self.y_std + self.y_mean, var * (self.y_std**2)


def estimate_homoscedastic_variance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    se_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            se_sum += torch.sum((pred - yb) ** 2).item()
            n += yb.numel()
    sigma2 = se_sum / max(1, n)
    return float(sigma2 if sigma2 > 1e-12 else 1e-4)


# ----------------------------
# Build estimator per method
# ----------------------------


def build_estimator(
    method: str,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    y_mean: float,
    y_std: float,
):
    method = method.lower()
    if method == "fmgp":
        from bayesipy.fmgp import FMGP

        return FMGP(
            model=model,
            likelihood="regression",
            kernel=args.fmgp_kernel,
            inducing_locations=args.fmgp_inducing_locations,
            num_inducing=args.fmgp_num_inducing,
            noise_variance=args.fmgp_noise_variance,
            seed=args.seed,
            y_mean=y_mean,
            y_std=y_std,
        )

    if method == "map":
        # Wrap with homoscedastic variance estimated from train later
        return "MAP_WRAPPER"  # sentinel; we will finalize after estimating sigma^2

    if method == "lla":
        from bayesipy.laplace import Laplace

        return Laplace(
            model,
            likelihood="regression",
            subset_of_weights=args.lla_subset,
            hessian_structure=args.lla_hessian,
            y_mean=y_mean,
            y_std=y_std,
        )

    if method == "ella":
        from bayesipy.laplace import ELLA

        return ELLA(
            model,
            likelihood="regression",
            subsample_size=args.ella_subsample,
            n_eigenvalues=args.ella_n_eigenvalues,
            prior_precision=args.ella_prior,
            y_mean=y_mean,
            y_std=y_std,
            seed=args.seed,
        )

    if method == "valla":
        from bayesipy.laplace import VaLLA

        return VaLLA(
            model,
            likelihood="regression",
            inducing_locations=args.valla_inducing_locations,
            num_inducing=args.valla_num_inducing,
            noise_variance=args.valla_noise_variance,
            y_mean=y_mean,
            y_std=y_std,
            seed=args.seed,
        )

    if method == "mfvi":
        from bayesipy.mfvi import MFVI

        return MFVI(
            model,
            likelihood="regression",
            noise_variance=args.mfvi_noise_variance,
            prior_precision=args.mfvi_prior,
            n_samples=args.mfvi_n_samples,
            seed=args.seed,
            y_mean=y_mean,
            y_std=y_std,
        )

    if method == "sngp":
        from bayesipy.sngp import SNGP

        return SNGP(
            model,
            likelihood="regression",
            gp_kernel_scale=args.sngp_kernel_scale,
            n_random_features=args.sngp_n_random_features,
            gp_output_bias=args.sngp_gp_output_bias,
            layer_norm_eps=args.sngp_layer_norm_eps,
            n_power_iterations=args.sngp_n_power_iterations,
            scale_random_features=args.sngp_scale_random_features,
            normalize_input=args.sngp_normalize_input,
            gp_cov_momentum=args.sngp_gp_cov_momentum,
            gp_cov_ridge_penalty=args.sngp_gp_cov_ridge_penalty,
            noise_variance=args.sngp_noise_variance,  # initial value for log_noise
            y_mean=y_mean,
            y_std=y_std,
            seed=args.seed,
        )

    raise ValueError(f"Unknown method: {method}")


# ----------------------------
# Training / Fitting
# ----------------------------


def fit(
    estimator,
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyper_param_grid,
    args: argparse.Namespace,
) -> None:
    # FMGP / Laplace / ELLA / MFVI / SNGP typically have .fit; MAP has none
    if method == "map":
        return

    if method == "lla":
        estimator.fit(
            train_loader=train_loader,
            progress_bar=args.verbose,
        )
        estimator.optimize_prior_precision()

    if method == "ella":
        estimator.fit(
            train_loader=train_loader,
            verbose=args.verbose,
        )
        estimator.optimize_hyperparameters(
            val_loader,
            hyper_param_grid,
            verbose=args.verbose,
        )

    if method == "valla":
        estimator.fit(
            train_loader=train_loader,
            iterations=args.valla_iterations,
            lr=args.valla_lr,
            verbose=args.verbose,
        )

    if method == "mfvi":
        estimator.fit(
            iterations=args.mfvi_iterations,
            train_loader=train_loader,
            verbose=args.verbose,
        )

    if method == "sngp":
        estimator.fit(
            train_loader=train_loader,
            iterations=args.sngp_iterations,
            lr=args.sngp_lr,
            weight_decay=args.sngp_weight_decay,
            verbose=args.verbose,
        )

    if method == "fmgp":
        estimator.fit(
            train_loader=train_loader,
            iterations=args.fmgp_iterations,
            lr=args.fmgp_lr,
            verbose=args.verbose,
        )


# ----------------------------
# Evaluation + Saving
# ----------------------------


def evaluate_and_save(
    estimator,
    method: str,
    test_loader: DataLoader,
    args: argparse.Namespace,
    outdir: str,
    train_time: float,
    test_time: float,
):
    # score expects a model-like object with .predict returning (Fmean, Fvar)
    results = score(estimator, test_loader, Regression, verbose=True)
    results["method"] = method
    results["train_time"] = train_time
    results["test_time"] = time() - test_time
    results["dataset"] = args.dataset
    results["seed"] = args.seed

    ensure_dir(outdir)
    df = pd.DataFrame.from_dict(results, orient="index").transpose()
    print(df)
    csv_path = os.path.join(outdir, f"{method}_{args.seed}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return results, csv_path


# ----------------------------
# Argparse
# ----------------------------


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified regression experiment runner")
    # Config file (optional)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file specifying all arguments",
    )
    # core
    p.add_argument("--verbose", action="store_true", help="Verbose")
    p.add_argument(
        "--output",
        type=str,
        default="benchmarks/regression/results",
        help="Output dir",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["airline", "year", "taxi"],
        required=True,
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["map", "fmgp", "lla", "ella", "mfvi", "sngp", "valla"],
    )
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="Seed")

    # FMGP
    p.add_argument("--fmgp_iterations", type=int, default=70000)
    p.add_argument("--fmgp_lr", type=float, default=1e-3)
    p.add_argument("--fmgp_noise_variance", type=float, default=1e-5)
    p.add_argument("--fmgp_kernel", type=str, default="RBF")
    p.add_argument(
        "--fmgp_inducing_locations",
        type=str,
        default="kmeans",
        choices=["kmeans", "random"],
    )
    p.add_argument("--fmgp_num_inducing", type=int, default=100)

    # LLA
    p.add_argument(
        "--lla_subset",
        type=str,
        default="last_layer",
        choices=["last_layer", "all"],
    )
    p.add_argument(
        "--lla_hessian",
        type=str,
        default="kron",
        choices=["diag", "kron", "full"],
    )

    # ELLA
    p.add_argument("--ella_subsample", type=int, default=1024)
    p.add_argument("--ella_n_eigenvalues", type=int, default=50)
    p.add_argument("--ella_prior", type=float, default=1.0)

    # VaLLA
    p.add_argument(
        "--valla_inducing_locations",
        type=str,
        choices=["kmeans", "random"],
        default="kmeans",
    )
    p.add_argument("--valla_num_inducing", type=int, default=100)
    p.add_argument("--valla_noise_variance", type=float, default=1e-5)
    p.add_argument("--valla_iterations", type=int, default=40000)
    p.add_argument("--valla_lr", type=float, default=1e-3)

    # MFVI
    p.add_argument("--mfvi_iterations", type=int, default=8000)
    p.add_argument("--mfvi_prior", type=float, default=1.0)
    p.add_argument("--mfvi_n_samples", type=int, default=20)
    p.add_argument("--mfvi_noise_variance", type=float, default=1)

    # SNGP
    p.add_argument("--sngp_kernel_scale", type=float, default=1.0)
    p.add_argument("--sngp_n_random_features", type=int, default=1024)
    p.add_argument("--sngp_gp_output_bias", type=float, default=0.0)
    p.add_argument("--sngp_layer_norm_eps", type=float, default=1e-6)
    p.add_argument("--sngp_n_power_iterations", type=int, default=1)
    p.add_argument("--sngp_scale_random_features", action="store_true")
    p.add_argument("--sngp_normalize_input", action="store_true")
    p.add_argument("--sngp_gp_cov_momentum", type=float, default=0.999)
    p.add_argument("--sngp_gp_cov_ridge_penalty", type=float, default=1e-3)
    p.add_argument("--sngp_iterations", type=int, default=20000)
    p.add_argument("--sngp_lr", type=float, default=1e-5)
    p.add_argument("--sngp_weight_decay", type=float, default=0.1)
    p.add_argument("--sngp_noise_variance", type=float, default=1)

    return p

METHOD_PREFIXES = {
    "fmgp": ["fmgp_"],
    "lla": ["lla_"],
    "ella": ["ella_"],
    "valla": ["valla_"],
    "mfvi": ["mfvi_"],
    "sngp": ["sngp_"],
    "map": [],  # MAP has no special hyperparameters
}


CORE_PARAM_KEYS = {
    "dataset",
    "method",
    "batch_size",
    "seed",
    "seeds",
    "output",
    "verbose",
    "config",
}


def params_for_method(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    """Return a dict of params to log for the given method.

    - Always include 'core' params (dataset, method, seed, ...)
    - Only include method-specific params, e.g. lla_* when method='lla'
    """
    all_params = vars(args).copy()
    method = method.lower()

    prefixes = METHOD_PREFIXES.get(method, [])
    params: Dict[str, Any] = {}

    for k, v in all_params.items():
        # Always keep core params
        if k in CORE_PARAM_KEYS:
            params[k] = v
            continue

        # Keep only params with the right prefix for this method
        if any(k.startswith(pref) for pref in prefixes):
            params[k] = v

    return params


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    # If a config file is provided, override args from it
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    seed = 2147483647 - int(args.seed)

    set_seed(seed)

    # Prepare output directory and save config
    outdir = os.path.join(args.output, args.dataset)
    ensure_dir(outdir)
    config_path = save_config(args, outdir)

    # Load data & pretrained base model for the chosen dataset
    (train_ds, test_ds, val_ds), base_model, y_mean, y_std = load_dataset_and_model(
        args.dataset,
        device,
        dtype,
    )
    train_loader, test_loader, val_loader = make_loaders(
        train_ds,
        test_ds,
        val_ds,
        args.batch_size,
    )

    # Build estimator (may return sentinel for MAP)
    estimator = build_estimator(
        args.method,
        base_model,
        args,
        device,
        y_mean,
        y_std,
    )

    # If MAP, estimate homoscedastic variance and wrap
    if estimator == "MAP_WRAPPER":
        sigma2 = estimate_homoscedastic_variance(
            base_model,
            train_loader,
            device,
        )
        estimator = MapWrapper(base_model, sigma2, y_mean, y_std)

    # Hyper-parameter grid (only used by ELLA currently)
    prior_precisions = [1000, 100, 10, 1, 0.1, 0.01]
    sigma_noises = [0.5, 1, 1.5]
    combinations = [(pp, sn) for pp in prior_precisions for sn in sigma_noises]

    # ---- MLflow tracking ----
    mlflow.set_experiment("bayesipy_regression")
    run_name = f"{args.dataset}_{args.method}_seed{args.seed}"

    with mlflow.start_run(run_name=run_name):
        # Log all CLI/config args as params
        method_params = params_for_method(args, args.method)
        mlflow.log_params(method_params)
        # Log config file as an artifact
        mlflow.log_artifact(config_path)

        # Fit (if applicable)
        t0 = time()
        fit(
            estimator,
            args.method,
            train_loader,
            val_loader,
            combinations,
            args,
        )
        t1 = time()

        # Evaluate
        t2 = time()
        results, csv_path = evaluate_and_save(
            estimator,
            args.method,
            test_loader,
            args,
            outdir=outdir,
            train_time=t1 - t0,
            test_time=t2,
        )

        # Log metrics (only numeric values)
        numeric_results = {
            k: float(v)
            for k, v in results.items()
            if isinstance(v, numbers.Number) and k[0] != "Q"
        }
        mlflow.log_metrics(numeric_results)

        # Log CSV results as artifact
        mlflow.log_artifact(csv_path)


if __name__ == "__main__":
    main()
