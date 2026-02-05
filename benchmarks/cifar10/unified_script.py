from __future__ import annotations

"""
Unified CIFAR10 classification experiment runner using BayesiPy models and
pre-trained ResNet backbones.

This script mirrors the unified regression runner but for image classification:

- Dataset: CIFAR10 (optionally rotated, OOD, and CIFAR10-C corrupted datasets).
- Backbone: pre-trained ResNets from `bayesipy.utils.pretrained_models.CIFAR10_Resnet`.
- Methods: MAP, FMGP, LLA, ELLA, VaLLA, MFVI, SNGP.

Features
--------
- Single entry-point to run any method on CIFAR10.
- Encapsulated setup, training, evaluation, and saving.
- Argparse flags for each method's hyper-parameters.
- Optional evaluation on:
    * rotated CIFAR10 (`--test_rotations`)
    * CIFAR10 OOD dataset (`--test_ood`)
    * CIFAR10-C corrupted test set (`--test_corrupted`)
- Saves metrics to CSV and logs params/metrics/artifacts to MLflow.
"""

import os
import sys
import json
import numbers
import argparse
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import mlflow

# Make sure we can import bayesipy from repo root when running as a script
sys.path.append(".")

# Optional: stricter CUDA reproducibility for some methods
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# BayesiPy imports
from bayesipy.utils.datasets import (
    CIFAR10_Dataset,
    CIFAR10_Rotated_Dataset,
    CIFAR10_OOD_Dataset,
    Precomputed_Output_Embedding_Dataset,
    CIFAR10C_Dataset,
)
from bayesipy.utils.metrics import (
    SoftmaxClassification,
    SoftmaxClassificationSamples,
    OOD,
    OOD_Samples,
    score,
    score_samples,
)
from bayesipy.utils.pretrained_models import CIFAR10_Resnet
from bayesipy.utils import assert_reproducibility

from bayesipy.fmgp import FMGP
from bayesipy.laplace import Laplace, ELLA, VaLLA
from bayesipy.mfvi import MFVI
from bayesipy.sngp import SNGP


torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_config(args: argparse.Namespace, outdir: str) -> str:
    """Save the run configuration to a JSON file and return its path."""
    cfg = vars(args).copy()
    cfg_path = os.path.join(
        outdir,
        f"config_cifar10_{args.method}_{args.resnet}_seed{args.seed}.json",
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg_path


def dtype_for_method(method: str) -> torch.dtype:
    """Match the dtypes used in the original CIFAR10 scripts."""
    method = method.lower()
    if method in {"lla", "mfvi", "sngp"}:
        return torch.float32
    return torch.float64


# ---------------------------------------------------------------------
# Data & Models
# ---------------------------------------------------------------------


def load_cifar10_and_resnet(
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[
    torch.nn.Module,
    Optional[torch.nn.Module],
    Optional[torch.nn.Module],
    Any,
    DataLoader,
    Optional[DataLoader],
    DataLoader,
]:
    """Load CIFAR10 datasets + pre-trained ResNet backbone.

    Returns
    -------
    base_model : nn.Module
        The pre-trained ResNet backbone.
    embedding : nn.Module or None
        Embedding network (for FMGP) if needed, else None.
    classifier : nn.Module or None
        Classifier network (for FMGP) if needed, else None.
    transform : torchvision transform
        Data transform corresponding to the chosen ResNet.
    train_loader, val_loader, test_loader : DataLoader
        Train/validation/test loaders.
    """
    resnet_name = args.resnet
    batch_size = args.batch_size

    # Data transform is always tied to the chosen backbone
    transform = CIFAR10_Resnet(resnet_name, get_transform=True)

    # Base model (full classifier)
    base_model = CIFAR10_Resnet(resnet_name).to(device).to(dtype)

    embedding: Optional[torch.nn.Module] = None
    classifier: Optional[torch.nn.Module] = None

    if method.lower() == "fmgp":
        # For FMGP we follow the original script and use a precomputed
        # output+embedding dataset for (train, test).
        embedding = CIFAR10_Resnet(resnet_name, embedding=True).to(device).to(dtype)
        classifier = CIFAR10_Resnet(resnet_name, classifier=True).to(device).to(dtype)

        dataset = Precomputed_Output_Embedding_Dataset(
            model=base_model,
            embedding=embedding,
            model_name=resnet_name,
            dataset=CIFAR10_Dataset,
            transform=transform,
        )
        train_ds, test_ds = dataset.train_test_splits()
        # No explicit validation set in the original FMGP CIFAR10 script
        val_loader = None
    else:
        dataset = CIFAR10_Dataset(transform=transform)
        train_ds, test_ds = dataset.train_test_splits()
        # ELLA uses validation_split(lower=...), others just validation_split()
        if method.lower() == "ella":
            val_ds = dataset.validation_split(lower=args.ella_lower_da)
        else:
            val_ds = dataset.validation_split()
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return base_model, embedding, classifier, transform, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------
# MAP wrapper for classification
# ---------------------------------------------------------------------


def attach_map_predict(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int = 10,
    eps: float = 1e-6,
) -> torch.nn.Module:
    """Attach a `.predict(x) -> (Fmean, Fvar)` method to a deterministic classifier.

    """
    model.device = device
    model.dtype = dtype

    def predict(self, x: torch.Tensor):
        f_mean = self(x)
        return f_mean.unsqueeze(0)

    model.predict = predict.__get__(model)  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------
# Build estimator per method
# ---------------------------------------------------------------------


def build_estimator(
    method: str,
    base_model: torch.nn.Module,
    embedding: Optional[torch.nn.Module],
    classifier: Optional[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    method = method.lower()

    if method == "map":
        # Deterministic classifier with tiny fixed covariance
        return attach_map_predict(base_model, device=device, dtype=dtype)

    if method == "fmgp":
        if embedding is None or classifier is None:
            raise ValueError("FMGP requires embedding and classifier networks.")
        return FMGP(
            embedding=embedding,
            classifier=classifier,
            likelihood="classification",
            kernel=args.fmgp_kernel,
            inducing_locations=args.fmgp_inducing_locations,
            num_inducing=args.fmgp_num_inducing,
            seed=args.seed,
        )

    if method == "lla":
        return Laplace(
            model=base_model,
            likelihood="classification",
            subset_of_weights=args.lla_subset,
            hessian_structure=args.lla_hessian,
        )

    if method == "ella":
        return ELLA(
            model=base_model,
            likelihood="classification",
            subsample_size=args.ella_subsample,
            n_eigenvalues=args.ella_n_eigenvalues,
            prior_precision=args.ella_prior,
            seed=args.seed,
        )

    if method == "valla":
        return VaLLA(
            model=base_model,
            likelihood="classification",
            inducing_locations=args.valla_inducing_locations,
            num_inducing=args.valla_num_inducing,
            seed=args.seed,
        )

    if method == "mfvi":
        return MFVI(
            model=base_model,
            n_samples=args.mfvi_n_samples,
            likelihood="classification",
            prior_precision=args.mfvi_prior,
            seed=args.seed,
        )

    if method == "sngp":
        return SNGP(
            model=base_model,
            n_random_features=args.sngp_n_random_features,
        )

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------
# Training / Fitting
# ---------------------------------------------------------------------


def fit_estimator(
    estimator: torch.nn.Module,
    method: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args: argparse.Namespace,
) -> None:
    """Dispatch to the appropriate `.fit(...)` call for each method."""
    method = method.lower()

    if method == "map":
        # MAP is pre-trained; nothing to do
        return

    if method == "fmgp":
        estimator.fit(  # type: ignore[call-arg]
            iterations=args.fmgp_iterations,
            lr=args.fmgp_lr,
            scheduler_gamma=args.fmgp_scheduler_gamma,
            train_loader=train_loader,
        )
        return

    if method == "lla":
        estimator.fit(train_loader=train_loader)  # type: ignore[call-arg]
        estimator.optimize_prior_precision()  # type: ignore[attr-defined]
        return

    if method == "ella":
        estimator.fit(  # type: ignore[call-arg]
            train_loader=train_loader,
            val_loader=val_loader,
            val_steps=args.ella_val_steps,
            verbose=args.verbose,
        )
        return

    if method == "valla":
        estimator.fit(  # type: ignore[call-arg]
            iterations=args.valla_iterations,
            lr=args.valla_lr,
            train_loader=train_loader,
            val_loader=val_loader,
            val_steps=args.valla_val_steps,
            verbose=args.verbose,
        )
        return

    if method == "mfvi":
        estimator.fit(  # type: ignore[call-arg]
            train_loader=train_loader,
            iterations=args.mfvi_iterations,
            verbose=args.verbose,
        )
        return

    if method == "sngp":
        estimator.fit(  # type: ignore[call-arg]
            train_loader=train_loader,
            iterations=args.sngp_iterations,
            lr=args.sngp_lr,
            weight_decay=args.sngp_weight_decay,
            verbose=args.verbose,
        )
        return


# ---------------------------------------------------------------------
# Evaluation (clean CIFAR10 + optional rotations)
# ---------------------------------------------------------------------


def evaluate_and_save(
    estimator: torch.nn.Module,
    method: str,
    transform,
    test_loader: DataLoader,
    args: argparse.Namespace,
    outdir: str,
    train_time: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], str]:
    """Evaluate on the clean CIFAR10 test set and (optionally) on rotated variants.

    Returns
    -------
    results : dict
        Dictionary of scalar metrics (including optional rotated_* metrics).
    csv_path : str
        Path to the saved CSV file.
    """
    method = method.lower()

    # Clean CIFAR10 test metrics
    t_test_start = time()
    if method in {"mfvi", "sngp", "map"}:
        results = score_samples(
            estimator,
            test_loader,
            SoftmaxClassificationSamples,
            verbose=True,
        )
    else:
        results = score(
            estimator,
            test_loader,
            SoftmaxClassification,
            verbose=True,
        )
    t_test_end = time()

    results["method"] = method
    results["dataset"] = "cifar10"
    results["resnet"] = args.resnet
    results["seed"] = args.seed
    results["train_time"] = train_time
    results["test_time"] = t_test_end - t_test_start

    # Optional: Rotated CIFAR10 metrics (10º to 180º)
    if args.test_rotations:
        angles = np.arange(10, 190, 10)
        for angle in angles:
            rotated_dataset = CIFAR10_Rotated_Dataset(angle=angle, transform=transform)
            _, rotated_test_ds = rotated_dataset.train_test_splits()
            rotated_loader = DataLoader(
                rotated_test_ds,
                batch_size=args.batch_size,
                shuffle=False,
            )

            if method in {"mfvi", "sngp", "map"}:
                rot_metrics = score_samples(
                    estimator,
                    rotated_loader,
                    SoftmaxClassificationSamples,
                    verbose=False,
                )
            else:
                rot_metrics = score(
                    estimator,
                    rotated_loader,
                    SoftmaxClassification,
                    verbose=False,
                )

            for key, value in rot_metrics.items():
                results[f"rotated_{angle}_{key}"] = value

    # Optional: CIFAR10 OOD evaluation
    if args.test_ood :
        ood_metrics = evaluate_ood(
            estimator,
            method,
            transform,
            args,
        )
        results["ood_auc"] = ood_metrics["auc"]
        if "auc_gauss" in ood_metrics:
            results["ood_auc_gauss"] = ood_metrics["auc_gauss"]

    # Optional: CIFAR10-C corrupted evaluation
    if args.test_corrupted:
        for severity in range(1, 6):
            for type in range(1, 20):
                corrupted_metrics = evaluate_corrupted(
                    estimator,
                    method,
                    transform,
                    severity,
                    type,
                    args,
                )
                for k, v in corrupted_metrics.items():
                    results[f"corrupted_sev{severity}_type{type}_{k}"] = v

    ensure_dir(outdir)
    df = pd.DataFrame.from_dict(results, orient="index").transpose()
    print(df)
    csv_name = make_results_filename(args, method)
    csv_path = os.path.join(outdir, csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return results, csv_path


# ---------------------------------------------------------------------
# CIFAR10 OOD evaluation
# ---------------------------------------------------------------------


def evaluate_ood(
    estimator: torch.nn.Module,
    method: str,
    transform,
    args: argparse.Namespace,
) -> Dict[str, str]:
    """Evaluate on the CIFAR10 OOD dataset and save labels/preds."""
    method = method.lower()

    ood_dataset = CIFAR10_OOD_Dataset(transform=transform)
    _, ood_test_ds = ood_dataset.train_test_splits()
    ood_loader = DataLoader(
        ood_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if method in {"mfvi", "sngp", "map"}:
        metrics = score_samples(
            estimator,
            ood_loader,
            OOD_Samples,
            verbose=False,
        )
    else:
        metrics = score(
            estimator,
            ood_loader,
            OOD,
            verbose=False,
        )

    return metrics


# ---------------------------------------------------------------------
# CIFAR10-C corrupted evaluation
# ---------------------------------------------------------------------


def evaluate_corrupted(
    estimator: torch.nn.Module,
    method: str,
    transform,
    severity,
    type,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Evaluate on the CIFAR10-C corrupted test set.

    Uses the test subset of CIFAR10-C (all corruption types, severity=5 by default).
    """
    method = method.lower()

    corrupted_dataset = CIFAR10C_Dataset(
        root="./data",
        transform=transform,
        severity=severity,
        type=type,
    )
    corrupted_loader = DataLoader(
        corrupted_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if method in {"mfvi", "sngp", "map"}:
        metrics = score_samples(
            estimator,
            corrupted_loader,
            SoftmaxClassificationSamples,
            verbose=False,
        )
    else:
        metrics = score(
            estimator,
            corrupted_loader,
            SoftmaxClassification,
            verbose=False,
        )

    return metrics


# ---------------------------------------------------------------------
# Filename helpers (to remain compatible with original scripts)
# ---------------------------------------------------------------------


def make_results_filename(args: argparse.Namespace, method: str) -> str:
    method = method.lower()
    if method == "lla":
        return f"lla_{args.lla_subset}_{args.lla_hessian}_{args.resnet}_{args.seed}.csv"
    if method == "ella":
        return f"ella_{args.batch_size}_{args.ella_lower_da}_{args.resnet}_{args.seed}.csv"
    # For other methods the original scripts only used method/resnet/seed
    return f"{method}_{args.resnet}_{args.seed}.csv"


def make_ood_filename(args: argparse.Namespace, method: str) -> str:
    method = method.lower()
    if method == "lla":
        return f"lla_{args.lla_subset}_{args.lla_hessian}_{args.resnet}_{args.seed}.txt"
    if method == "ella":
        return f"ella_{args.batch_size}_{args.ella_lower_da}_{args.resnet}_{args.seed}.txt"
    return f"{method}_{args.resnet}_{args.seed}.txt"


# ---------------------------------------------------------------------
# Argparse + MLflow helpers
# ---------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified CIFAR10 classification experiment runner",
    )

    # Optional config file
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file specifying all arguments",
    )

    # Core options
    p.add_argument("--verbose", action="store_true", help="Verbose")
    p.add_argument(
        "--output",
        type=str,
        default="results",
        help="Base output directory (default: results)",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["map", "fmgp", "lla", "ella", "valla", "mfvi", "sngp"],
        required=True,
    )
    p.add_argument(
        "--resnet",
        type=str,
        default="resnet20",
        help="ResNet backbone name (as in CIFAR10_Resnet)",
    )
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # Evaluation flags
    p.add_argument(
        "--test_rotations",
        action="store_true",
        help="Also evaluate on rotated CIFAR10 (10°–180°).",
    )
    p.add_argument(
        "--test_ood",
        action="store_true",
        help="Also evaluate on the CIFAR10 OOD dataset.",
    )
    p.add_argument(
        "--test_corrupted",
        action="store_true",
        help="Also evaluate on the CIFAR10-C corrupted test set.",
    )

    # FMGP hyper-parameters (classification)
    p.add_argument("--fmgp_kernel", type=str, default="RBFxNTK")
    p.add_argument(
        "--fmgp_inducing_locations",
        type=str,
        default="kmeans",
        choices=["kmeans", "random"],
    )
    p.add_argument("--fmgp_num_inducing", type=int, default=20)
    p.add_argument("--fmgp_iterations", type=int, default=50000)
    p.add_argument("--fmgp_lr", type=float, default=1e-4)
    p.add_argument("--fmgp_scheduler_gamma", type=float, default=0.85)

    # LLA hyper-parameters
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

    # ELLA hyper-parameters
    p.add_argument("--ella_subsample", type=int, default=2000)
    p.add_argument("--ella_n_eigenvalues", type=int, default=20)
    p.add_argument("--ella_prior", type=float, default=25.0)  # 1 / 0.04
    p.add_argument("--ella_lower_da", type=float, default=0.5)
    p.add_argument("--ella_val_steps", type=int, default=1)

    # VaLLA hyper-parameters
    p.add_argument(
        "--valla_inducing_locations",
        type=str,
        default="kmeans",
        choices=["kmeans", "random"],
    )
    p.add_argument("--valla_num_inducing", type=int, default=100)
    p.add_argument("--valla_iterations", type=int, default=40000)
    p.add_argument("--valla_lr", type=float, default=1e-3)
    p.add_argument("--valla_val_steps", type=int, default=500)

    # MFVI hyper-parameters
    p.add_argument("--mfvi_iterations", type=int, default=8000)
    p.add_argument("--mfvi_prior", type=float, default=1.0)
    p.add_argument("--mfvi_n_samples", type=int, default=20)

    # SNGP hyper-parameters
    p.add_argument("--sngp_iterations", type=int, default=5000)
    p.add_argument("--sngp_lr", type=float, default=1e-5)
    p.add_argument("--sngp_weight_decay", type=float, default=0.1)
    p.add_argument("--sngp_n_random_features", type=int, default=1024)

    return p


METHOD_PREFIXES = {
    "map": [],
    "fmgp": ["fmgp_"],
    "lla": ["lla_"],
    "ella": ["ella_"],
    "valla": ["valla_"],
    "mfvi": ["mfvi_"],
    "sngp": ["sngp_"],
}

CORE_PARAM_KEYS = {
    "method",
    "resnet",
    "batch_size",
    "seed",
    "test_rotations",
    "test_ood",
    "test_corrupted",
    "output",
    "verbose",
    "config",
}


def params_for_method(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    """Return a dict of params to log for the given method.

    - Always include 'core' params (method, resnet, seed, ...)
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    # If a config file is provided, override args from it
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            setattr(args, k, v)

    method = args.method.lower()

    # Device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = dtype_for_method(method)

    # Reproducibility (BayesiPy helper + additional seeding)
    assert_reproducibility(args.seed)
    seed = 2147483647 - int(args.seed)
    set_seed(seed)

    # Output directories
    base_outdir = args.output
    metrics_outdir = os.path.join(base_outdir, "cifar10")
    ensure_dir(metrics_outdir)

    if args.test_ood:
        ood_outdir = os.path.join(base_outdir, "cifar10_ood")
        ensure_dir(ood_outdir)
    else:
        ood_outdir = None

    # Save config
    config_path = save_config(args, metrics_outdir)

    # Load data & models for this method
    (
        base_model,
        embedding,
        classifier,
        transform,
        train_loader,
        val_loader,
        test_loader,
    ) = load_cifar10_and_resnet(method, args, device, dtype)

    # Build estimator (MAP / FMGP / LLA / ELLA / VaLLA / MFVI / SNGP)
    estimator = build_estimator(
        method,
        base_model,
        embedding,
        classifier,
        args,
        device,
        dtype,
    )

    # ---- MLflow tracking ----
    mlflow.set_experiment("bayesipy_cifar10")
    run_name = f"cifar10_{args.method}_{args.resnet}_seed{args.seed}"

    with mlflow.start_run(run_name=run_name):
        # Log params for this run
        method_params = params_for_method(args, args.method)
        mlflow.log_params(method_params)
        mlflow.log_artifact(config_path)

        # Fit (if applicable)
        t0 = time()
        fit_estimator(estimator, method, train_loader, val_loader, args)
        t1 = time()
        train_time = t1 - t0

        # Evaluate on CIFAR10 (and optionally rotations)
        results, csv_path = evaluate_and_save(
            estimator,
            method,
            transform,
            test_loader,
            args,
            outdir=metrics_outdir,
            train_time=train_time,
            device=device,
            dtype=dtype,
        )

        # Log metrics (only numeric values)
        numeric_results = {
            k: float(v)
            for k, v in results.items()
            if isinstance(v, numbers.Number)
        }
        mlflow.log_metrics(numeric_results)

        # Log CSV as artifact
        mlflow.log_artifact(csv_path)


if __name__ == "__main__":
    main()
