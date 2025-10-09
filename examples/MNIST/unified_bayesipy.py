"""
mnist_unified_bayesipy_metrics.py

Unified MNIST evaluator using BayesiPy class names (MAP, LLA, ELLA, MFVI, SNGP, FMGP)
**and** BayesiPy metrics:
  - SoftmaxClassification (predictive distribution: Fmean, Fvar)
  - SoftmaxClassificationSamples (logit samples: [S, B, C])
  - OOD (predictive distribution)
  - OOD_Samples (logit samples)

It automatically picks the right metrics class depending on what the chosen
method outputs. OOD benchmark uses Fashion-MNIST via predictive-entropy; we also
report AUROC.

Examples
--------
# MAP (with pretrained weights)
python mnist_unified_bayesipy_metrics.py \
  --method map --weights ./runs/mnist_resnet18/best_mnist.pt \
  --outdir ./eval/map

# LLA post-hoc
python mnist_unified_bayesipy_metrics.py \
  --method lla --weights ./runs/mnist_resnet18/best_mnist.pt \
  --lla-prior 1.0 --outdir ./eval/lla

# FMGP with hyper-params
python mnist_unified_bayesipy_metrics.py \
  --method fmgp --weights ./runs/mnist_resnet18/best_mnist.pt \
  --fmgp-M 512 --fmgp-kernel rbf --fmgp-lengthscale 1.0 --fmgp-variance 1.0 \
  --outdir ./eval/fmgp
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Any
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import sys
sys.path.append(".")

from bayesipy.utils.metrics import (
    SoftmaxClassification,
    SoftmaxClassificationSamples,
)  # type: ignore

# ----------------------------
# Model (must match your MNIST checkpoint)
# ----------------------------

def build_resnet18_mnist(num_classes: int = 10) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ----------------------------
# Data
# ----------------------------

def get_data(root: str, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=train_tf)
    test = datasets.MNIST(root=root, train=False, download=True, transform=test_tf)
    ood_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ood_loader = DataLoader(ood_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, ood_loader

# ----------------------------
# Build BayesiPy inferencer by class name only
# ----------------------------

def build_inferencer(method: str, model: nn.Module, args) -> Any:

    if method == 'fmgp':
        from bayesipy.fmgp import FMGP
        embedding = copy.deepcopy(model)
        embedding.fc = nn.Identity()
        classifier = model.fc
        return FMGP(embedding=embedding,
                    classifier=classifier,
                    likelihood="classification",
                    kernel=args.fmgp_kernel,
                    inducing_locations=args.fmgp_inducing_locations,
                    num_inducing=args.fmgp_num_inducing,
                    seed=args.seed)
    if method == "map":
        return model  # No wrapper needed
    if method == "lla":
        from bayesipy.laplace import Laplace
        return Laplace(model,
                       likelihood="classification",
                       subset_of_weights=args.lla_subset,
                       hessian_structure=args.lla_hessian)
    if method == "ella":
        from bayesipy.laplace import ELLA
        return ELLA(model,
                    likelihood="classification",
                    subsample_size=args.ella_subsample,
                    n_eigenvalues=args.ella_n_eigenvalues,
                    prior_precision=args.ella_prior,
                    seed=args.seed)
    if method == "valla":
            from bayesipy.laplace import VaLLA
            return VaLLA(model,
                        likelihood="classification",
                        inducing_locations=args.valla_inducing_locations,
                        num_inducing=args.valla_num_inducing,
                        seed=args.seed)
    if method == "mfvi":
        from bayesipy.mfvi import MFVI
        return MFVI(model,
                    likelihood="classification",
                    prior_precision=args.mfvi_prior,
                    n_samples=args.mfvi_n_samples,
                    seed=args.seed)
    if method == "sngp":
        from bayesipy.sngp import SNGP
        return SNGP(model,
                    gp_kernel_scale=args.sngp_kernel_scale,
                    n_random_features=args.sngp_n_random_features,
                    gp_output_bias=args.sngp_gp_output_bias,
                    layer_norm_eps=args.sngp_layer_norm_eps,
                    n_power_iterations=args.sngp_n_power_iterations,
                    scale_random_features=args.sngp_scale_random_features,
                    normalize_input=args.sngp_normalize_input,
                    gp_cov_momentum=args.sngp_gp_cov_momentum,
                    gp_cov_ridge_penalty=args.sngp_gp_cov_ridge_penalty,
                    seed=args.seed)
    raise ValueError(f"Unknown method {method}")

def train(method: str, inferencer: Any, train_loader: DataLoader, args: Any):
    if method == 'map':
        inferencer.eval()  # no training, just eval
        return

    # Update Hyper-parameters
    if method == "lla":
        inferencer.fit(
            train_loader=train_loader,
            progress_bar=args.verbose,
        )
        inferencer.optimize_prior_precision()
    if method == "ella":
        inferencer.fit(
            train_loader=train_loader,
            progress_bar=args.verbose,
        )
    if method == "valla":
        inferencer.fit(
            train_loader=train_loader,
            iterations=args.iterations,
            lr=args.lr,
            verbose=args.verbose,
        )
    if method == "mfvi":
        inferencer.fit(
            iterations=args.iterations,
            train_loader=train_loader,
            verbose=args.verbose,
        )
    if method == "sngp":
        inferencer.fit(
            train_loader=train_loader,
            epochs=args.epochs,
            lr=args.lr,
            verbose=args.verbose,
        )
    if method == "fmgp":
        inferencer.fit(
            train_loader=train_loader,
            iterations=args.iterations,
            lr=args.lr,
            verbose=args.verbose,
        )

# ----------------------------
# Introspect output type & get outputs
# ----------------------------

def get_outputs(method: str, inferencer: Any, model: nn.Module, x: torch.Tensor):
    """Return either:
       - ("dist", Fmean[B,C], Fvar[B,C,C]) for predictive distributions, or
       - ("samples", Fsamples[S,B,C]) for Monte Carlo logit samples.
    For MAP, we synthesize a tiny covariance and report a distribution.
    """
    with torch.no_grad():
        if method == 'map':
            logits = model(x)
            B, C = logits.shape
            Fvar = torch.eye(C, device=logits.device, dtype=logits.dtype).unsqueeze(0).repeat(B,1,1) * 1e-8
            return 'dist', logits, Fvar
        
        out = inferencer.predict(x)
        try:
            fmean, fvar = out
            # If fvar is (B, B, C, C), get diagonal (B, C, C)
            if fvar.ndim == 4:
                fvar = fvar.diagonal(dim1=0, dim2=1).permute(2,0,1)
            return 'dist', fmean, fvar
        except Exception:
            return 'samples', out  # assume samples [S,B,C]

# ----------------------------
# Evaluation
# ----------------------------

def evaluate(method: str, inferencer: Any, model: nn.Module, id_loader: DataLoader, ood_loader: DataLoader, outdir: str, device: str):
    os.makedirs(outdir, exist_ok=True)

    # ID metrics (SoftmaxClassification vs SoftmaxClassificationSamples)
    # OOD metrics likewise
    # Also collect per-sample entropy & correctness to CSV, and AUROC
    id_rows = []
    ood_rows = []

    # ID pass
    id_metric = None
    for x, y in id_loader:
        x = x.to(device)
        ycol = y.to(device).view(-1, 1)
        out_kind = get_outputs(method, inferencer, model, x)
        if out_kind[0] == 'dist':
            _, Fmean, Fvar = out_kind
            if id_metric is None:
                id_metric = SoftmaxClassification()
                id_metric.set_device_dtype(device, Fmean.dtype)
                id_metric.reset()
            id_metric.update(y=ycol, Fmean=Fmean, Fvar=Fvar)
            probs = Fmean.softmax(-1)
        else:
            _, Fsamples = out_kind
            if id_metric is None:
                id_metric = SoftmaxClassificationSamples()
                id_metric.set_device_dtype(device, Fsamples.dtype)
                id_metric.reset()
            id_metric.update(y=ycol, F=Fsamples)
            probs = Fsamples.softmax(-1).mean(0)
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
        yhat = probs.argmax(-1)
        for e, pmax, corr in zip(ent.cpu().tolist(), probs.max(-1).values.cpu().tolist(), (yhat.cpu()==y).numpy().tolist()):
            id_rows.append({'entropy': e, 'max_prob': pmax, 'correct': int(corr)})
    id_summary = id_metric.get_dict()

    # OOD pass

    id_entropy_all = [r['entropy'] for r in id_rows]
    for x, _ in ood_loader:
        x = x.to(device)
        out_kind = get_outputs(method, inferencer, model, x)
        if out_kind[0] == 'dist':
            _, Fmean, Fvar = out_kind
            probs = Fmean.softmax(-1)
        else:
            _, Fsamples = out_kind
            probs = Fsamples.softmax(-1).mean(0)
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
        for e, pmax in zip(ent.cpu().tolist(), probs.max(-1).values.cpu().tolist()):
            ood_rows.append({'entropy': e, 'max_prob': pmax})

    # Compute AUROC (higher entropy => OOD)
    try:
        from sklearn.metrics import roc_auc_score
        # our CSV entropies are fine too
        y = np.concatenate([np.zeros(len(id_entropy_all)), np.ones(len(ood_rows))])
        s = np.concatenate([np.array(id_entropy_all), np.array([r['entropy'] for r in ood_rows])])
        auroc = float(roc_auc_score(y, s))
    except Exception:
        auroc = float('nan')

    summary = dict(id_summary)
    summary['OOD_AUROC_Entropy'] = auroc
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary:', summary)

# ----------------------------
# CLI
# ----------------------------

@dataclass
class Args:
    method: str = 'map'
    weights: str = './examples/MNIST/mnist_resnet18/best_mnist.pt'
    outdir: str = './examples/MNIST/results/'+ method
    data: str = './data'
    batch_size: int = 32
    num_workers: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Hyper-params per method
    # LLA
    lla_subset: str = 'all'  # 'all', 'last'
    lla_hessian: str = 'full'  # 'full', 'kron', 'diag'
    # ELLA
    ella_subsample: int = 1024
    ella_n_eigenvalues: int = 200
    # FMGP
    fmgp_inducing_locations: str = 'KMEANS' 
    fmgp_num_inducing: int = 512
    fmgp_kernel: str = 'RBFxNTK'  
    # MFVI
    mfvi_prior: float = 1.0
    mfvi_n_samples: int = 20
    # SNGP
    sngp_kernel_scale: float = 2.0
    sngp_n_random_features: int = 1024
    sngp_gp_output_bias: bool = True
    sngp_layer_norm_eps: float = 1e-6
    sngp_n_power_iterations: int = 1
    sngp_scale_random_features: bool = True
    sngp_normalize_input: bool = False
    sngp_gp_cov_momentum: float = 0.999
    sngp_gp_cov_ridge_penalty: float = 1.0
    # General
    seed: int = 0
    
    


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, choices=['map','lla','ella','mfvi','sngp','fmgp','valla'], default=Args.method)
    p.add_argument('--weights', type=str, default=Args.weights)
    p.add_argument('--outdir', type=str, default=Args.outdir)
    p.add_argument('--data', type=str, default=Args.data)
    p.add_argument('--batch-size', type=int, default=Args.batch_size)
    p.add_argument('--num-workers', type=int, default=Args.num_workers)
    # Hyper-params
    # LLA
    p.add_argument('--lla-subset', type=str, choices=['all','last_layer'], default=Args.lla_subset)
    p.add_argument('--lla-hessian', type=str, choices=['full','kron','diag'], default=Args.lla_hessian)
    # ELLA
    p.add_argument('--ella-subsample', type=int, default=Args.ella_subsample)
    p.add_argument('--ella-n-eigenvalues', type=int, default=Args.ella_n_eigenvalues)
    p.add_argument('--ella-prior', type=float, default=1.0)
    p.add_argument('--ella-seed', type=int, default=0)
    # VALLA
    p.add_argument('--valla-inducing-locations', type=str, choices=['kmeans','random'], default='kmeans')
    p.add_argument('--valla-num-inducing', type=int, default=512)
    # FMGP
    p.add_argument('--fmgp-inducing-locations', type=str, choices=['kmeans','random'], default=Args.fmgp_inducing_locations)
    p.add_argument('--fmgp-num-inducing', type=int, default=Args.fmgp_num_inducing)
    p.add_argument('--fmgp-kernel', type=str, choices=['RBF','RBFxNTK'], default=Args.fmgp_kernel)
    # MFVI 
    p.add_argument('--mfvi-prior', type=float, default=Args.mfvi_prior)
    p.add_argument('--mfvi-n-samples', type=int, default=Args.mfvi_n_samples)
    # SNGP
    p.add_argument('--sngp-kernel-scale', type=float, default=Args.sngp_kernel_scale)
    p.add_argument('--sngp-n-random-features', type=int, default=Args.sngp_n_random_features)
    p.add_argument('--sngp-gp-output-bias', action='store_true', default=Args.sngp_gp_output_bias)
    p.add_argument('--sngp-no-gp-output-bias', dest='sngp_gp_output_bias', action='store_false')
    p.add_argument('--sngp-layer-norm-eps', type=float, default=Args.sngp_layer_norm_eps)
    p.add_argument('--sngp-n-power-iterations', type=int, default=Args.sngp_n_power_iterations)
    p.add_argument('--sngp-scale-random-features', action='store_true', default=Args.sngp_scale_random_features)
    p.add_argument('--sngp-no-scale-random-features', dest='sngp_scale_random_features', action='store_false')
    p.add_argument('--sngp-normalize-input', action='store_true', default=Args.sngp_normalize_input)
    p.add_argument('--sngp-no-normalize-input', dest='sngp_normalize_input', action='store_false')
    p.add_argument('--sngp-gp-cov-momentum', type=float, default=Args.sngp_gp_cov_momentum)
    p.add_argument('--sngp-gp-cov-ridge-penalty', type=float, default=Args.sngp_gp_cov_ridge_penalty)
    # General
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--device', type=str, default=Args.device)
    p.add_argument('--verbose', action='store_true', default=False)
    p.add_argument('--no-verbose', dest='verbose', action='store_false')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir = args.outdir.replace('map', args.method)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'hparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model & load weights
    model = build_resnet18_mnist().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    try:
        model.load_state_dict(state)
    except Exception:
        if 'module.' not in next(iter(state.keys())):
            state = {f'module.{k}': v for k, v in state.items()}
        model.load_state_dict(state, strict=False)


    # Inferencer
    inferencer = build_inferencer(args.method, model, args)

    train_loader, test_loader, ood_loader = get_data(args.data, args.batch_size, args.num_workers)
    
    # Train
    train(args.method, inferencer, train_loader, args)

    # Eval
    evaluate(args.method, inferencer, model, test_loader, ood_loader, args.outdir, device)


if __name__ == '__main__':
    main()
