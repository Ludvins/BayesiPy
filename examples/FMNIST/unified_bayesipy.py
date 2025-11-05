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
from typing import Tuple, Any
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

import sys

sys.path.append(".")

from bayesipy.utils.metrics import (
    SoftmaxClassification,
    SoftmaxClassificationSamples,
)  
from bayesipy.utils import safe_cholesky

# ----------------------------
# Model (must match your MNIST checkpoint)
# ----------------------------

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
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
            nn.Linear(64 * 3 * 3, 200),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = self.fc(x)
        return x
    
class ConvHead(nn.Module):
    def __init__(self, n_features: int, scale_params: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
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
            nn.Linear(64 * 3 * 3, 200),
            nn.ReLU(),
            nn.Linear(200, n_features * 10),
        )
        self.scale = torch.tensor(math.sqrt(scale_params / n_features))
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x).reshape(x.size(0), self.n_features, 10)
        return self.scale * x



# ----------------------------
# Data
# ----------------------------

def get_data(dir: str, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    transform=transforms.Compose([ transforms.ToTensor(),])
    train_ds = datasets.FashionMNIST(dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(dir, train=False, transform=transform)
    context_ds = datasets.MNIST(root=dir, train = True, download=True, transform=transform)
    ood_ds = datasets.KMNIST(dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    context_loader = DataLoader(context_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ood_loader = torch.utils.data.DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, context_loader, ood_loader

# ----------------------------
# Build BayesiPy inferencer by class name only
# ----------------------------

def build_inferencer(model: nn.Module, args) -> Any:

    if args.method == 'fmgp':
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
        
    if args.method == "map":
        return model  # No wrapper needed
    
    if args.method == "lla":
        from bayesipy.laplace import Laplace
        return Laplace(model,
                       likelihood="classification",
                       subset_of_weights=args.lla_subset,
                       hessian_structure=args.lla_hessian)
        
    if args.method == "ella":
        from bayesipy.laplace import ELLA
        return ELLA(model,
                    likelihood="classification",
                    subsample_size=args.ella_subsample,
                    n_eigenvalues=args.ella_n_eigenvalues,
                    prior_precision=args.ella_prior,
                    seed=args.seed)
        
    if args.method == "valla":
            from bayesipy.laplace import VaLLA
            return VaLLA(model,
                        likelihood="classification",
                        inducing_locations=args.valla_inducing_locations,
                        num_inducing=args.valla_num_inducing,
                        seed=args.seed)
            
    if args.method == "alla":
        from bayesipy.laplace import ScaLLA
        scale_params = sum(p.numel() for p in model.parameters())
        feature_model = ConvHead(n_features=args.alla_n_features, scale_params=scale_params)
        return ScaLLA(model,
                    feature_model,
                    likelihood="classification",
                    seed=args.seed)
        
    if args.method == "mfvi":
        from bayesipy.mfvi import MFVI
        return MFVI(model,
                    likelihood="classification",
                    prior_precision=args.mfvi_prior,
                    n_samples=args.mfvi_n_samples,
                    seed=args.seed)
        
    if args.method == "sngp":
        from bayesipy.sngp import SNGP
        return SNGP(model,
                    likelihood="classification",
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
    raise ValueError(f"Unknown method {args.method}")

def train(inferencer: Any, train_loader: DataLoader, context_loader: DataLoader, args: Any):
    if args.method == 'map':
        inferencer.eval()  # no training, just eval
        return

    # Update Hyper-parameters
    if args.method == "lla":
        inferencer.fit(
            train_loader=train_loader,
            progress_bar=args.verbose,
        )
        inferencer.optimize_prior_precision()
    if args.method == "ella":
        inferencer.fit(
            train_loader=train_loader,
            verbose=args.verbose,
        )
    if args.method == "valla":
        inferencer.fit(
            train_loader=train_loader,
            iterations=args.valla_iterations,
            lr=args.valla_lr,
            verbose=args.verbose,
        )
    if args.method == "alla":
        inferencer.fit(
            train_loader=train_loader,
            context_points_loader=context_loader,
            iterations=args.alla_iterations,
            lr=args.alla_lr,
            weight_decay=args.alla_weight_decay,
            optimize_hyper_parameters=args.alla_optimize_hyper_parameters,
            prior_opt_iterations=args.alla_prior_opt_iterations,
            prior_opt_lr=args.alla_prior_opt_lr,
            zero_crossed_variances=args.alla_zero_crossed_variances,
            verbose=args.verbose,
        )
    if args.method == "mfvi":
        inferencer.fit(
            iterations=args.mfvi_iterations,
            train_loader=train_loader,
            verbose=args.verbose,
        )
    if args.method == "sngp":
        inferencer.fit(
            train_loader=train_loader,
            weight_decay=args.sngp_weight_decay,
            iterations=args.sngp_iterations,
            lr=args.sngp_lr,
            verbose=args.verbose,
        )
    if args.method == "fmgp":
        inferencer.fit(
            train_loader=train_loader,
            iterations=args.fmgp_iterations,
            lr=args.fmgp_lr,
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
            chol = safe_cholesky(Fvar)
            z = torch.randn(
                2048,
                Fmean.shape[0],
                Fvar.shape[-1],
                device=Fmean.device,
                dtype=Fmean.dtype,
            )
            # Use re-parameterization Trick
            logit_samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
            # Get probailities
            prob_samples = logit_samples.softmax(-1)
            # Average and compute logarithm to scale to logit again
            probs = prob_samples.mean(0).softmax(-1)
            ent_gauss = 0.5 * torch.logdet(Fvar) + 0.5 * Fvar.shape[-1] * (1.0 + math.log(2 * math.pi))

        else:
            _, Fsamples = out_kind
            if id_metric is None:
                id_metric = SoftmaxClassificationSamples()
                id_metric.set_device_dtype(device, Fsamples.dtype)
                id_metric.reset()
            id_metric.update(y=ycol, F=Fsamples)
            probs = Fsamples.softmax(-1).mean(0)
            ent_gauss = torch.zeros(probs.shape[0], device=probs.device, dtype=probs.dtype)
        ent = -(probs * probs.log()).sum(-1)
        yhat = probs.argmax(-1)
        for e, ent_gauss, pmax, corr in zip(ent.cpu().tolist(), ent_gauss.cpu().tolist(), probs.max(-1).values.cpu().tolist(), (yhat.cpu()==y).numpy().tolist()):
            id_rows.append({'entropy': e, 'entropy_gauss': ent_gauss, 'max_prob': pmax, 'correct': int(corr)})
    id_summary = id_metric.get_dict()

    # OOD pass

    id_entropy_all = [r['entropy'] for r in id_rows]
    id_entropy_gaussian_all = [r['entropy_gauss'] for r in id_rows]
    for x, _ in ood_loader:
        x = x.to(device)
        out_kind = get_outputs(method, inferencer, model, x)
        if out_kind[0] == 'dist':
            _, Fmean, Fvar = out_kind
            chol = safe_cholesky(Fvar)
            z = torch.randn(
                2048,
                Fmean.shape[0],
                Fvar.shape[-1],
                device=Fmean.device,
                dtype=Fmean.dtype,
            )
            # Use re-parameterization Trick
            logit_samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
            # Get probailities
            prob_samples = logit_samples.softmax(-1)
            # Average and compute logarithm to scale to logit again
            probs = prob_samples.mean(0).softmax(-1)
            ent_gauss = 0.5 * torch.logdet(Fvar) + 0.5 * Fvar.shape[-1] * (1.0 + math.log(2 * math.pi))
        else:
            _, Fsamples = out_kind
            probs = Fsamples.softmax(-1).mean(0)
            ent_gauss = torch.zeros(probs.shape[0], device=probs.device, dtype=probs.dtype)
        ent = -(probs * probs.log()).sum(-1)
        for e, ent_gauss, pmax in zip(ent.cpu().tolist(), ent_gauss.cpu().tolist(), probs.max(-1).values.cpu().tolist()):
            ood_rows.append({'entropy': e, 'entropy_gauss': ent_gauss, 'max_prob': pmax})

    # Compute AUROC (higher entropy => OOD)
    try:
        from sklearn.metrics import roc_auc_score
        # our CSV entropies are fine too
        y = np.concatenate([np.zeros(len(id_entropy_all)), np.ones(len(ood_rows))])
        s = np.concatenate([np.array(id_entropy_all), np.array([r['entropy'] for r in ood_rows])])
        auroc = float(roc_auc_score(y, s))
    except Exception:
        auroc = float('0')
    
    try:
        from sklearn.metrics import roc_auc_score
        # our CSV entropies are fine too
        y = np.concatenate([np.zeros(len(id_entropy_gaussian_all)), np.ones(len(ood_rows))])
        s = np.concatenate([np.array(id_entropy_gaussian_all), np.array([r['entropy_gauss'] for r in ood_rows])])
        auroc_gauss = float(roc_auc_score(y, s))
    except Exception as e:
        print("Failed to compute Gaussian AUROC:", e)
        auroc_gauss = float('0')

    summary = dict(id_summary)
    summary['OOD_AUROC_Entropy'] = auroc
    summary['OOD_AUROC_Entropy_Gaussian'] = auroc_gauss
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary:', summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='./examples/FMNIST/results/fmnist_map.pt')
    p.add_argument('--outdir', type=str, default='./examples/FMNIST/results/map')
    p.add_argument('--data', type=str, default='./data', help='Directory to download data to')
    p.add_argument('--num-workers', type=int, default=2)    
    p.add_argument("--verbose", action="store_true", help="Verbose")
    p.add_argument("--method", type=str, choices=["map","fmgp","lla","ella","mfvi","sngp", "valla", "alla"], required=True)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="Seed")
    # ALLA
    p.add_argument("--alla-iterations", type=int, default=1_000_000)
    p.add_argument("--alla_lr", type=float, default=0.5e-4)
    p.add_argument("--alla_weight_decay", type=float, default=0.0)
    p.add_argument("--alla_n_features", type=int, default=100)
    p.add_argument("--alla_optimize_hyper_parameters", action="store_true")
    p.add_argument("--alla_prior_opt_iterations", type=int, default=1_000)
    p.add_argument("--alla_prior_opt_lr", type=float, default=0.1)
    p.add_argument("--alla_zero_crossed_variances", action="store_true")
    # FMGP
    p.add_argument("--fmgp_iterations", type=int, default=50000)
    p.add_argument("--fmgp_lr", type=float, default=1e-4)
    p.add_argument("--fmgp_kernel", type=str, default="RBFxNTK")
    p.add_argument("--fmgp_inducing_locations", type=str, default="kmeans", choices=['kmeans','random'])
    p.add_argument("--fmgp_num_inducing", type=int, default=200)
    # LLA
    p.add_argument("--lla_subset", type=str, default="last_layer", choices=["last_layer","all"])
    p.add_argument("--lla_hessian", type=str, default="kron", choices=["diag","kron","full"])
    # ELLA
    p.add_argument("--ella_subsample", type=int, default=1024)
    p.add_argument("--ella_n_eigenvalues", type=int, default=50)
    p.add_argument("--ella_prior", type=float, default=1.0)
    # VaLLA
    p.add_argument("--valla_inducing_locations", type=str, choices=['kmeans','random'], default='kmeans')
    p.add_argument("--valla_num_inducing", type=int, default=100)
    p.add_argument("--valla_iterations", type=float, default=15000)
    p.add_argument("--valla_lr", type=float, default=1e-3)
    # MFVI
    p.add_argument("--mfvi_iterations", type=int, default=8000)
    p.add_argument("--mfvi_prior", type=float, default=1.0)
    p.add_argument("--mfvi_n_samples", type=int, default=20)
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
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir = args.outdir.replace('map', args.method)
    
    if args.alla_zero_crossed_variances:
        args.outdir += '_zero_crossed_variances'
    
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'hparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model & load weights
    model = Net().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    try:
        model.load_state_dict(state)
    except Exception:
        if 'module.' not in next(iter(state.keys())):
            state = {f'module.{k}': v for k, v in state.items()}
        model.load_state_dict(state, strict=False)


    # Inferencer
    inferencer = build_inferencer(model, args)

    train_loader, test_loader, context_loader, ood_loader = get_data(args.data, args.batch_size, args.num_workers)
    
    # Train
    train(inferencer, train_loader, context_loader, args)

    # Eval
    evaluate(args.method, inferencer, model, test_loader, ood_loader, args.outdir, device)


if __name__ == '__main__':
    main()
