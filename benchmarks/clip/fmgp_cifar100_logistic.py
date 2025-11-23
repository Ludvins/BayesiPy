#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-100 × OpenAI CLIP × Fixed-Mean GP (BayesiPy FMGP)
- Mean function = Multiclass Logistic Regression (fitted on CLIP image embeddings)
- Inputs to FMGP: X = [z_img ; z_txt_all.flatten()], but mean uses only z_img
- Kernel: choose (e.g., RBF/DotProduct)
- Eval: score each image vs all class prompts (class order = CIFAR-100 classes)

Run:
  python fmgp_cifar100_clip_logreg_mean.py --batch-size 256 --epochs 1 --iterations 5000
"""

import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

import clip  # OpenAI CLIP

import sys
sys.path.append(".")
from bayesipy.fmgp import FMGP
from bayesipy.utils.metrics import SoftmaxClassification

# NEW: sklearn logistic regression
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# CLIP (OpenAI) encoders
# ----------------------------
def build_clip(model_name="ViT-B/32", device="cpu"):
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root='./models')
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_images(model, preprocess, images: torch.Tensor, device="cpu", batch_size=256) -> torch.Tensor:
    """
    images: [N, 3, 32, 32] float in [0,1]
    returns: [N, D] L2-normalized CLIP image embeddings
    """
    to_pil = transforms.ToPILImage()
    embs = []
    N = images.shape[0]
    for i in range(0, N, batch_size):
        chunk = images[i:i+batch_size].cpu()
        ims = torch.stack([preprocess(to_pil(img)) for img in chunk]).to(device)  # [B,3,224,224]
        z = model.encode_image(ims)
        #z = z / z.norm(dim=-1, keepdim=True)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_texts(model, texts: List[str], device="cpu", batch_size=256) -> torch.Tensor:
    """
    texts: list[str]
    returns: [M, D] L2-normalized CLIP text embeddings
    """
    embs = []
    for i in range(0, len(texts), batch_size):
        toks = clip.tokenize(texts[i:i+batch_size]).to(device)
        z = model.encode_text(toks)
        #z = z / z.norm(dim=-1, keepdim=True)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)


# ----------------------------
# Pair dataset construction
# ----------------------------
def class_prompts(template: str = "a photo of a {}") -> List[str]:
    # We only need the class order; CIFAR100 train split has .classes
    train_set = datasets.CIFAR100(root="./data", train=True, download=True)
    return [template.format(c) for c in train_set.classes]


class MulticlassDataset(torch.utils.data.Dataset):
    """
    Builds X = [z_img ; z_txt_all.flatten()] for each image.
    y = class index
    """
    def __init__(self, Z_img, labels, Z_txt_all):
        self.Z_img = Z_img
        self.labels = labels
        self.Z_txt_all = Z_txt_all
        self.D = Z_img.shape[1]
        self.num_classes = Z_txt_all.shape[0]
        self.X, self.y = self._build()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _build(self):
        Xs, ys = [], []
        N = self.Z_img.shape[0]
        zt_flat = self.Z_txt_all.flatten()  # [C*D]
        for i in range(N):
            zi = self.Z_img[i]
            yi = int(self.labels[i].item())
            Xs.append(zi)  # [D + C*D]
            ys.append(torch.tensor(yi))
        X = torch.stack(Xs, dim=0)
        y = torch.stack(ys, dim=0)
        print(f"MulticlassDataset: {X.shape[0]} items, X.shape={X.shape}, y.shape={y.shape}")
        return X, y


# ----------------------------
# NEW: Logistic-mean module
# ----------------------------
class LogisticMean(nn.Module):
    """
    Fixed mean: m(x) = (W z_img + b), where z_img is the first D dims of X
    Returns a length-C vector per input (class logits).
    The dataset passes X = [z_img ; z_txt_all.flatten()], but mean ignores text part.
    """
    def __init__(self, D: int, W: torch.Tensor, b: torch.Tensor):
        super().__init__()
        assert W.ndim == 2 and W.shape[1] == D, "W must be [C, D]"
        assert b.ndim == 1 and b.shape[0] == W.shape[0], "b must be [C]"
        # Register as buffers (fixed mean), not learnable parameters
        self.placeholder = nn.Parameter(torch.zeros(1))  # to appease FMGP checks
        self.register_buffer("W", W.clone().detach())
        self.register_buffer("b", b.clone().detach())
        self.D = D
        self.C = W.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = x @ self.W.t() + self.b       # (..., C)
        return logits


# ----------------------------
# Build loaders, train, eval
# ----------------------------
def build_cifar_and_pairs(batch_size, device, subset_train=None):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR100(root="./data", train=True, transform=tfm, download=True)
    test_set  = datasets.CIFAR100(root="./data", train=False, transform=tfm, download=True)

    if subset_train is not None and subset_train < len(train_set):
        idx = torch.randperm(len(train_set))[:subset_train]
        train_set = torch.utils.data.Subset(train_set, idx)

    # Stack tensors for CLIP encoding
    def stack_imgs_labels(ds):
        imgs, labs = [], []
        for img, lab in ds:
            imgs.append(img)
            labs.append(lab)
        return torch.stack(imgs, dim=0), torch.tensor(labs)

    Xtr, ytr = stack_imgs_labels(train_set)  # [N,3,32,32], [N]
    Xte, yte = stack_imgs_labels(test_set)

    # CLIP
    model, preprocess = build_clip(device=device)

    # Class prompts — we keep them to preserve class order (not used by mean)
    prompts = class_prompts("a photo of a {}")
    Z_txt_all = encode_texts(model, prompts, device=device)  # [C, D]
    D = Z_txt_all.shape[1]
    C = Z_txt_all.shape[0]
    print(f"Emb dim D={D}, num classes C={C}")

    # Image embeddings (L2-normalized)
    Z_img_tr = encode_images(model, preprocess, Xtr, device=device, batch_size=256)  # [N,D]
    Z_img_te = encode_images(model, preprocess, Xte, device=device, batch_size=256)  # [Nt,D]

    # Build pair datasets (X = [z_img ; z_txt_all.flatten()])
    multiclass_train = MulticlassDataset(
        Z_img=Z_img_tr, labels=ytr, Z_txt_all=Z_txt_all
    )
    multiclass_test = MulticlassDataset(
        Z_img=Z_img_te, labels=yte, Z_txt_all=Z_txt_all
    )

    train_loader = data.DataLoader(multiclass_train, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(multiclass_test, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, (Z_img_tr, ytr, Z_img_te, yte, Z_txt_all, D, C)


def fit_logreg_on_embeddings(Z_img_tr: torch.Tensor, ytr: torch.Tensor,
                             C_inv: float = 0.316, max_iter: int = 1000, seed: int = 0):
    """
    Fits multinomial logistic regression on CLIP image embeddings.
    Returns W [C,D], b [C] as torch.float32 on CPU, plus the sklearn classifier.
    """
    X = Z_img_tr.cpu().numpy()
    y = ytr.cpu().numpy().astype(int)
    
    
    clf = LogisticRegression(
        random_state=seed,
        C=C_inv,                # inverse of regularization strength's inverse; keeps user’s value
        max_iter=max_iter,
        verbose=1,
    )
    clf.fit(X, y)

    # sklearn stores coef_ as [C, D], intercept_ as [C]
    W = torch.from_numpy(clf.coef_).to(torch.float32)       # [C,D]
    b = torch.from_numpy(clf.intercept_).to(torch.float32)  # [C]
    return W, b, clf


def zero_shot_logreg_baseline(Z_img_te: torch.Tensor, yte: torch.Tensor,
                              W: torch.Tensor, b: torch.Tensor, batch_size: int = 256):
    """
    Baseline (no FMGP): logits = z_img @ W^T + b
    """
    metrics = SoftmaxClassification()
    Nt = Z_img_te.shape[0]
    for i in range(0, Nt, batch_size):
        zi = Z_img_te[i:i+batch_size].to(W.device).to(W.dtype)                 # [B,D]
        logits = zi @ W.t() + b                       # [B,C]
        labs = yte[i:i+logits.size(0)].long()
        metrics.update(
            y=labs,
            Fmean=logits.to(torch.float64),
            Fvar=torch.eye(logits.size(1), dtype=torch.float64).unsqueeze(0).tile(logits.size(0),1,1) * 1e-6
        )
    print("Baseline Logistic Regression evaluation:")
    print(metrics.get_dict())


def train_fmgp_with_logreg_mean(train_loader, test_loader, feature_dim, W, b,
                                iterations=1, lr=1e-3, num_inducing=256, kernel="RBF"):
    # Mean model uses only the first D dims (image), ignores concatenated text block.
    mean_model = LogisticMean(D=feature_dim, W=W, b=b).to(get_device()).to(torch.float32)

    fmgp = FMGP(
        model=mean_model,
        likelihood="classification",
        kernel=kernel,
        inducing_locations="kmeans",
        mc_softmax_samples=1024,
        num_inducing=num_inducing,
        subrogate_regularizer=False,
    )

    loss = fmgp.fit(
        iterations=iterations,
        lr=lr,
        train_loader=train_loader,
        val_loader=test_loader,
        val_steps=100,
        metrics_cls=SoftmaxClassification,
        verbose=True
    )
    try:
        loss, _ = loss
    except (TypeError, ValueError):
        pass

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(loss)) / len(train_loader), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("FMGP training loss (LogReg mean)")
    plt.show()
    return fmgp


@torch.no_grad()
def zero_shot_fmgp_metrics(fmgp: FMGP, Z_img_te: torch.Tensor, yte: torch.Tensor, Z_txt_all: torch.Tensor,
                           batch_size: int = 256):
    """
    FMGP evaluation: build X = [z_img ; z_txt_all.flatten()] and let the model return class logits
    """
    Nt, D = Z_img_te.shape
    metrics = SoftmaxClassification()

    for i in range(0, Nt, batch_size):
        zi = Z_img_te[i:i+batch_size]                      # [B,D]

        mu, var = fmgp.predict(zi)                          # [B, C] mean and [B, C, C] var (depending on impl)
        labs = yte[i:i+mu.size(0)].long().to(get_device())
        metrics.update(y=labs, Fmean=mu, Fvar=var)
    print("FMGP (LogReg-mean) evaluation:")
    print(metrics.get_dict())


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-inducing", type=int, default=200)
    ap.add_argument("--subset-train", type=int, default=None, help="use only first N train images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logreg_C", type=float, default=0.316)
    ap.add_argument("--logreg-max-iter", type=int, default=1000)
    ap.add_argument("--kernel", type=str, default="DotProduct")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader, (Z_img_tr, ytr, Z_img_te, yte, Z_txt_all, D, C) = build_cifar_and_pairs(
        batch_size=args.batch_size,
        device=device,
        subset_train=args.subset_train
    )

    # ---- Train Logistic Regression on image embeddings
    W, b, clf = fit_logreg_on_embeddings(
        Z_img_tr=Z_img_tr,
        ytr=ytr,
        C_inv=args.logreg_C,  # in case of shell hyphen quirk
        max_iter=args.logreg_max_iter,
        seed=args.seed
    )

    # ---- Baseline (no FMGP): evaluate LR directly
    zero_shot_logreg_baseline(
        Z_img_te=Z_img_te,
        yte=yte,
        W=W,
        b=b,
        batch_size=args.batch_size
    )

    # ---- FMGP with Logistic mean
    fmgp = train_fmgp_with_logreg_mean(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_dim=D,
        W=W,
        b=b,
        iterations=args.iterations,
        lr=args.lr,
        num_inducing=args.num_inducing,
        kernel=args.kernel
    )

    # ---- FMGP evaluation
    zero_shot_fmgp_metrics(
        fmgp=fmgp,
        Z_img_te=Z_img_te,
        yte=yte,
        Z_txt_all=Z_txt_all,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
