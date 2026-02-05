#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 × OpenAI CLIP × Fixed-Mean GP (BayesiPy FMGP)
- Pairs (image, class-text). Positives=1.0, sampled negatives=0.0
- Fixed mean = tau * cosine(clip_image, clip_text)
- Kernel = RBF over concatenated pair features X = [z_img ; z_txt]
- Evaluation = zero-shot: score each image vs 10 prompts, argmax

Run:
  python fmgp_cifar10_clip_bayesipy.py --batch-size 256 --negatives-per-image 3 --epochs 1
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
        z = z / z.norm(dim=-1, keepdim=True)
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
        z = z / z.norm(dim=-1, keepdim=True)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)


# ----------------------------
# Pair dataset construction
# ----------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def class_prompts(template: str = "a photo of a {}") -> List[str]:
    return [template.format(c) for c in CIFAR10_CLASSES]


class MulticlassDataset(torch.utils.data.Dataset):
    def __init__(self, Z_img, labels, Z_txt_all, negatives_per_image=3, tau=1.0):
        self.Z_img = Z_img
        self.labels = labels
        self.Z_txt_all = Z_txt_all
        self.neg_k = negatives_per_image
        self.tau = float(tau)              # (kept only if you log it; not needed for dataset)
        self.D = Z_img.shape[1]
        self.num_classes = Z_txt_all.shape[0]
        self.X, self.y = self._build()   

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    # <-- return ONLY (X, y)

    def _build(self):
        Xs, ys = [], []
        N = self.Z_img.shape[0]
        for i in range(N):
            zi = self.Z_img[i]
            yi = int(self.labels[i].item())
            zt = self.Z_txt_all.flatten()           
            Xs.append(torch.cat([zi, zt], dim=0))
            ys.append(torch.tensor(yi))
        
        X = torch.stack(Xs, dim=0)
        y = torch.stack(ys, dim=0)
        print(f"MulticlassDataset: {X.shape[0]} pairs, X.shape={X.shape}, y.shape={y.shape}")
        return X, y


# ----------------------------
# Fixed-mean module
# ----------------------------
class ClipMean(nn.Module):
    """
    Fixed mean: m(x) = tau * cosine(zI, zT) or affine(tau*cos(.)) if learn_affine=True.
    Works with X = [zI ; zT ; ...] (we compute cosine from the first 2D dims).
    """
    def __init__(self, D, tau=1.0, learn_affine=False):
        super().__init__()
        self.D = D
        self.learn_affine = learn_affine
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.b1 = nn.Parameter(torch.ones(1))
        self.register_buffer("tau_buf", torch.tensor(float(tau)))

    def forward(self, x):
        C = x.shape[-1] // self.D - 1
        # Get first D features
        zi = x[..., :self.D]                                    # (..., D)
        zt = x[..., self.D:].reshape(-1, C, self.D)                # (..., C, D)
        # recover v (take any column, or average across columns to be safe)
        zi = zi.unsqueeze(-2).contiguous()                  # (..., 512,)

        # recover M
        zt = zt.contiguous()                 # (10,512)
        cos = (zi * zt).sum(dim=-1)

        if self.learn_affine:
            base = self.b0 + self.b1 * self.tau * cos
        else:
            base = self.tau_buf * cos
        return base

# ----------------------------
# Build loaders, train, eval
# ----------------------------
def build_cifar_and_pairs(batch_size, negatives_per_image, device, subset_train=None, augment_features=False):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root="./data", train=True, transform=tfm, download=True)
    test_set  = datasets.CIFAR10(root="./data", train=False, transform=tfm, download=True)

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

    # Text embeddings (10 prompts)
    prompts = class_prompts("a photo of a {}")
    Z_txt_all = encode_texts(model, prompts, device=device)  # [10, D]
    D = Z_txt_all.shape[1]

    # Image embeddings
    Z_img_tr = encode_images(model, preprocess, Xtr, device=device, batch_size=256)  # [N,D]
    Z_img_te = encode_images(model, preprocess, Xte, device=device, batch_size=256)  # [Nt,D]

    # Given Z_img_val, y_val, Z_txt_all
    tau = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([tau], max_iter=100, lr=0.1)

    ZT = Z_txt_all.t()  # [D, C]
    def closure():
        opt.zero_grad()
        logits = tau * (Z_img_tr @ ZT)
        loss = torch.nn.functional.cross_entropy(logits, ytr, reduction="mean")
        loss.backward()
        return loss
    opt.step(closure)
    tau_star = tau.item()
    print(f"Learned tau (via CLIP zero-shot on train set): {tau_star:.4f} (init was 1.0)")

    # Pair dataset for training
    multiclass_train = MulticlassDataset(
        Z_img=Z_img_tr, labels=ytr, Z_txt_all=Z_txt_all,
        negatives_per_image=negatives_per_image, tau=tau_star
    )
    multiclass_test = MulticlassDataset(
        Z_img=Z_img_te, labels=yte, Z_txt_all=Z_txt_all,
        negatives_per_image=negatives_per_image, tau=tau_star
    )
    
    #train_loader = data.DataLoader(pair_train, batch_size=batch_size, shuffle=True, drop_last=False)
    train_loader = data.DataLoader(multiclass_train, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(multiclass_test, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, (Z_img_te, yte, Z_txt_all, D, tau_star)


def train_fmgp(train_loader, test_loader, feature_dim, tau, iterations=1, lr=1e-3, num_inducing=256, kernel="RBF", learn_affine=True):
    xb, yb = next(iter(train_loader))
    # If augmented, xb.size(1) > 2*feature_dim; mean model still uses first 2D dims.
    mean_model = ClipMean(D=feature_dim, tau=tau, learn_affine=learn_affine).to(get_device()).to(torch.float32)


    fmgp = FMGP(
        model=mean_model,
        likelihood="classification",
        kernel=kernel,
        inducing_locations="kmeans",
        mc_softmax_samples=1024,
        num_inducing=num_inducing,
    )

    loss = fmgp.fit(
        iterations=iterations,
        lr=lr,
        train_loader=train_loader,
        metrics_cls=SoftmaxClassification,
        verbose=True
    )
    try:
        loss, _ = loss
    except (TypeError, ValueError):
        pass
    import matplotlib.pyplot as plt
    # Mean loss per epoch
    plt.plot(np.arange(len(loss)) / len(train_loader), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("FMGP training loss")
    plt.show()
    return fmgp



@torch.no_grad()
def zero_shot_fmgp_metrics(fmgp: FMGP, Z_img_te: torch.Tensor, yte: torch.Tensor, Z_txt_all: torch.Tensor,
                           batch_size: int = 256):
    """
    Baseline (no FMGP): CLIP cosine similarities as logits:
      logits = tau * (z_img @ z_text^T)
    Returns (mean_nll, accuracy).
    """
    Nt, D = Z_img_te.shape
    C = Z_txt_all.shape[0]
    metrics = SoftmaxClassification()
    for i in range(0, Nt, batch_size):
   
        zi = Z_img_te[i:i+batch_size]                      # [B,D]
        B = zi.shape[0]
        zt = Z_txt_all.flatten().unsqueeze(0).repeat(B,1)   # [B,10*D]   

        X = torch.cat([zi, zt], dim=-1)                     # [B,10,2D]
        mu, var = fmgp.predict(X)                           # [B*C,1]
        labs = yte[i:i+mu.size(0)].long()
        
        metrics.update(y = labs.to(get_device()),
                       Fmean = mu.to(get_device()),
                       Fvar = var.to(get_device())
                       )
    print("FMGP zero-shot evaluation:")
    print(metrics.get_dict())




@torch.no_grad()
def zero_shot_baseline_metrics(Z_img_te: torch.Tensor, yte: torch.Tensor, Z_txt_all: torch.Tensor,
                               tau: float = 1.0, batch_size: int = 256):
    """
    Baseline (no FMGP): CLIP cosine similarities as logits:
      logits = tau * (z_img @ z_text^T)
    Returns (mean_nll, accuracy).
    """
    Nt, D = Z_img_te.shape
    ZT = Z_txt_all.t()  # [D, C]
    metrics = SoftmaxClassification()

    for i in range(0, Nt, batch_size):
        zi = Z_img_te[i:i+batch_size]          # [B,D]
        logits = tau * (zi @ ZT)               # [B,C]
        labs = yte[i:i+logits.size(0)].long()
        metrics.update(y = labs,
                       Fmean = logits.to(torch.float64),
                       Fvar = torch.eye(logits.size(1)).to(logits.device).to(torch.float64).unsqueeze(0).tile(logits.size(0),1,1) * 1e-6  # dummy near-zero var
                       )
    print("Baseline CLIP zero-shot evaluation:")
    print(metrics.get_dict())


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--negatives-per-image", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-inducing", type=int, default=200)
    ap.add_argument("--augment-features", action="store_true")
    ap.add_argument("--subset-train", type=int, default=None, help="use only first N train images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader, (Z_img_te, yte, Z_txt_all, D, tau) = build_cifar_and_pairs(
        batch_size=args.batch_size,
        negatives_per_image=args.negatives_per_image,
        device=device,
        subset_train=args.subset_train,
        augment_features=args.augment_features
    )

    # Baseline CLIP (no FMGP): NLL and Acc
    zero_shot_baseline_metrics(
        Z_img_te=Z_img_te,
        yte=yte,
        Z_txt_all=Z_txt_all,
        tau=tau,
        batch_size=args.batch_size
    )


    fmgp = train_fmgp(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_dim=D,
        tau=tau,
        iterations=args.iterations,
        lr=args.lr,
        num_inducing=args.num_inducing,
        kernel="DotProduct",
        learn_affine=False  # set False for strictly fixed mean (no affine calib)
    )

    zero_shot_fmgp_metrics(
        fmgp=fmgp,
        Z_img_te=Z_img_te,
        yte=yte,
        Z_txt_all=Z_txt_all,
        batch_size=args.batch_size,
    )



if __name__ == "__main__":
    main()
