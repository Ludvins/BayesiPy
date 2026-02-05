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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, Z_img, labels, Z_txt_all, negatives_per_image=3, tau=1.0, augment=False):
        self.Z_img = Z_img
        self.labels = labels
        self.Z_txt_all = Z_txt_all
        self.neg_k = negatives_per_image
        self.tau = float(tau)              # (kept only if you log it; not needed for dataset)
        self.augment = augment
        self.D = Z_img.shape[1]
        self.num_classes = Z_txt_all.shape[0]
        self.X, self.y = self._build()   

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _feat(self, zi, zt):
        if not self.augment:
            return torch.cat([zi, zt], dim=-1)

    def _build(self):
        Xs, ys = [], []
        N = self.Z_img.shape[0]
        for i in range(N):
            zi = self.Z_img[i]
            yi = int(self.labels[i].item())
            # positive (target=1)
            zt_pos = self.Z_txt_all[yi]
            Xs.append(self._feat(zi, zt_pos))
            ys.append(torch.tensor([1.0]))
            # negatives (target=0)
            negs = [c for c in range(self.num_classes) if c != yi]
            random.shuffle(negs)
            for k in range(self.neg_k):
                cj = negs[k % len(negs)]
                zt_neg = self.Z_txt_all[cj]
                Xs.append(self._feat(zi, zt_neg))
                ys.append(torch.tensor([0.0]))
        X = torch.stack(Xs, dim=0)
        y = torch.stack(ys, dim=0)
        eps = 0.01
        y_smooth = y.float().clamp_min(eps).clamp_max(1-eps)
        t = torch.log(y_smooth) - torch.log1p(-y_smooth)  # logit
        print(f"PairDataset: {X.shape[0]} pairs, X.shape={X.shape}, y.shape={y.shape}, augment={self.augment}")

        return X, t



# ----------------------------
# Fixed-mean module
# ----------------------------
class ClipMean(nn.Module):
    """
    Fixed mean: m(x) = tau * cosine(zI, zT) or affine(tau*cos(.)) if learn_affine=True.
    Works with X = [zI ; zT ; ...] (we compute cosine from the first 2D dims).
    """
    def __init__(self, D, a=1.0, b=0.0, learn_affine=False):
        super().__init__()
        self.D = D
        self.learn_affine = learn_affine
        self.b0 = nn.Parameter(torch.zeros(1))
        self.a = a
        self.b = b

    def forward(self, x):
        zI, zT = x[..., :self.D], x[..., self.D:2*self.D]
        cos = (zI * zT).sum(dim=-1)
        if self.learn_affine:
            base = self.b0 + self.tau_buf * cos
        else:
            base = self.a * cos + self.b
        return base.unsqueeze(-1)


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
    pair_train = PairDataset(
        Z_img=Z_img_tr, labels=ytr, Z_txt_all=Z_txt_all,
        negatives_per_image=negatives_per_image, tau=tau_star, augment=augment_features
    )
    train_loader = data.DataLoader(pair_train, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, pair_train, (Z_img_te, yte, Z_txt_all, D, tau_star)


def train_fmgp(train_loader, feature_dim, tau, epochs=1, lr=1e-3, num_inducing=256, kernel="RBF", learn_affine=True, a=1.0, b=0.0):
    xb, yb = next(iter(train_loader))
    # If augmented, xb.size(1) > 2*feature_dim; mean model still uses first 2D dims.
    mean_model = ClipMean(D=feature_dim, a=a, b=b, learn_affine=False)

    fmgp = FMGP(
        model=mean_model,
        likelihood="regression",
        kernel=kernel,
        inducing_locations="kmeans",
        num_inducing=num_inducing,
        noise_variance=np.exp(-5),
    )
    print("Initial fmgp.bias:", fmgp.bias, "fmgp.scale:", fmgp.scale)
    loss = fmgp.fit(
        iterations=epochs*len(train_loader),
        lr=lr,
        train_loader=train_loader,
        verbose=True
    )
    print("Final fmgp.bias:", fmgp.bias, "fmgp.scale:", fmgp.scale)
    import matplotlib.pyplot as plt
    # Average loss per epoch
    loss = np.array(loss).reshape(epochs, -1).mean(axis=1)
    
    plt.plot(np.arange(len(loss)), loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("FMGP training loss")
    plt.grid()
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
    total_nll = 0.0
    correct = 0
    total = 0

    for i in range(0, Nt, batch_size):
        zi = Z_img_te[i:i+batch_size]                      # [B,D]
        B = zi.shape[0]
        zi_exp = zi.unsqueeze(1).expand(B, C, D)          # [B,10,D]
        zt_exp = Z_txt_all.unsqueeze(0).expand(B, C, D)   # [B,10,D]

        X = torch.cat([zi_exp, zt_exp], dim=-1)       # [B,10,2D]

        X = X.reshape(B*C, -1)
        mu, var = fmgp.predict(X)                           # [B*C,1]
        samples = mu + torch.randn(512, B*C, 1, device=mu.device) * torch.sqrt(var + 1e-8)  # [B*C,1]
        logits = samples.reshape(512, B, C)                        # treat GP mean as logits
        probs = torch.softmax(logits, dim=-1)               # [512,B,C]
        probs = probs.mean(dim=0)                      # [B,C]
        labs = yte[i:i+B].long()
        nll = torch.nn.functional.cross_entropy(probs.log(), labs, reduction="sum")

        total_nll += float(nll.item())
        preds = probs.argmax(dim=-1)
        correct += (preds.cpu() == labs.cpu()).sum().item()
        total += probs.size(0)

    return total_nll / float(total), correct / float(total)




@torch.no_grad()
def zero_shot_baseline_metrics(Z_img_te: torch.Tensor, yte: torch.Tensor, Z_txt_all: torch.Tensor,
                               a: float = 1.0, b: float = 0.0, batch_size: int = 256):
    """
    Multiclass baseline using logits = a * (z_img @ z_text^T) + b.
    Note: b cancels in softmax per image, so only 'a' matters for CE/argmax.
    Returns (mean_nll, accuracy).
    """
    Nt, D = Z_img_te.shape
    ZT = Z_txt_all.t()  # [D, C]
    total_nll = 0.0
    correct = 0
    total = 0

    for i in range(0, Nt, batch_size):
        zi = Z_img_te[i:i+batch_size]          # [B,D]
        logits = a * (zi @ ZT) + b             # [B,C]  (b irrelevant for softmax)
        labs = yte[i:i+logits.size(0)].long()
        nll = torch.nn.functional.cross_entropy(logits, labs, reduction="sum")
        total_nll += float(nll.item())
        preds = logits.argmax(dim=-1)
        correct += (preds.cpu() == labs.cpu()).sum().item()
        total += logits.size(0)

    return total_nll / float(total), correct / float(total)



# ----------------------------
# Logistic calibration on cosine
# ----------------------------
def fit_logistic_scale(cosines: torch.Tensor, labels: torch.Tensor, pos_weight=None, lr=0.1, steps=200):
    """
    Learn a,b so that p(y=1|cos) = sigmoid(a*cos + b).
    cosines: [N], labels: [N] in {0,1}
    """
    device = cosines.device
    a = torch.nn.Parameter(torch.tensor(1.0, device=device))
    b = torch.nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        logits = a * cosines + b
        if pos_weight is None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(),
                pos_weight=torch.tensor(float(pos_weight), device=device)
            )
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().item(), b.detach().item()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--negatives-per-image", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-inducing", type=int, default=256)
    ap.add_argument("--augment-features", action="store_true")
    ap.add_argument("--subset-train", type=int, default=None, help="use only first N train images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, pair_train, (Z_img_te, yte, Z_txt_all, D, tau) = build_cifar_and_pairs(
        batch_size=args.batch_size,
        negatives_per_image=args.negatives_per_image,
        device=device,
        subset_train=args.subset_train,
        augment_features=args.augment_features
    )
        
    X_pairs = pair_train.X           # [N_pairs, 2D]
    y_pairs = pair_train.y.squeeze(1)  # [N_pairs]
    # y_pairs positive to 1 and negative to 0
    y_pairs_ = (y_pairs > 0).float()

    D_feat = D
    zI = X_pairs[:, :D_feat]
    zT = X_pairs[:, D_feat:2*D_feat]
    cos_pairs = (zI * zT).sum(dim=-1)

    # pos_weight to counter neg>>pos (approx ratio)
    num_pos = (y_pairs == 1).sum().item()
    num_total = y_pairs.numel()
    num_neg = num_total - num_pos
    pos_w = (num_neg / max(1, num_pos)) if num_pos > 0 else None

    a_cal, b_cal = fit_logistic_scale(cos_pairs, y_pairs_, pos_weight=pos_w, lr=0.1, steps=200)
    print(f"[Calibration] logistic: a={a_cal:.4f}, b={b_cal:.4f}, pos_weight={pos_w}")

    # Baseline CLIP (no FMGP): NLL and Acc
    base_nll, base_acc = zero_shot_baseline_metrics(
        Z_img_te=Z_img_te,
        yte=yte,
        Z_txt_all=Z_txt_all,
        batch_size=args.batch_size,
        a=a_cal,
        b=b_cal
    )
    print(f"Zero-shot (CLIP baseline) — Acc: {base_acc*100:.2f}% | NLL: {base_nll:.4f}")


    fmgp = train_fmgp(
        train_loader=train_loader,
        feature_dim=D,
        tau=1.0,
        epochs=args.epochs,
        lr=args.lr,
        num_inducing=args.num_inducing,
        kernel="RBF",
        learn_affine=False,  # set False for strictly fixed mean (no affine calib)
        a=a_cal, b=b_cal
    )

    nll, acc =zero_shot_fmgp_metrics(
        fmgp=fmgp,
        Z_img_te=Z_img_te,
        yte=yte,
        Z_txt_all=Z_txt_all,
        batch_size=args.batch_size,
    )
    print(f"Zero-shot (FMGP) — Acc: {acc*100:.2f}% | NLL: {nll:.4f}")



if __name__ == "__main__":
    main()
