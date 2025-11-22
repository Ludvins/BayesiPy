"""
train_map.py — MNIST trainer (PyTorch 2.x)

Run examples:
    python examples/MNIST/train_map.py --outdir ./examples/MNIST/mnist_resnet18 --epochs 40 --batch-size 512 --lr 0.2 --swa
    python examples/MNIST/train_map.py --outdir ./examples/MNIST/mnist_resnet18 --epochs 60 --swa --swa-start 40
    a

This recipe typically reaches ≥99.7% test accuracy on MNIST in ~30-60 epochs with a single GPU.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import json
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------------
# Utilities
# ----------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


# ----------------------------
# Model
# ----------------------------

def build_resnet18_mnist(num_classes: int = 10) -> nn.Module:
    """ResNet18 adapted for 28x28 grayscale input.
    - 3x3 conv stem, stride=1, no maxpool
    - 1 input channel
    """
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ----------------------------
# Data
# ----------------------------

def get_data(root: str, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    # Avoid flips (6↔9). Keep moderate rotation & translation.
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2, padding_mode="reflect"),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=train_tf)
    test = datasets.MNIST(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# Random Erasing (tensor space)
# ----------------------------

def random_erasing_batch(x: torch.Tensor, p: float = 0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)) -> torch.Tensor:
    if p <= 0.0:
        return x
    B, C, H, W = x.shape
    for i in range(B):
        if random.random() > p:
            continue
        area = H * W
        for _ in range(10):
            target = random.uniform(*scale) * area
            r = random.uniform(*ratio)
            h = int(round(math.sqrt(target * r)))
            w = int(round(math.sqrt(target / r)))
            if 0 < h < H and 0 < w < W:
                y0 = random.randint(0, H - h)
                x0 = random.randint(0, W - w)
                x[i, :, y0:y0 + h, x0:x0 + w] = 0.0
                break
    return x


# ----------------------------
# Train / Eval
# ----------------------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    scaler: torch.amp.GradScaler | None,
                    device: str,
                    criterion: nn.Module,
                    use_random_erasing: bool = True) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_random_erasing:
            x = random_erasing_batch(x, p=0.25, scale=(0.02, 0.12))

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (logits.argmax(1) == y).float().sum().item()
        n += bs

    return total_loss / n, total_acc / n


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += (logits.argmax(1) == y).float().sum().item()
            n += bs
    return total_loss / n, total_acc / n


# ----------------------------
# Args
# ----------------------------

@dataclass
class Args:
    epochs: int = 40
    batch_size: int = 512
    lr: float = 0.2
    wd: float = 5e-4
    momentum: float = 0.9
    label_smoothing: float = 0.1
    data: str = "./data"
    seed: int = 42
    swa: bool = False
    swa_start: int = 30  # 0-indexed epoch to start SWA averaging
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    outdir: str = "./outputs"


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=Args.epochs)
    p.add_argument('--batch-size', type=int, default=Args.batch_size)
    p.add_argument('--lr', type=float, default=Args.lr)
    p.add_argument('--wd', type=float, default=Args.wd)
    p.add_argument('--momentum', type=float, default=Args.momentum)
    p.add_argument('--label-smoothing', type=float, default=Args.label_smoothing)
    p.add_argument('--data', type=str, default=Args.data)
    p.add_argument('--seed', type=int, default=Args.seed)
    p.add_argument('--swa', action='store_true')
    p.add_argument('--swa-start', type=int, default=Args.swa_start)
    p.add_argument('--outdir', type=str, default=Args.outdir)
    args = p.parse_args()

    seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    # Save run config
    with open(os.path.join(outdir, 'hparams.json'), 'w') as f:
        json.dump({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'label_smoothing': args.label_smoothing,
            'seed': args.seed,
            'swa': bool(args.swa),
            'swa_start': args.swa_start,
            'device': device
        }, f, indent=2)

    train_loader, test_loader = get_data(args.data, args.batch_size)

    model = build_resnet18_mnist().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.wd, nesterov=True)

    # Cosine schedule over epochs (no epoch arg on step)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-3)

    # AMP GradScaler via torch.amp
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device == 'cuda'))

    use_swa = bool(args.swa)
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start = int(args.swa_start)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    best_acc, best_epoch = 0.0, -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion,
            use_random_erasing=True,
        )

        # ---- LR scheduler step AFTER optimizer updates (end of epoch) ----
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        val_loss, val_acc = evaluate(model, test_loader, device)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': (swa_model.module.state_dict() if use_swa and epoch >= swa_start else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': best_acc,
            }, os.path.join(outdir, 'best_mnist.pt'))

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"train {train_loss:.4f}/{train_acc*100:.2f}% | "
              f"val {val_loss:.4f}/{val_acc*100:.2f}% | "
              f"best {best_acc*100:.2f}% @ {best_epoch+1}")

    if use_swa:
        # Update BN stats and evaluate SWA weights
        update_bn(train_loader, swa_model, device=device)
        swa_loss, swa_acc = evaluate(swa_model, test_loader, device)
        torch.save({'epoch': args.epochs,
                    'model_state_dict': swa_model.module.state_dict(),
                    'acc': swa_acc}, os.path.join(outdir, 'best_mnist_swa.pt'))
        print(f"SWA eval | val {swa_loss:.4f}/{swa_acc*100:.2f}%")


if __name__ == '__main__':
    main()
