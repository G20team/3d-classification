#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier.py  (おまけ / たたき台)
=======================================
glb_silhouette_dataset.py が生成した ImageFolder 形式のデータセットを使って、
シルエットからポケモンを当てる分類器を転移学習する最小スクリプト。

依存:
    pip install torch torchvision

使い方:
    python train_classifier.py --data ./dataset --epochs 15 --out model.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def build_transforms(res=224, train=True):
    # シルエットは1ch相当 → 3chに複製して既存の事前学習重みに合わせる
    aug = []
    if train:
        aug = [
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05),
                                    scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),  # 左右どちら向きでも当てたい場合
        ]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((res, res)),
        *aug,
        transforms.ToTensor(),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder ルート")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    full = datasets.ImageFolder(args.data, transform=build_transforms(args.res, True))
    classes = full.classes
    n_val = int(len(full) * args.val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(0))
    # 検証側は拡張を切る
    val_set.dataset = datasets.ImageFolder(
        args.data, transform=build_transforms(args.res, False))

    train_dl = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        tot = correct = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
        sched.step()

        # validation
        model.eval()
        v_tot = v_top1 = v_top5 = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_top1 += (out.argmax(1) == y).sum().item()
                top5 = out.topk(min(5, len(classes)), 1).indices
                v_top5 += (top5 == y[:, None]).any(1).sum().item()
                v_tot += len(y)
        print(f"epoch {ep+1:2d}/{args.epochs}  "
              f"train_acc={correct/tot:.3f}  "
              f"val_top1={v_top1/max(v_tot,1):.3f}  "
              f"val_top5={v_top5/max(v_tot,1):.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
    print("保存:", args.out)


if __name__ == "__main__":
    main()
