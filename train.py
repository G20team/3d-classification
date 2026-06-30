"""
MVCNN学習・評価スクリプト。

使い方の想定:
    python train.py --root dataset --backbone resnet18 --epochs 30

主な処理:
    1. dataset/ から角度ホールドアウト分割でtrain/testを作る
    2. MVCNNを学習(クラス分類, CrossEntropyLoss)
    3. テスト(=3Dモデルの未学習角度)での精度・混同行列を出す
    4. (オプション) イラスト画像があれば predict_illustrations() で単一画像推論を試せる
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import MultiViewDataset, SingleImageDataset, make_holdout_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="dataset", help="dataset root directory")
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "simple_cnn"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_views", type=int, default=24)
    p.add_argument("--holdout_stride", type=int, default=4, help="N枚に1枚をテスト用ホールドアウト")
    p.add_argument("--batch_size", type=int, default=4, help="個体数単位のバッチサイズ")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_pretrained", action="store_true", help="resnet18の事前学習重みを使わない")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="runs")
    return p.parse_args()


def evaluate(model, loader, device, label_map=None):
    """
    精度と混同行列を計算する。
    label_map: {individual_id_str: class_index} があれば、混同行列の見出しに使える。
    """
    model.eval()
    correct = 0
    total = 0
    num_classes = model.classifier[-1].out_features
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for views, labels in loader:
            views = views.to(device)
            labels = labels.to(device)
            logits = model(views)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, pr in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), pr.long()] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, confusion


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = make_holdout_indices(num_views=args.num_views, stride=args.holdout_stride)
    print(f"Holdout view indices (test): {holdout}")

    train_ds = MultiViewDataset(
        args.root, split="train", holdout_indices=holdout, image_size=args.image_size
    )
    test_ds = MultiViewDataset(
        args.root, split="test", holdout_indices=holdout, image_size=args.image_size
    )
    print(f"Individuals: {len(train_ds.individual_ids)}")
    print(f"Train samples (individuals): {len(train_ds)} / Test samples (individuals): {len(test_ds)}")

    # 注意: 個体ごとにビュー数が異なる場合、デフォルトのcollateだとstackできずエラーになる。
    # 今回は「各個体24ビュー固定」を前提にしているため batch_size>1 でもそのまま動く。
    # ビュー数が個体によって変わる場合は collate_fn を別途用意する必要がある。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    label_map = train_ds.get_label_map()
    num_classes = len(label_map)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        input_channels=1,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for views, labels in train_loader:
            views = views.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = model(views)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        avg_loss = running_loss / len(train_ds)
        test_acc, _ = evaluate(model, test_loader, args.device, label_map)
        print(f"[epoch {epoch:03d}] train_loss={avg_loss:.4f}  test_acc(holdout views)={test_acc:.4f}")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"Best holdout accuracy: {best_acc:.4f}")

    # 最終的な混同行列も保存しておく(誤分類の傾向を見るため)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=args.device))
    final_acc, confusion = evaluate(model, test_loader, args.device, label_map)
    print(f"Final loaded-best accuracy on holdout: {final_acc:.4f}")
    torch.save(confusion, out_dir / "confusion_matrix.pt")
    print("Confusion matrix saved to", out_dir / "confusion_matrix.pt")

    return model, label_map


def predict_illustrations(model, image_paths, label_map, device, image_size=224):
    """
    イラスト実験用の推論関数。
    image_paths: イラストのファイルパスのリスト
    label_map: train_ds.get_label_map() で得た {individual_id_str: class_index}

    各画像について、予測クラス(個体ID)と確信度(softmax確率)を返す。
    正解ラベルが分かっている場合は、別途accuracyを自分で計算してください
    (このタスクはあくまで定性的なケーススタディとして位置づける想定)。
    """
    inv_label_map = {v: k for k, v in label_map.items()}
    ds = SingleImageDataset(image_paths, labels=None, image_size=image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    results = []
    with torch.no_grad():
        for i, (views, _) in enumerate(loader):
            views = views.to(device)
            logits = model(views)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            top3_vals, top3_idx = probs[0].topk(min(3, probs.shape[1]))
            top3 = [(inv_label_map[idx.item()], val.item()) for idx, val in zip(top3_idx, top3_vals)]
            results.append({
                "image": image_paths[i],
                "pred_individual": inv_label_map[pred_idx],
                "confidence": confidence,
                "top3": top3,
            })
    return results


if __name__ == "__main__":
    args = parse_args()
    model, label_map = train(args)

    # イラスト画像が illustrations/ にあれば、ここで試す例(なければスキップされる)
    illust_dir = Path("illustrations")
    if illust_dir.exists():
        illust_paths = sorted(str(p) for p in illust_dir.glob("*.png"))
        if illust_paths:
            print("\n--- Illustration experiment (qualitative) ---")
            results = predict_illustrations(
                model, illust_paths, label_map, args.device, args.image_size
            )
            for r in results:
                print(
                    f"{r['image']}: pred={r['pred_individual']} "
                    f"(conf={r['confidence']:.3f}) top3={r['top3']}"
                )
