# ポケモン「だーれだ？」シルエット・データセット生成ツール

`.glb`（glTF binary）のポケモン3Dモデルから、シルエット（輪郭）画像の
データセットを自動生成します。出力は PyTorch の `ImageFolder` 形式なので、
そのまま画像分類の学習に使えます。

検証済み：`128-1.glb`（タウロス）→ 横・斜めなど多視点で綺麗な黒シルエットを生成。

---

## 1. セットアップ

```bash
pip install numpy pillow matplotlib DracoPy
# 学習も行う場合（おまけスクリプト用）
pip install torch torchvision
```

- `DracoPy` は Draco 圧縮された `.glb`（今回のモデルはこれ）を解凍するために必須。
- レンダリングは GPU 不要（CPU のソフトウェア描画）。

---

## 2. データセット生成

```bash
# フォルダ内の全 .glb から、各モデル36枚を生成
python glb_silhouette_dataset.py --input ./models --output ./dataset --views 36

# 「だーれだ」本番寄り（横〜斜め45度を厚め）、解像度224
python glb_silhouette_dataset.py -i ./models -o ./dataset --mode quiz --views 24 --res 224

# 全方位サンプリング（最も頑健・どの角度でも当てたい場合）
python glb_silhouette_dataset.py -i ./models -o ./dataset --mode sphere --views 64
```

### 出力構成（ImageFolder 形式）
```
dataset/
├── 128-1/
│   ├── 128-1_000.png
│   ├── 128-1_001.png
│   └── ...
├── 0006-1/
│   └── ...
└── manifest.csv      # 各画像の label / 元glb / 方位・仰角 を記録
```

### 主なオプション
| オプション | 説明 | 既定 |
|---|---|---|
| `--mode` | `quiz`(横〜斜め重視) / `turntable`(一周) / `sphere`(全方位) | `quiz` |
| `--views` | 1モデルあたりの枚数 | 36 |
| `--res` | 出力解像度(正方形) | 256 |
| `--invert` | 黒背景に白シルエットで出力 | 白背景に黒 |
| `--label-mode` | `stem`(=ファイル名) / `species`(=先頭番号) | `stem` |
| `--up` | モデルの上方向軸 `y`/`z` | `y` |
| `--base-az` | 正面基準の方位オフセット(度) | 0 |
| `--supersample` | アンチエイリアス用の内部拡大率 | 2 |

> ラベルはファイル名から決まります。`128-1.glb` → ラベル `128-1`。
> 形違いをまとめて「128」にしたい場合は `--label-mode species`。

---

## 3. 学習（おまけ）

```bash
python train_classifier.py --data ./dataset --epochs 15 --out model.pt
```

- ResNet18 の転移学習。`val_top1` / `val_top5` を毎エポック表示。
- 左右どちら向きのシルエットでも当てたいので水平反転を有効化済み
  （非対称ポケモンを厳密に区別したい場合は `train_classifier.py` 内の
  `RandomHorizontalFlip` を外してください）。

---

## 4. 精度を上げるコツ

- **本番の角度に寄せる**：アニメの「だーれだ」は公式アート寄りの横〜斜め45度が
  多いので `--mode quiz` が基本。実クイズ画像が手元にあるなら、その角度に
  合わせて `--base-az` や anchors を調整。
- **輪郭が似た種の取りこぼし対策**：四足獣・鳥など似た輪郭が多いので、
  評価は Top-1 だけでなく Top-5 も見る。混同しやすいペアは視点を増やす。
- **データ拡張**：生成時に多視点・微小回転を入れているので、学習側の拡張は
  軽め（平行移動・スケール・少回転）で十分。
- **前処理の一致**：本番のクイズ画像も「白背景に黒シルエット・正方形・余白そろえ」
  に揃えると、ドメインギャップが減って精度が安定します。

---

## 仕組み（概要）

1. GLB を JSON+BIN に分解し、`KHR_draco_mesh_compression` を `DracoPy` で解凍
   （非圧縮モデルはアクセサを直接読む）。
2. glTF ノード階層のワールド変換を適用し、body / eye / fur 等のパーツを合成。
3. 中心化・正規化（上方向を Y に統一）。
4. 視点ごとに回転 → 正射影 → 全三角形を黒で塗りつぶし（=輪郭の和集合）→
   2倍解像度で描いて縮小しアンチエイリアス。
