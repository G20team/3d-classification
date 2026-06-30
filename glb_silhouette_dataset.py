#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glb_silhouette_dataset.py
=========================
ポケモンの .glb (glTF binary) 3Dモデル群から、「だーれだ？」クイズ用の
シルエット（輪郭）画像データセットを生成するツール。

特徴:
- KHR_draco_mesh_compression で圧縮された .glb を自動デコード
- 非圧縮 .glb（通常のfloat POSITION + indices）にもフォールバック対応
- glTFノード階層のワールド変換を適用してパーツを正しく合成
- GPU不要（投影 + matplotlibラスタライズによるソフトウェアレンダリング）
- 1モデルから任意枚数のシルエットを多視点で生成（データ拡張込み）
- 出力は PyTorch ImageFolder 形式（dataset/<label>/<label>_xxx.png）+ manifest.csv

視点生成について:
- 既定(--mode turntable)は Y軸まわりの水平回転のみによる「正規の回転」。
  仰角は固定(--elevation, 既定0度)、方位角を 360度/views で等間隔に回す。
  MVCNN等で「一定角度刻みの回転ビュー」を前提にする用途に対応するための既定値。
- quiz / sphere モードは過去互換のために残しているが、quizはランダムジッタを含み
  sphereは仰角も変化するため、「単一軸での回転」にはならない点に注意。

ファイル名について:
- 出力ファイル名は `{label}_{view_index:03d}.png` に統一
  (例: label="25" なら 25_000.png, 25_001.png, ...)
  以前のように元の.glbファイル名やカテゴリ名を含む長いキーは使わない。
  同一labelに複数の.glbが集約される場合（--label-mode species等）は、
  view_indexがモデル間で連番になり重複しないようにしている。

使い方の例:
    # フォルダ内の全 .glb から、各モデル36枚のシルエットをturntable回転で生成(既定)
    python glb_silhouette_dataset.py --input ./models --output ./dataset --views 36

    # 24枚、解像度224、水平回転のみ
    python glb_silhouette_dataset.py -i ./models -o ./dataset --views 24 --res 224

    # 過去互換: quizモード(ランダムジッタあり、単一軸回転ではない)
    python glb_silhouette_dataset.py -i ./models -o ./dataset --mode quiz --views 24

    # 過去互換: sphereモード(全方位、単一軸回転ではない)
    python glb_silhouette_dataset.py -i ./models -o ./dataset --mode sphere --views 64
"""

import os
import sys
import csv
import json
import struct
import argparse
import traceback
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Draco デコーダ（任意。無ければ非圧縮モデルのみ対応）
try:
    import DracoPy
    _HAS_DRACO = True
except Exception:
    _HAS_DRACO = False


# ----------------------------------------------------------------------------
# GLB 解析
# ----------------------------------------------------------------------------
def _parse_glb(path):
    """GLB バイナリを (json_dict, bin_blob) に分解する。"""
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError(f"{path}: glTFバイナリではありません")
    off = 12
    j = None
    bin_blob = b""
    while off < length:
        clen, ctype = struct.unpack("<II", data[off:off + 8])
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:      # 'JSON'
            j = json.loads(chunk)
        elif ctype == 0x004E4942:    # 'BIN\0'
            bin_blob = chunk
        off += 8 + clen
    if j is None:
        raise ValueError(f"{path}: JSONチャンクがありません")
    return j, bin_blob


_COMPONENT_DTYPE = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_TYPE_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
               "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _read_accessor(j, bin_blob, idx):
    """非圧縮アクセサを numpy 配列として読む。"""
    acc = j["accessors"][idx]
    bv = j["bufferViews"][acc["bufferView"]]
    dtype = _COMPONENT_DTYPE[acc["componentType"]]
    ncomp = _TYPE_NCOMP[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    buf = np.frombuffer(bin_blob, dtype=dtype,
                        count=count * ncomp, offset=start)
    return buf.reshape(count, ncomp) if ncomp > 1 else buf


def _bufferview_bytes(j, bin_blob, bv_idx):
    bv = j["bufferViews"][bv_idx]
    s = bv.get("byteOffset", 0)
    return bin_blob[s:s + bv["byteLength"]]


# ----------------------------------------------------------------------------
# ノード階層 → ワールド変換
# ----------------------------------------------------------------------------
def _node_local_matrix(node):
    if "matrix" in node:
        # glTF は列優先 → 転置して行優先(numpy)へ
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        S = np.eye(4); S[:3, :3] = np.diag(node["scale"]); M = M @ S
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        T = np.eye(4); T[:3, :3] = R; M = M @ T
    if "translation" in node:
        T = np.eye(4); T[:3, 3] = node["translation"]; M = T @ M
    return M


def _mesh_world_transforms(j):
    """各 mesh index に対するワールド変換行列を返す（最初に出現するノードを使用）。"""
    nodes = j.get("nodes", [])
    world = {}

    def visit(idx, parent):
        M = parent @ _node_local_matrix(nodes[idx])
        world[idx] = M
        for c in nodes[idx].get("children", []):
            visit(c, M)

    scene = j.get("scenes", [{}])[j.get("scene", 0)]
    for r in scene.get("nodes", range(len(nodes))):
        visit(r, np.eye(4))

    mesh_world = {}
    for idx, n in enumerate(nodes):
        if "mesh" in n:
            mesh_world.setdefault(n["mesh"], world[idx])
    return mesh_world


# ----------------------------------------------------------------------------
# メッシュ統合（Draco / 非圧縮 両対応）
# ----------------------------------------------------------------------------
def load_mesh(path):
    """
    .glb を読み込み、全パーツを合成した (V, F) を返す。
    V: (N,3) float, F: (M,3) int
    """
    j, bin_blob = _parse_glb(path)
    mesh_world = _mesh_world_transforms(j)

    allV, allF, voff = [], [], 0
    for mi, mesh in enumerate(j.get("meshes", [])):
        M = mesh_world.get(mi, np.eye(4))
        for prim in mesh["primitives"]:
            draco = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
            if draco is not None:
                if not _HAS_DRACO:
                    raise RuntimeError(
                        "Draco圧縮モデルですが DracoPy がありません。"
                        "`pip install DracoPy` を実行してください。")
                comp = _bufferview_bytes(j, bin_blob, draco["bufferView"])
                dm = DracoPy.decode(comp)
                v = np.asarray(dm.points, dtype=np.float64).reshape(-1, 3)
                f = np.asarray(dm.faces, dtype=np.int64).reshape(-1, 3)
            else:
                pos_idx = prim["attributes"]["POSITION"]
                v = _read_accessor(j, bin_blob, pos_idx).astype(np.float64)
                if "indices" in prim:
                    f = _read_accessor(j, bin_blob, prim["indices"]).astype(np.int64).reshape(-1, 3)
                else:
                    f = np.arange(len(v), dtype=np.int64).reshape(-1, 3)

            # ワールド変換を適用
            vh = np.c_[v, np.ones(len(v))] @ M.T
            v = vh[:, :3]
            allV.append(v)
            allF.append(f + voff)
            voff += len(v)

    if not allV:
        raise ValueError(f"{path}: メッシュが見つかりません")
    V = np.vstack(allV)
    F = np.vstack(allF)
    return V, F


# ----------------------------------------------------------------------------
# 正規化 & 視点 & レンダリング
# ----------------------------------------------------------------------------
def normalize(V, up="y"):
    """中心を原点に、最大半径1に正規化。up軸をYに揃える。"""
    V = V.copy()
    if up == "z":          # Z-up → Y-up へ変換
        V = V[:, [0, 2, 1]]
        V[:, 2] *= -1
    c = (V.max(0) + V.min(0)) / 2.0
    V = V - c
    r = np.abs(V).max()
    if r > 0:
        V = V / r
    return V


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def viewpoints(mode, n, base_az=0.0, elevation=0.0, rng=None):
    """
    (azimuth_deg, elevation_deg) のリストを返す。

    mode:
      turntable (既定) : Y軸まわりの水平回転のみで一周。
                         仰角は elevation で固定(既定0度)。
                         360度をn等分した「正規の回転」になる。
                         例: n=24 -> 0,15,30,...,345度
      quiz   (過去互換): 横〜斜め45度を中心に、ランダムジッタ＆仰角変動あり。
                         単一軸の規則的回転にはならない。
      sphere (過去互換): 視点球をほぼ均等にサンプル。仰角も変化するため
                         単一軸の規則的回転にはならない。

    base_az: モデル正面の基準方位オフセット
    elevation: turntableモードでの固定仰角(度)
    """
    rng = rng or np.random.default_rng(0)
    pts = []
    if mode == "turntable":
        # Y軸まわりの水平回転のみ。等間隔にn分割して一周する「正規の回転」。
        for k in range(n):
            az = base_az + 360.0 * k / n
            pts.append((az, float(elevation)))
    elif mode == "sphere":
        # フィボナッチ球（過去互換。単一軸回転ではない）
        ga = np.pi * (3 - np.sqrt(5))
        for k in range(n):
            y = 1 - 2 * (k + 0.5) / n          # -1..1
            el = np.degrees(np.arcsin(y))
            az = np.degrees((k * ga) % (2 * np.pi))
            pts.append((base_az + az, float(el)))
    elif mode == "quiz":
        # 過去互換。横向き(±90)と斜め(±45,±135)を厚めに、ランダムジッタ付与。
        anchors = [90, -90, 45, -45, 135, -135, 60, -60]
        for k in range(n):
            base = anchors[k % len(anchors)]
            az = base_az + base + float(rng.uniform(-12, 12))
            el = float(rng.uniform(0, 18))
            pts.append((az, el))
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return pts


def render_silhouette(V, F, az, el, res=256, supersample=2,
                      invert=False, pad=1.12, line_w=0.4, return_fill=False):
    """
    1視点のシルエットPNGを numpy画像(uint8, HxW)で返す。
    invert=False: 白背景に黒シルエット / True: 黒背景に白

    各ビューを「投影後の2Dバウンディングボックス」に合わせて自動フレーミングする。
    これにより、3D空間での外れ頂点・付属物・台座・スケール差に影響されず、
    どの角度でも被写体が枠いっぱいに一定の余白で収まる。
    """
    R = _rot_x(np.radians(el)) @ _rot_y(np.radians(az))
    P = V @ R.T
    xy = P[:, [0, 1]]
    tris = xy[F]

    # --- 2D 自動フレーミング（アスペクト維持・正方形枠） ---
    mn = xy.min(0); mx = xy.max(0)
    cx, cy = (mn + mx) / 2.0
    half = max(mx[0] - mn[0], mx[1] - mn[1]) / 2.0
    half = (half if half > 0 else 1.0) * pad

    fg, bg = ("white", "black") if invert else ("black", "white")
    s = res * supersample
    fig = plt.figure(figsize=(s / 100, s / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.axis("off")
    pc = PolyCollection(tris, facecolors=fg, edgecolors=fg, linewidths=line_w)
    ax.add_collection(pc)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4)
    plt.close(fig)

    from PIL import Image
    img = Image.fromarray(buf[:, :, :3]).convert("L")
    if supersample != 1:
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.asarray(img)
    if return_fill:
        mask = (arr < 128) if not invert else (arr > 128)
        return arr, float(mask.mean())
    return arr


# ----------------------------------------------------------------------------
# ラベル付け
# ----------------------------------------------------------------------------
import re

# 同一シルエット（色のみ違い）のカテゴリ → 非shinyに正規化するための対応
_SHINY_ALIAS = {
    "shiny": "regular", "megashiny": "mega",
    "fusionshiny": "fusion", "multishinyform": "multiform",
}


def _sanitize(s):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", s)


def make_label(path, mode, root=None):
    """
    ラベル決定。
      stem    : ファイル名そのまま（例 "128", "Black_Kyurem"）
      species : 先頭の図鑑番号（"668-M"->"668"）。番号が無ければstem。
                regular/alolan/mega…で同番号は同ラベルに集約（=種名レベル）。
      form    : <カテゴリ>_<stem>（例 "alolan_25"）。形態ごとに別クラス。
                shiny系は同形状の非shinyへ正規化して重複クラスを防ぐ。
    """
    p = Path(path)
    stem = p.stem
    if mode == "stem":
        return _sanitize(stem)
    if mode == "species":
        m = re.match(r"^(\d+)", stem)
        return m.group(1) if m else _sanitize(stem)
    # form
    cat = p.parent.name.lower()
    cat = _SHINY_ALIAS.get(cat, cat)
    return _sanitize(f"{cat}_{stem}")


def load_exclude(path):
    """除外リスト（1行1パターン）を読む。空・# はコメント。"""
    if not path:
        return set()
    pats = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            pats.add(s.lower())
    return pats


def is_excluded(path, root, pats):
    if not pats:
        return False
    p = Path(path)
    cand = {p.stem.lower(), p.name.lower()}
    try:
        rel = p.relative_to(root)
        cand.add(str(rel).lower())
        cand.add(str(rel.with_suffix("")).lower())          # regular/1000
        cand.add(str(rel.with_suffix("")).lower().replace(os.sep, "/"))
    except Exception:
        pass
    return bool(cand & pats)


def make_contact_sheet(out_dir, thumb=96, cols=24):
    """各クラス1枚を並べたQC用シートを out/_contact_sheet.png に出力。"""
    from PIL import Image, ImageDraw
    out = Path(out_dir)
    labels = sorted(d.name for d in out.iterdir() if d.is_dir())
    if not labels:
        return
    cells = []
    for lab in labels:
        pngs = sorted((out / lab).glob("*.png"))
        if not pngs:
            continue
        im = Image.open(pngs[0]).convert("L").resize((thumb, thumb), Image.LANCZOS)
        d = ImageDraw.Draw(im); d.text((2, 2), lab[:14], fill=0)
        cells.append(im)
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("L", (cols * thumb, rows * thumb), 255)
    for i, im in enumerate(cells):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    fn = out / "_contact_sheet.png"
    sheet.save(fn)
    print(f"[QC] コンタクトシート: {fn}  ({len(cells)}クラス)")


# ----------------------------------------------------------------------------
# メイン処理
# ----------------------------------------------------------------------------
def build_dataset(args):
    in_path = Path(args.input)
    files = []
    if in_path.is_dir():
        files = sorted(in_path.rglob("*.glb"))
    elif in_path.suffix.lower() == ".glb":
        files = [in_path]
    if not files:
        print(f"[!] .glb が見つかりません: {in_path}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    rng = np.random.default_rng(args.seed)

    skip = set(d.strip().lower() for d in (args.skip_dirs or "").split(",") if d.strip())
    if skip:
        files = [f for f in files if f.parent.name.lower() not in skip]
        print(f"[i] スキップ対象カテゴリ {sorted(skip)} を除外 -> 残り {len(files)} 体")

    root = in_path if in_path.is_dir() else in_path.parent
    excl = load_exclude(args.exclude)
    if excl:
        before = len(files)
        files = [f for f in files if not is_excluded(f, root, excl)]
        print(f"[i] 除外リストで {before - len(files)} 体を除外 -> 残り {len(files)} 体")
    n_ok = n_fail = n_imgs = 0

    # label毎に「次に振るview_index」を管理する。
    # 同一labelに複数の.glbが集約される場合(例: --label-mode species)でも
    # ファイル名が重複しないよう、label内で連番を継続させる。
    next_view_index = {}

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filepath", "label", "view_index", "source_glb", "azimuth", "elevation"])

        for fi, f in enumerate(files, 1):
            label = make_label(f, args.label_mode, root)
            try:
                V, F = load_mesh(f)
                V = normalize(V, up=args.up)
            except Exception as e:
                n_fail += 1
                print(f"[{fi}/{len(files)}] FAIL {f.name}: {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
                continue

            label_dir = out / label
            label_dir.mkdir(exist_ok=True)
            vps = viewpoints(args.mode, args.views, base_az=args.base_az,
                              elevation=args.elevation, rng=rng)

            from PIL import Image
            kept = 0
            view_idx = next_view_index.get(label, 0)
            for vi, (az, el) in enumerate(vps):
                img, fill = render_silhouette(
                    V, F, az, el, res=args.res, supersample=args.supersample,
                    invert=args.invert, pad=args.pad, return_fill=True)
                if args.min_fill > 0 and fill < args.min_fill:
                    continue  # 退化ビュー（ほぼ線）をスキップ

                # ファイル名: {label}_{view_index:03d}.png
                fn = label_dir / f"{label}_{view_idx:03d}.png"
                Image.fromarray(img).save(fn)
                writer.writerow([str(fn.relative_to(out)), label, view_idx,
                                 str(f.relative_to(root)), round(az, 2), round(el, 2)])
                n_imgs += 1
                kept += 1
                view_idx += 1

            next_view_index[label] = view_idx
            n_ok += 1
            print(f"[{fi}/{len(files)}] OK  {f.relative_to(root)} -> {label} "
                  f"({kept}枚, mode={args.mode})")

    print(f"\n完了: モデル {n_ok} 体 / 失敗 {n_fail} / 画像 {n_imgs} 枚")
    print(f"出力先: {out}")
    print(f"manifest: {manifest_path}")
    if args.contact_sheet:
        make_contact_sheet(out)
    return 0


def build_argparser():
    p = argparse.ArgumentParser(
        description="ポケモン .glb からシルエット画像データセットを生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", required=True,
                   help=".glb ファイル、またはそれらを含むフォルダ")
    p.add_argument("-o", "--output", required=True, help="出力フォルダ")
    p.add_argument("--views", type=int, default=36,
                   help="1モデルあたりの生成枚数（視点数）")
    p.add_argument("--mode", choices=["turntable", "quiz", "sphere"],
                   default="turntable",
                   help="視点サンプリング方式。既定turntableはY軸まわりの水平回転のみ"
                        "(単一軸での規則的な回転)。quiz/sphereは過去互換用で、"
                        "ランダムジッタや仰角変化を含み単一軸回転ではない。")
    p.add_argument("--elevation", type=float, default=0.0,
                   help="turntableモードでの固定仰角(度)。0なら完全に水平回転のみ。")
    p.add_argument("--res", type=int, default=256, help="出力解像度(正方形)")
    p.add_argument("--supersample", type=int, default=2,
                   help="アンチエイリアス用の内部拡大率")
    p.add_argument("--invert", action="store_true",
                   help="黒背景に白シルエットで出力（既定は白背景に黒）")
    p.add_argument("--pad", type=float, default=1.12,
                   help="フレーム余白率（1.0=ぴったり、1.12=12%%余白）")
    p.add_argument("--min-fill", type=float, default=0.0,
                   help="シルエット面積率がこの値未満の退化ビューを除外(例:0.02)")
    p.add_argument("--up", choices=["y", "z"], default="y",
                   help="モデルの上方向軸（glTFは通常 y）")
    p.add_argument("--base-az", type=float, default=0.0,
                   help="正面基準の方位オフセット(度)")
    p.add_argument("--label-mode", choices=["stem", "species", "form"],
                   default="species",
                   help="ラベル: stem=ファイル名 / species=図鑑番号(形態集約) / form=カテゴリ別")
    p.add_argument("--skip-dirs", default="",
                   help="除外するカテゴリ(カンマ区切り)。例: shiny,megaShiny,fusionShiny,multiShinyForm")
    p.add_argument("--exclude", default="",
                   help="除外モデルを列挙したテキスト(1行1名/相対パス)。台座付き等の不良モデル用")
    p.add_argument("--contact-sheet", action="store_true",
                   help="各クラス1枚を並べたQCシート(_contact_sheet.png)を出力")
    p.add_argument("--seed", type=int, default=0, help="乱数シード")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(build_dataset(args))
