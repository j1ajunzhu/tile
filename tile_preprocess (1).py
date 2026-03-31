"""
Tiling Preprocessor for High-Resolution Ecological UAV Images (YOLO Format)
============================================================================
Splits large images into patches, maps YOLO annotations to each patch,
filters truncated boxes and empty patches, and provides visualization.

Expected input:
    img_dir/001.jpg    label_dir/001.txt
    img_dir/002.jpg    label_dir/002.txt

YOLO label format (each line):
    class_id x_center y_center width height   (normalized 0-1)

Usage:
    # Tiling
    python tile_preprocess.py --img_dir /path/to/images \
                              --label_dir /path/to/labels \
                              --out_dir /path/to/output \
                              --patch_size 1280 \
                              --overlap 0.2

    # Visualize a single tiled patch
    python tile_preprocess.py --visualize /path/to/output/images/some_patch.jpg \
                              --label_dir /path/to/output/labels \
                              --class_names "species_a,species_b,species_c"
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict


def yolo_to_abs(bbox, img_w, img_h):
    """YOLO normalized [cx, cy, w, h] -> absolute [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox
    abs_w, abs_h = w * img_w, h * img_h
    x1 = cx * img_w - abs_w / 2
    y1 = cy * img_h - abs_h / 2
    return [x1, y1, x1 + abs_w, y1 + abs_h]


def abs_to_yolo(bbox, patch_w, patch_h):
    """Absolute [x1, y1, x2, y2] -> YOLO normalized [cx, cy, w, h]."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return [(x1 + w / 2) / patch_w, (y1 + h / 2) / patch_h,
            w / patch_w, h / patch_h]


def compute_patches(img_w, img_h, patch_size, overlap):
    """Compute all patch coordinates."""
    stride = int(patch_size * (1 - overlap))
    coords = set()
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            coords.add((max(0, x2 - patch_size), max(0, y2 - patch_size), x2, y2))
    return list(coords)


def clip_box(abs_box, patch, min_ratio):
    """Clip box to patch. Returns patch-local [x1,y1,x2,y2] or None."""
    bx1, by1, bx2, by2 = abs_box
    px1, py1, px2, py2 = patch
    ix1, iy1 = max(bx1, px1), max(by1, py1)
    ix2, iy2 = min(bx2, px2), min(by2, py2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    orig_area = (bx2 - bx1) * (by2 - by1)
    if orig_area <= 0 or (ix2 - ix1) * (iy2 - iy1) / orig_area < min_ratio:
        return None
    return [ix1 - px1, iy1 - py1, ix2 - px1, iy2 - py1]


def read_labels(path):
    """Read YOLO label file -> list of (class_id, [cx, cy, w, h])."""
    labels = []
    if not path.exists():
        return labels
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 5:
            labels.append((int(parts[0]), [float(v) for v in parts[1:5]]))
    return labels


def tile_dataset(img_dir, label_dir, out_dir, patch_size=1280, overlap=0.2,
                 min_bbox_ratio=0.3, empty_keep_ratio=0.25):
    img_dir, label_dir, out_dir = Path(img_dir), Path(label_dir), Path(out_dir)
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')])

    stats = defaultdict(int)
    empty_patches = []
    saved = 0

    for img_path in img_files:
        labels = read_labels(label_dir / (img_path.stem + ".txt"))
        img = Image.open(img_path)
        img_w, img_h = img.size

        abs_labels = [(cid, yolo_to_abs(bb, img_w, img_h)) for cid, bb in labels]
        patches = compute_patches(img_w, img_h, patch_size, overlap)

        for px1, py1, px2, py2 in patches:
            pw, ph = px2 - px1, py2 - py1
            patch_labels = []

            for cid, abox in abs_labels:
                local = clip_box(abox, (px1, py1, px2, py2), min_bbox_ratio)
                if local:
                    patch_labels.append((cid, abs_to_yolo(local, pw, ph)))
                    stats[cid] += 1

            name = f"{img_path.stem}_{px1}_{py1}_{px2}_{py2}"

            if not patch_labels:
                empty_patches.append((img_path, px1, py1, px2, py2, name))
                continue

            img.crop((px1, py1, px2, py2)).save(out_img / f"{name}.jpg", quality=95)
            with open(out_lbl / f"{name}.txt", 'w') as f:
                for cid, yb in patch_labels:
                    f.write(f"{cid} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}\n")
            saved += 1

    # Keep some empty patches as negatives
    n_keep = int(len(empty_patches) * empty_keep_ratio)
    if n_keep > 0:
        rng = np.random.default_rng(42)
        for idx in rng.choice(len(empty_patches), n_keep, replace=False):
            ip, px1, py1, px2, py2, name = empty_patches[idx]
            Image.open(ip).crop((px1, py1, px2, py2)).save(out_img / f"{name}.jpg", quality=95)
            open(out_lbl / f"{name}.txt", 'w').close()  # empty label
            saved += 1

    print(f"\n{'='*50}")
    print(f"Tiling complete")
    print(f"{'='*50}")
    print(f"Original images:  {len(img_files)}")
    print(f"Output patches:   {saved} ({saved - n_keep} with objects, {n_keep} empty)")
    print(f"\nPer-class instance counts:")
    print(f"{'Class':<10} {'Count':>8}")
    print(f"{'-'*18}")
    for cid, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{cid:<10} {cnt:>8}")
    print(f"{'='*50}")


def visualize(img_path, label_dir, class_names=None):
    img_path, label_dir = Path(img_path), Path(label_dir)
    label_path = label_dir / (img_path.stem + ".txt")
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    labels = read_labels(label_path)

    if not labels:
        print(f"No annotations for {img_path.name}")
        return

    names = {}
    if class_names:
        for i, n in enumerate(class_names.split(",")):
            names[i] = n.strip()

    rng = np.random.default_rng(0)
    colors = {cid: tuple(rng.integers(60, 230, size=3).tolist())
              for cid in set(c for c, _ in labels)}

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    print(f"\n{img_path.name}: {len(labels)} objects")
    for cid, bbox in labels:
        x1, y1, x2, y2 = yolo_to_abs(bbox, img_w, img_h)
        label = names.get(cid, str(cid))
        draw.rectangle([x1, y1, x2, y2], outline=colors[cid], width=2)
        draw.text((x1 + 2, y1 - 18), label, fill=colors[cid], font=font)
        print(f"  [{label}] bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    out = img_path.parent / f"vis_{img_path.name}"
    img.save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str)
    p.add_argument("--label_dir", type=str)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--patch_size", type=int, default=1280)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min_bbox_ratio", type=float, default=0.3)
    p.add_argument("--empty_keep_ratio", type=float, default=0.25)
    p.add_argument("--visualize", type=str, default=None)
    p.add_argument("--class_names", type=str, default=None)
    args = p.parse_args()

    if args.visualize:
        ldir = args.label_dir or str(Path(args.visualize).parent.parent / "labels")
        visualize(args.visualize, ldir, args.class_names)
    else:
        if not all([args.img_dir, args.label_dir, args.out_dir]):
            p.error("--img_dir, --label_dir, --out_dir required")
        tile_dataset(args.img_dir, args.label_dir, args.out_dir,
                     args.patch_size, args.overlap, args.min_bbox_ratio, args.empty_keep_ratio)
