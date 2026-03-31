"""
Tiling Preprocessor for High-Resolution Ecological UAV Images
==============================================================
Splits large images into patches, maps COCO annotations to each patch,
filters truncated boxes and empty patches, and provides visualization.

Usage:
    python tile_preprocess.py --img_dir /path/to/train \
                              --ann_file /path/to/annotations.json \
                              --out_dir /path/to/output \
                              --patch_size 1280 \
                              --overlap 0.2 \
                              --min_bbox_ratio 0.3 \
                              --empty_keep_ratio 0.25

    # Visualize a single image's tiled predictions
    python tile_preprocess.py --visualize /path/to/output/images/some_patch.jpg \
                              --ann_file /path/to/output/annotations.json
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict


def compute_patch_coords(img_w, img_h, patch_size, overlap):
    """Compute top-left coordinates for all patches with given overlap."""
    stride = int(patch_size * (1 - overlap))
    coords = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            # Shift back if patch extends beyond image edge
            x1 = max(0, x2 - patch_size)
            y1 = max(0, y2 - patch_size)
            coords.append((x1, y1, x2, y2))
    # Deduplicate (edge patches may repeat)
    return list(set(coords))


def clip_bbox_to_patch(bbox, patch):
    """Clip a COCO bbox [x,y,w,h] to patch region. Returns clipped bbox and overlap ratio."""
    bx, by, bw, bh = bbox
    px1, py1, px2, py2 = patch

    # Original box corners
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(bx, px1)
    iy1 = max(by, py1)
    ix2 = min(bx2, px2)
    iy2 = min(by2, py2)

    if ix1 >= ix2 or iy1 >= iy2:
        return None, 0.0

    orig_area = bw * bh
    if orig_area <= 0:
        return None, 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    ratio = inter_area / orig_area

    # Convert to patch-local coordinates
    local_bbox = [ix1 - px1, iy1 - py1, ix2 - ix1, iy2 - iy1]
    return local_bbox, ratio


def tile_dataset(img_dir, ann_file, out_dir, patch_size=1280, overlap=0.2,
                 min_bbox_ratio=0.3, empty_keep_ratio=0.25):
    """Main tiling pipeline."""
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # Build lookup: image_id -> annotations
    img_id_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    # Build lookup: image_id -> image info
    img_id_to_info = {img['id']: img for img in coco['images']}

    new_images = []
    new_annotations = []
    new_img_id = 0
    new_ann_id = 0
    empty_patches = []

    stats = defaultdict(int)

    for img_info in coco['images']:
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        anns = img_id_to_anns[img_info['id']]
        patches = compute_patch_coords(img_w, img_h, patch_size, overlap)

        for px1, py1, px2, py2 in patches:
            patch_anns = []

            for ann in anns:
                clipped, ratio = clip_bbox_to_patch(ann['bbox'], (px1, py1, px2, py2))
                if clipped is None or ratio < min_bbox_ratio:
                    continue

                new_ann_id += 1
                patch_anns.append({
                    'id': new_ann_id,
                    'image_id': new_img_id,
                    'category_id': ann['category_id'],
                    'bbox': [round(c, 2) for c in clipped],
                    'area': round(clipped[2] * clipped[3], 2),
                    'iscrowd': ann.get('iscrowd', 0)
                })
                stats[ann['category_id']] += 1

            # Handle empty patches
            if len(patch_anns) == 0:
                empty_patches.append((img_path, px1, py1, px2, py2, new_img_id))
                new_img_id += 1
                continue

            # Crop and save
            patch_img = img.crop((px1, py1, px2, py2))
            patch_name = f"{img_path.stem}_{px1}_{py1}_{px2}_{py2}.jpg"
            patch_img.save(out_img_dir / patch_name, quality=95)

            new_images.append({
                'id': new_img_id,
                'file_name': patch_name,
                'width': px2 - px1,
                'height': py2 - py1
            })
            new_annotations.extend(patch_anns)
            new_img_id += 1

    # Keep a subset of empty patches as negative samples
    n_keep = int(len(empty_patches) * empty_keep_ratio)
    if n_keep > 0:
        rng = np.random.default_rng(42)
        kept = rng.choice(len(empty_patches), size=n_keep, replace=False)
        for idx in kept:
            img_path, px1, py1, px2, py2, eid = empty_patches[idx]
            img = Image.open(img_path)
            patch_img = img.crop((px1, py1, px2, py2))
            patch_name = f"{img_path.stem}_{px1}_{py1}_{px2}_{py2}.jpg"
            patch_img.save(out_img_dir / patch_name, quality=95)
            new_images.append({
                'id': eid,
                'file_name': patch_name,
                'width': px2 - px1,
                'height': py2 - py1
            })

    # Build output COCO JSON
    out_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco['categories']
    }

    out_ann_path = out_dir / "annotations.json"
    with open(out_ann_path, 'w') as f:
        json.dump(out_coco, f)

    # Print stats
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    print(f"\n{'='*50}")
    print(f"Tiling complete")
    print(f"{'='*50}")
    print(f"Original images:  {len(coco['images'])}")
    print(f"Output patches:   {len(new_images)} ({len(new_images) - n_keep} with objects, {n_keep} empty)")
    print(f"Output annotations: {len(new_annotations)}")
    print(f"\nPer-category instance counts:")
    print(f"{'Category':<30} {'Count':>8}")
    print(f"{'-'*38}")
    for cat_id, count in sorted(stats.items(), key=lambda x: -x[1]):
        name = cat_names.get(cat_id, f"id_{cat_id}")
        print(f"{name:<30} {count:>8}")
    print(f"{'='*50}")

    return out_ann_path


def visualize(img_path, ann_file):
    """Draw bounding boxes and category labels on a single image."""
    img_path = Path(img_path)
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    cat_names = {c['id']: c['name'] for c in coco['categories']}

    # Find image entry
    img_entry = None
    for im in coco['images']:
        if im['file_name'] == img_path.name:
            img_entry = im
            break
    if img_entry is None:
        print(f"[ERROR] {img_path.name} not found in annotations")
        return

    # Gather annotations
    anns = [a for a in coco['annotations'] if a['image_id'] == img_entry['id']]

    # Distinct colors per category
    rng = np.random.default_rng(0)
    colors = {}
    for cid in cat_names:
        colors[cid] = tuple(rng.integers(60, 230, size=3).tolist())

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for ann in anns:
        x, y, w, h = ann['bbox']
        cid = ann['category_id']
        color = colors.get(cid, (255, 0, 0))
        label = cat_names.get(cid, str(cid))

        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x + 2, y - 18), label, fill=color, font=font)

    out_path = img_path.parent / f"vis_{img_path.name}"
    img.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Objects: {len(anns)}")
    for ann in anns:
        cid = ann['category_id']
        bx, by, bw, bh = [round(v, 1) for v in ann['bbox']]
        print(f"  [{cat_names.get(cid, cid)}] bbox=({bx}, {by}, {bw}, {bh})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile high-res images with COCO annotations")
    parser.add_argument("--img_dir", type=str, help="Input image directory")
    parser.add_argument("--ann_file", type=str, help="COCO annotation JSON path")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--patch_size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min_bbox_ratio", type=float, default=0.3)
    parser.add_argument("--empty_keep_ratio", type=float, default=0.25)
    parser.add_argument("--visualize", type=str, default=None,
                        help="Path to a single patch image to visualize")

    args = parser.parse_args()

    if args.visualize:
        visualize(args.visualize, args.ann_file)
    else:
        if not all([args.img_dir, args.ann_file, args.out_dir]):
            parser.error("--img_dir, --ann_file, and --out_dir are required for tiling")
        tile_dataset(
            args.img_dir, args.ann_file, args.out_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_bbox_ratio=args.min_bbox_ratio,
            empty_keep_ratio=args.empty_keep_ratio
        )
