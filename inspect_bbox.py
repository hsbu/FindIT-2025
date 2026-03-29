"""
inspect_bbox.py — Browse bbox_cache.json results one image at a time.

Usage:
    python inspect_bbox.py                  # all images, random order
    python inspect_bbox.py fake_printed     # filter by class/keyword
    python inspect_bbox.py --no-bbox        # only show images with no detected face

Controls:
    [Next] / [Prev] buttons or ← → arrow keys  — navigate
    Transform buttons                           — modify image, re-detect face, update bbox
    [→ class] buttons                           — physically move image to that class folder
    Close window                                — quit

Transform buttons (overwrite image on disk + re-run MTCNN):
    ↻ Rotate CW   — rotate 90° clockwise
    ↺ Rotate CCW  — rotate 90° counter-clockwise
    ↕ Rotate 180  — rotate 180°
    ⊡ Crop Middle — crop centre 50% of image

Move: physically moves the image file into train/<class>/ and updates bbox_cache.json key.
"""

import json
import sys
import shutil
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image

BBOX_CACHE = Path("output/bbox_cache.json")
BASE_DIR   = Path(".")
TRAIN_DIR  = Path("train")

CLASSES = [
    "realperson",
    "fake_unknown",
    "fake_mask",
    "fake_screen",
    "fake_mannequin",
    "fake_printed",
]

CLASS_COLORS = {
    "realperson":     "#27ae60",
    "fake_unknown":   "#7f8c8d",
    "fake_mask":      "#8e44ad",
    "fake_screen":    "#2980b9",
    "fake_mannequin": "#c0392b",
    "fake_printed":   "#d35400",
}

TRANSFORMS = [
    ("rotate_cw",   "↻ Rotate CW",   "#e67e22"),
    ("rotate_ccw",  "↺ Rotate CCW",  "#e67e22"),
    ("rotate_180",  "↕ Rotate 180",  "#e67e22"),
    ("crop_middle", "⊡ Crop Middle", "#8e44ad"),
    ("zoom_out",    "⊖ Zoom Out 50%","#27ae60"),
]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_full_cache():
    with open(BBOX_CACHE, encoding="utf-8") as f:
        return json.load(f)


def save_full_cache(cache: dict):
    with open(BBOX_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def load_items(filter_kw: str = "", only_missing: bool = False):
    cache = load_full_cache()
    items = list(cache.keys())
    if only_missing:
        items = [p for p in items if cache[p] is None]
    if filter_kw:
        items = [p for p in items if filter_kw.lower() in p.lower()]
    random.shuffle(items)
    return items


def get_bbox(cache, img_path):
    entry = cache.get(img_path)
    if isinstance(entry, dict):
        return entry.get("bbox")
    return entry


# ── Image transforms + re-detection ───────────────────────────────────────────

def _redetect(img_path: str, cache: dict):
    """Run MTCNN on the (possibly modified) image and update bbox in cache."""
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(keep_all=False, device=device, select_largest=True,
                      min_face_size=20, post_process=False)
        img = Image.open(img_path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            new_bbox = [int(x1), int(y1), int(x2), int(y2)]
        else:
            new_bbox = None
        cache[img_path] = new_bbox
        save_full_cache(cache)
        print(f"  Re-detected bbox: {new_bbox}")
        return new_bbox
    except Exception as e:
        print(f"  [warn] MTCNN re-detection failed: {e}")
        cache[img_path] = None
        save_full_cache(cache)
        return None


def apply_transform(img_path: str, transform_key: str, cache: dict):
    """Transform image on disk, overwrite, then re-run face detection."""
    full_path = BASE_DIR / img_path
    if not full_path.exists():
        print(f"  [error] file not found: {full_path}")
        return

    img = Image.open(str(full_path)).convert("RGB")

    if transform_key == "rotate_cw":
        img = img.rotate(-90, expand=True)
    elif transform_key == "rotate_ccw":
        img = img.rotate(90, expand=True)
    elif transform_key == "rotate_180":
        img = img.rotate(180, expand=True)
    elif transform_key == "crop_middle":
        w, h = img.size
        left   = w // 4
        top    = h // 4
        right  = w - w // 4
        bottom = h - h // 4
        img = img.crop((left, top, right, bottom))
    elif transform_key == "zoom_out":
        w, h = img.size
        small = img.resize((w // 2, h // 2), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
        img = canvas

    img.save(str(full_path), quality=95)
    print(f"  ✔ Applied [{transform_key}] → {full_path.name}")
    _redetect(img_path, cache)


# ── Move to class ──────────────────────────────────────────────────────────────

def move_to_class(cache, items, idx, target_class):
    img_path = items[idx]
    src = BASE_DIR / img_path
    if not src.exists():
        print(f"  [error] file not found: {src}")
        return

    dest_dir = TRAIN_DIR / target_class
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and dest != src:
        dest = dest_dir / f"{src.stem}_moved{src.suffix}"

    shutil.move(str(src), str(dest))
    new_key = str(dest)

    old_entry = cache.pop(img_path, None)
    cache[new_key] = old_entry
    save_full_cache(cache)
    items[idx] = new_key
    print(f"  ➜ Moved [{target_class}]: {src.name}  ({img_path} → {new_key})")


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw(ax, img_path, cache, idx, total):
    ax.clear()
    full_path = BASE_DIR / img_path
    if not full_path.exists():
        ax.text(0.5, 0.5, f"File not found:\n{img_path}",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.axis("off")
        ax.figure.canvas.draw()
        return

    img = Image.open(str(full_path)).convert("RGB")
    ax.imshow(img)

    bbox = get_bbox(cache, img_path)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        ))
        ax.text(x1, y1 - 6, "face", color="lime", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
        bbox_str = f"bbox: [{x1}, {y1}, {x2}, {y2}]"
    else:
        bbox_str = "NO FACE DETECTED"

    current_class = Path(img_path).parts[-2] if len(Path(img_path).parts) >= 2 else "?"
    ax.figure.suptitle(
        f"[{idx+1}/{total}]  {current_class} — {Path(img_path).name}\n{bbox_str}",
        fontsize=9, y=0.99
    )
    ax.axis("off")
    ax.figure.canvas.draw()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    only_missing = "--no-bbox" in args
    filter_kw    = next((a for a in args if not a.startswith("--")), "")

    items = load_items(filter_kw, only_missing)
    if not items:
        print("No images match the filter.")
        return

    print(f"Found {len(items)} images.")

    cache = load_full_cache()
    state = {"idx": 0}

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 1 (bottom, y=0.02): transform buttons
    # Row 2 (y=0.095):        move-to-class buttons
    # Row 3 (y=0.165):        prev / next
    # Image (y=0.23+):        main view
    fig = plt.figure(figsize=(10, 9))
    ax  = fig.add_axes([0.04, 0.23, 0.92, 0.74])

    # Nav row
    ax_prev = fig.add_axes([0.20, 0.165, 0.25, 0.052])
    ax_next = fig.add_axes([0.55, 0.165, 0.25, 0.052])
    btn_prev = Button(ax_prev, "◀  Prev", color="#ecf0f1", hovercolor="#bdc3c7")
    btn_next = Button(ax_next, "Next  ▶", color="#ecf0f1", hovercolor="#bdc3c7")

    # Move-to-class row
    move_buttons = []
    n_cls   = len(CLASSES)
    cls_w   = 0.13
    cls_gap = (1.0 - n_cls * cls_w) / (n_cls + 1)
    for i, cls in enumerate(CLASSES):
        x = cls_gap + i * (cls_w + cls_gap)
        ax_c  = fig.add_axes([x, 0.095, cls_w, 0.058])
        btn_c = Button(ax_c, f"→ {cls}", color=CLASS_COLORS[cls], hovercolor="#f1c40f")
        btn_c.label.set_color("white")
        btn_c.label.set_fontsize(7)
        move_buttons.append((cls, btn_c))

    # Transform row
    tfm_buttons = []
    n_tfm   = len(TRANSFORMS)
    tfm_w   = 0.18
    tfm_gap = (1.0 - n_tfm * tfm_w) / (n_tfm + 1)
    for i, (tkey, tlabel, tcolor) in enumerate(TRANSFORMS):
        x = tfm_gap + i * (tfm_w + tfm_gap)
        ax_t  = fig.add_axes([x, 0.02, tfm_w, 0.058])
        btn_t = Button(ax_t, tlabel, color=tcolor, hovercolor="#f39c12")
        btn_t.label.set_color("white")
        btn_t.label.set_fontsize(8.5)
        tfm_buttons.append((tkey, btn_t))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def refresh():
        draw(ax, items[state["idx"]], cache, state["idx"], len(items))
        fig.canvas.draw_idle()

    def on_next(event=None):
        if state["idx"] < len(items) - 1:
            state["idx"] += 1
            refresh()

    def on_prev(event=None):
        if state["idx"] > 0:
            state["idx"] -= 1
            refresh()

    def make_transform_handler(tkey):
        def handler(event=None):
            apply_transform(items[state["idx"]], tkey, cache)
            refresh()  # redraw with new image + updated bbox
        return handler

    def make_move_handler(target_cls):
        def handler(event=None):
            move_to_class(cache, items, state["idx"], target_cls)
            if state["idx"] >= len(items):
                state["idx"] = len(items) - 1
            refresh()
        return handler

    def on_key(event):
        if event.key == "right":
            on_next()
        elif event.key == "left":
            on_prev()
        elif event.key in [str(i + 1) for i in range(len(TRANSFORMS))]:
            make_transform_handler(TRANSFORMS[int(event.key) - 1][0])()

    # ── Wire up ───────────────────────────────────────────────────────────────
    btn_next.on_clicked(on_next)
    btn_prev.on_clicked(on_prev)
    fig.canvas.mpl_connect("key_press_event", on_key)
    for tkey, btn in tfm_buttons:
        btn.on_clicked(make_transform_handler(tkey))
    for cls, btn in move_buttons:
        btn.on_clicked(make_move_handler(cls))

    print("  Keyboard shortcuts: 1=rotate_cw  2=rotate_ccw  3=rotate_180  4=crop_middle  5=zoom_out")
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
