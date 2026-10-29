"""Build NSF_Dashboard_Demo.mp4 from the captured dashboard walkthrough frames.

Reads the nsf_frame_XX.png screenshots captured from the live dashboard, adds a
title card and a caption banner to each frame, and stitches them into an MP4
with short crossfade transitions.

Usage:
    python scripts/build_dashboard_video.py [frames_dir]

Requires: imageio[ffmpeg], pillow, numpy.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = Path(os.path.expandvars(r"%LOCALAPPDATA%\Temp\cursor\screenshots"))
OUT = ROOT / "NSF_Dashboard_Demo.mp4"

TEAL = (12, 92, 86)
INK = (26, 35, 50)

# Caption for each frame, keyed by frame number.
CAPTIONS = {
    1: "Coordinator view - global model accuracy and aggregate metrics only",
    2: "Strategy comparison: FedAvg vs FedProx vs FedAdagrad, plus test accuracy per round",
    3: "Neuro-symbolic confusion matrix and non-IID client partitions (Dirichlet split)",
    4: "Test cases filtered to symbolic overrides (where logic corrected the neural output)",
    5: "Override case: neural predicted Low Risk, symbolic rules corrected it to High Risk",
    6: "Verified explanation - hallucination guard passed, only fired rules are cited",
    7: "Same framework on Banking fraud: 99% accuracy, 100% hallucination-free",
    8: "Client role - each participant must sign in to see only their own data",
    9: "Signing in as Client 0 using local demo credentials",
    10: "Client 0 private view - only their own 367 records, never another client's data",
    11: "Predict - enter a card transaction to verify it in real time",
    12: "Result: Fraudulent - neural and symbolic layers agree (99% confidence)",
    13: "Verified explanation for the prediction - hallucination guard passed",
}

TITLE_MAIN = "Federated Neuro-Symbolic Framework"
TITLE_SUB = "Dashboard walkthrough - reliable, explainable, privacy-preserving reasoning"


def load_font(size, bold=False):
    candidates = (
        ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    )
    for name in candidates:
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def add_caption(img: Image.Image, step, total, text):
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(42)
    badge_font = load_font(34, bold=True)
    max_w = W - 160
    lines = wrap(draw, text, font, max_w)
    line_h = 54
    banner_h = 60 + line_h * len(lines)
    # translucent banner at the bottom
    draw.rectangle([0, H - banner_h, W, H], fill=(16, 30, 28, 220))
    # teal accent bar
    draw.rectangle([0, H - banner_h, 14, H], fill=(*TEAL, 255))
    y = H - banner_h + 22
    # step badge
    badge = f"STEP {step}/{total}"
    draw.text((40, y), badge, font=badge_font, fill=(150, 220, 205, 255))
    y += 46
    for ln in lines:
        draw.text((40, y), ln, font=font, fill=(245, 248, 247, 255))
        y += line_h
    return img


def make_title(size):
    W, H = size
    img = Image.new("RGB", (W, H), (247, 247, 243))
    draw = ImageDraw.Draw(img)
    # teal header band
    draw.rectangle([0, 0, W, 16], fill=TEAL)
    # NSF mark
    box = 150
    bx, by = (W - box) // 2, H // 2 - 220
    draw.rectangle([bx, by, bx + box, by + box], fill=TEAL)
    mf = load_font(70, bold=True)
    tw = draw.textlength("NSF", font=mf)
    draw.text((bx + (box - tw) / 2, by + 34), "NSF", font=mf, fill=(255, 255, 255))
    # title
    f1 = load_font(72, bold=True)
    f2 = load_font(40)
    t1w = draw.textlength(TITLE_MAIN, font=f1)
    draw.text(((W - t1w) / 2, by + box + 50), TITLE_MAIN, font=f1, fill=INK)
    for i, ln in enumerate(wrap(draw, TITLE_SUB, f2, W - 300)):
        lw = draw.textlength(ln, font=f2)
        draw.text(((W - lw) / 2, by + box + 150 + i * 52), ln, font=f2, fill=(80, 90, 102))
    return np.asarray(img)


def pad16(arr):
    h, w = arr.shape[:2]
    nh = (h + 15) // 16 * 16
    nw = (w + 15) // 16 * 16
    if nh == h and nw == w:
        return arr
    out = np.full((nh, nw, 3), 255, dtype=np.uint8)
    out[:h, :w] = arr
    return out


def build(frames_dir: Path):
    files = sorted(glob.glob(str(frames_dir / "nsf_frame_*.png")))
    if not files:
        raise SystemExit(f"No frames found in {frames_dir}")
    total = len(files)

    keyframes = []
    # first captured frame defines the canvas size
    base = Image.open(files[0]).convert("RGB")
    size = base.size

    keyframes.append(make_title(size))
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        if img.size != size:
            img = img.resize(size)
        cap = CAPTIONS.get(i, "")
        img = add_caption(img, i, total, cap)
        keyframes.append(np.asarray(img))

    keyframes = [pad16(k) for k in keyframes]

    fps = 30
    hold_title = int(fps * 3.0)
    hold = int(fps * 3.2)
    fade = int(fps * 0.5)

    writer = imageio.get_writer(
        str(OUT), fps=fps, codec="libx264", quality=8,
        macro_block_size=16, pixelformat="yuv420p",
    )
    try:
        for idx, frame in enumerate(keyframes):
            hold_n = hold_title if idx == 0 else hold
            for _ in range(hold_n):
                writer.append_data(frame)
            # crossfade into the next keyframe
            if idx < len(keyframes) - 1:
                nxt = keyframes[idx + 1]
                for s in range(1, fade + 1):
                    a = s / (fade + 1)
                    blended = (frame * (1 - a) + nxt * a).astype(np.uint8)
                    writer.append_data(blended)
    finally:
        writer.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    fd = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAMES
    build(fd)
