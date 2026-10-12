"""Redraw of manuscript Fig. 2 (architecture) as a vector PDF, matching the revised method text.

Changes vs. the submitted figure: UniSal marked frozen; "Flatten" replaced by the CNN encoder phi;
"Time-series Model" -> LSTM; two output heads (gaze, task) instead of "Gaze & View";
horizon-selection block added; notation box uses "current frame".
Input icons: a real EHTask viewport frame and its UniSal saliency map.
Output: paper_review/revision_assets/figures/fig2_architecture.{pdf,png}
"""
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
from core import utils  # noqa: E402

OUT = "/mnt/DataLake/AlienBackup/access/paper_review/revision_assets/figures"
CLIP = "/mnt/DataLake/AlienBackup/access/processed_dataset/eht_dataset_prossesed_vid/Video_2/User_11_Task_3/Video_2_User_11_Task_3_10.mp4"

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, RED, GREY = "#2a78d6", "#e34948", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7.5,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ icons from real data
def load_icons(n=3, step=5):
    cap = cv2.VideoCapture(CLIP)
    frames, sals = [], []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unisal = utils.load_unisal(os.path.join(os.path.dirname(HERE), "models", "unisal.pth"), dev)
    k = 0
    while len(frames) < n:
        ret, fr = cap.read()
        if not ret:
            break
        if k % step == 0:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                s = unisal(utils.preprocess_sal_img(fr).unsqueeze(0).to(dev))[0, 0].cpu().numpy()
            sals.append((s - s.min()) / (s.max() - s.min() + 1e-9))
        k += 1
    cap.release()
    return frames, sals


# ------------------------------------------------------------------ drawing helpers
def box(ax, x, y, w, h, text=None, ec=INK, fc="white", lw=0.8, r=0.6, ls="-", fs=7.5, weight="normal", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                weight=weight, zorder=z + 1, linespacing=1.35)


def arrow(ax, p0, p1, color=INK, lw=0.9, ls="-", conn="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                 ls=ls, connectionstyle=conn, shrinkA=0, shrinkB=0, zorder=4))


def stack(ax, imgs, x, y, w, h, dx=1.6, dy=1.4, cmap=None):
    for i, im in enumerate(imgs):
        xi, yi = x + i * dx, y - i * dy
        ax.imshow(im, extent=(xi, xi + w, yi, yi + h), cmap=cmap, zorder=3 + i, interpolation="bilinear",
                  aspect="auto")
        ax.add_patch(Rectangle((xi, yi), w, h, fill=False, ec=INK2, lw=0.5, zorder=3 + i))


def main():
    frames, sals = load_icons()
    fig = plt.figure(figsize=(7.2, 3.55))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # ---------------- Saliency encoder (top)
    box(ax, 1, 34, 82, 24, ec=BLUE, lw=1.4, r=2.5, z=1)
    ax.text(2, 59, "Saliency Encoder", fontsize=8.5, color=INK, va="bottom")
    ax.text(10, 54.3, "Viewport frames", ha="center", fontsize=7.5)
    stack(ax, frames, 3.5, 44, 11, 8.3)
    ax.text(10, 36.2, r"$I_{f-p+1},\ldots,I_f$", ha="center", fontsize=7.5)
    arrow(ax, (19.6, 45.5), (23.5, 45.5))
    box(ax, 23.5, 42.2, 11, 6.6, "UniSal\n(frozen)")
    arrow(ax, (34.5, 45.5), (38.5, 45.5))
    ax.text(45, 54.3, "Saliency maps", ha="center", fontsize=7.5)
    stack(ax, sals, 38.5, 44, 9.5, 8.3, cmap="viridis")
    ax.text(45, 36.2, r"$\mathbf{S}_{1:p}=[S_{f-p+1},\ldots,S_f]$", ha="center", fontsize=7.5)
    arrow(ax, (53.2, 45.5), (57, 45.5))
    box(ax, 57, 41.4, 23.5, 8.2, "CNN encoder $\\phi$\n$25{\\times}25 \\rightarrow 10$-d per frame")
    ax.text(68.75, 38.4, r"$\phi(S_t)\in\mathbb{R}^{10}$", ha="center", fontsize=7.5, color=INK2)

    # ---------------- Notation (top right)
    box(ax, 86, 38, 37, 20, ec=INK, lw=0.8, r=0.4)
    ax.text(88, 55.2, "Notation", fontsize=8, weight="bold", va="center")
    for i, t in enumerate([r"Current (last observed) frame: $f$",
                           r"Past frames: $p=15$ (500 ms)",
                           r"Future steps: $q=20$ (666 ms)",
                           r"Sampling rate: 30 Hz",
                           r"Task conditions: $v\in\mathcal{V}$, $|\mathcal{V}|=4$"]):
        ax.text(88, 51.6 - i * 3.0, t, fontsize=7, va="center", color=INK)

    # ---------------- Sequence encoder (bottom)
    box(ax, 1, 1, 102, 29.5, ec=RED, lw=1.4, r=2.5, z=1)
    ax.text(2, 31.2, "Sequence Encoder", fontsize=8.5, color=INK, va="bottom")
    ax.text(3, 26.8, "Angular velocity", fontsize=7.5)
    box(ax, 3, 19.3, 27, 6.3)
    ax.text(16.5, 23.6, r"$\boldsymbol{\omega}_t=(\omega^{az}_t,\ \omega^{el}_t)$", ha="center", va="center", fontsize=7.2)
    ax.text(16.5, 20.9, r"$t=f-p+1,\ldots,f$", ha="center", va="center", fontsize=6.6, color=INK2)
    ax.text(3, 15.3, "Angular displacement", fontsize=7.5)
    box(ax, 3, 7.8, 27, 6.3)
    ax.text(16.5, 12.1, r"$\Delta\boldsymbol{\theta}_t=(\Delta\theta^{az}_t,\ \Delta\theta^{el}_t)$", ha="center", va="center", fontsize=7.2)
    ax.text(16.5, 9.4, r"$t=f-p+1,\ldots,f$", ha="center", va="center", fontsize=6.6, color=INK2)
    arrow(ax, (30, 22.4), (32.6, 18.3), conn="arc3,rad=0")
    arrow(ax, (30, 10.9), (32.6, 15.0), conn="arc3,rad=0")
    box(ax, 32.6, 9.5, 21.8, 14.5)
    ax.text(43.5, 20.6, r"$\mathbf{H}_{1:p}=[\mathbf{h}_{f-p+1},\ldots,\mathbf{h}_f]$", ha="center", fontsize=7.2)
    ax.text(43.5, 16.8, "where", ha="center", fontsize=7)
    ax.text(43.5, 13.0, r"$\mathbf{h}_t=(\omega^{az}_t,\omega^{el}_t,\Delta\theta^{az}_t,\Delta\theta^{el}_t)$", ha="center", fontsize=6.6)

    # concat
    cx, cy, cr = 58, 16.75, 1.7
    ax.add_patch(Circle((cx, cy), cr, fc="white", ec=INK, lw=0.8, zorder=5))
    ax.plot([cx - cr, cx + cr], [cy, cy], color=INK, lw=0.8, zorder=6)
    ax.plot([cx, cx], [cy - cr, cy + cr], color=INK, lw=0.8, zorder=6)
    arrow(ax, (54.4, cy), (cx - cr, cy))
    arrow(ax, (cx, 41.4), (cx, cy + cr))  # phi(S_t) down into concat
    ax.text(cx, 11.2, "concat", ha="center", fontsize=7)
    ax.text(cx, 7.9, r"$\mathbf{T}_t=[\phi(S_t),\mathbf{h}_t]$", ha="center", fontsize=6.8)

    # LSTM
    arrow(ax, (cx + cr, cy), (63.5, cy))
    box(ax, 63.5, 12.7, 10.5, 8.1, "LSTM\n(32 units)")
    ax.text(68.75, 9.6, r"$\mathbf{o}_{1:p}$", ha="center", fontsize=7, color=INK2)

    # two heads
    box(ax, 78, 18.4, 23.5, 10.8)
    ax.text(89.75, 26.9, r"Gaze head $f_{\mathrm{reg}}$ (linear)", ha="center", va="center", fontsize=7.2)
    ax.text(89.75, 23.8, r"input $\mathbf{z}=\mathbf{o}_p$", ha="center", va="center", fontsize=6.8, color=INK2)
    ax.text(89.75, 20.6, r"$\hat{\mathbf{Y}}^{\mathrm{reg}}_{1:q}\in\mathbb{R}^{q\times 2}$", ha="center", va="center", fontsize=7.2)
    box(ax, 78, 3.4, 23.5, 10.8)
    ax.text(89.75, 11.9, r"Task head $f_{\mathrm{cls}}$ (MLP)", ha="center", va="center", fontsize=7.2)
    ax.text(89.75, 8.8, r"input $\bar{\mathbf{z}}$ = mean of $\mathbf{o}_t$", ha="center", va="center", fontsize=6.8, color=INK2)
    ax.text(89.75, 5.7, r"$\hat{p}(v),\ v\in\mathcal{V}$", ha="center", va="center", fontsize=7.2)
    arrow(ax, (74, 18.5), (78, 23.8), conn="arc3,rad=0")
    arrow(ax, (74, 15.0), (78, 8.8), conn="arc3,rad=0")

    # ---------------- Horizon selection (inference)
    box(ax, 104.5, 3.4, 19.3, 25.8, ec=GREY, lw=0.9, ls=(0, (3, 2)), r=1.2)
    ax.text(114.15, 26.4, "Horizon selection", ha="center", fontsize=7, weight="bold")
    ax.text(114.15, 23.6, "(inference)", ha="center", fontsize=6.8, color=INK2)
    ax.text(114.15, 18.6, r"$\hat{v}=\mathrm{arg\,max}\ \hat{p}(v)$", ha="center", fontsize=7)
    ax.text(114.15, 14.6, r"$H(\hat{v})$ fixed on", ha="center", fontsize=7)
    ax.text(114.15, 12.0, "validation data", ha="center", fontsize=7)
    ax.text(114.15, 7.0, r"output $\hat{\mathbf{Y}}^{\mathrm{reg}}_{1:H(\hat{v})}$", ha="center", fontsize=7)
    arrow(ax, (101.5, 8.8), (104.5, 12.5), conn="arc3,rad=0")
    arrow(ax, (101.5, 23.8), (104.5, 20.5), conn="arc3,rad=0")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig2_architecture.pdf"), dpi=600)  # vector; dpi only affects the embedded thumbnails
    fig.savefig(os.path.join(OUT, "fig2_architecture.png"), dpi=300)
    print("saved")


if __name__ == "__main__":
    main()
