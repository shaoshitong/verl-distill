#!/usr/bin/env python3
"""Draw the generator teacher-feature loss flow for STE vs MSE representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#dbeafe"
BLUE_E = "#2563eb"
GRAY = "#f1f5f9"
GRAY_E = "#94a3b8"
GREEN = "#dcfce7"
GREEN_E = "#16a34a"
RED = "#fee2e2"
RED_E = "#dc2626"


def box(ax, x, y, w, h, text, *, face, edge, fs=11, dashed=False, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.6,
        facecolor=face,
        edgecolor=edge,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, fontweight=weight)
    return (x, y, w, h)


def arrow(ax, start, end, *, color="#334155", style="-|>", width=1.6, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            linewidth=width,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(15.5, 11.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14.6)
    ax.axis("off")

    ax.text(
        10,
        14.32,
        "DMD generator update - teacher feature losses",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        10,
        13.9,
        "latent (0th) keeps STE          |          5th / 15th / 25th switched to MSE",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )

    # ---- forward path (left column) ----
    box(ax, 3.0, 12.4, 4.4, 0.75, "clean latent  x_real   (data)", face=GRAY, edge=GRAY_E)
    box(ax, 3.0, 11.3, 5.6, 0.75, "x_t = sigma*eps + (1-sigma)*x_real", face=GRAY, edge=GRAY_E)
    box(
        ax,
        3.0,
        10.2,
        4.4,
        0.8,
        "G_theta(x_t, sigma)   (generator)",
        face=BLUE,
        edge=BLUE_E,
        weight="bold",
    )
    box(
        ax,
        3.0,
        9.1,
        5.4,
        0.8,
        "x_hat0 = x_t - sigma*G_theta     (live x0)",
        face=BLUE,
        edge=BLUE_E,
        weight="bold",
    )
    arrow(ax, (3.0, 12.0), (3.0, 11.7))
    arrow(ax, (3.0, 10.9), (3.0, 10.6))
    arrow(ax, (3.0, 9.8), (3.0, 9.5))
    arrow(ax, (3.0, 8.7), (3.0, 7.15))

    # ---- score branches (middle) ----
    box(
        ax,
        10.4,
        9.1,
        5.6,
        0.8,
        "q_t = (1-t)*x_hat0.detach() + t*eps2",
        face=GRAY,
        edge=GRAY_E,
        dashed=True,
    )
    arrow(ax, (5.7, 9.1), (7.6, 9.1), color=GRAY_E, dashed=True)
    ax.text(6.6, 9.38, "detach", fontsize=9.5, color="#64748b", ha="center")
    box(
        ax,
        10.4,
        7.8,
        5.6,
        0.78,
        "R_cfg(q_t, t)  real teacher  ->  r0 = q_t - t*R",
        face=GRAY,
        edge=GRAY_E,
        dashed=True,
    )
    box(
        ax,
        10.4,
        6.5,
        5.6,
        0.78,
        "F_phi(q_t, t)  fake model   ->  f0 = q_t - t*F",
        face=GRAY,
        edge=GRAY_E,
        dashed=True,
    )
    arrow(ax, (10.4, 8.7), (10.4, 8.2), color=GRAY_E, dashed=True)
    arrow(ax, (10.4, 8.7), (10.4, 6.9), color=GRAY_E, dashed=True, rad=-0.25)

    # ---- representations (right) ----
    box(
        ax,
        16.9,
        9.9,
        5.2,
        0.85,
        "h_live = H_k(x_hat0)\n[differentiable -> G_theta]",
        face=BLUE,
        edge=BLUE_E,
        fs=10.5,
    )
    box(
        ax,
        16.9,
        7.8,
        5.2,
        0.85,
        "h_real = H_k(r0)\n[no_grad]",
        face=GRAY,
        edge=GRAY_E,
        fs=10.5,
        dashed=True,
    )
    box(
        ax,
        16.9,
        6.5,
        5.2,
        0.85,
        "h_fake = H_k(f0)\n[no_grad]",
        face=GRAY,
        edge=GRAY_E,
        fs=10.5,
        dashed=True,
    )
    arrow(ax, (5.7, 9.1), (14.3, 9.9), color=BLUE_E, rad=-0.16, width=2.0)
    arrow(ax, (13.2, 7.8), (14.3, 7.8), color=GRAY_E, dashed=True)
    arrow(ax, (13.2, 6.5), (14.3, 6.5), color=GRAY_E, dashed=True)
    ax.text(9.6, 10.55, "H_k = frozen teacher block-k features", fontsize=9.5, color="#475569")
    ax.text(15.0, 10.55, "J_k = d h_live / d theta_G", fontsize=10, color=BLUE_E)

    # ---- losses ----
    box(
        ax,
        4.0,
        5.9,
        6.6,
        2.0,
        "STE  (k = latent / 0th)\n"
        "L_k = mean( ( h_live - stopgrad( h_live + D_k ) )^2 )\n"
        "D_k = (h_real - h_fake) / denom\n"
        "grad = -2 * D_k * J_k",
        face=RED,
        edge=RED_E,
        fs=10,
        weight="normal",
    )
    box(
        ax,
        13.0,
        3.6,
        7.6,
        2.0,
        "MSE  (k = 5th / 15th / 25th)\n"
        "L_k = mean( ( h_live - h_real )^2 )\n"
        "h_real is a constant target (detached)\n"
        "grad = +2 * (h_live - h_real) * J_k",
        face=GREEN,
        edge=GREEN_E,
        fs=10.5,
        weight="normal",
    )
    arrow(ax, (16.9, 9.45), (16.9, 4.65), color=BLUE_E, width=2.0)
    arrow(ax, (16.9, 7.35), (16.2, 4.65), color=GRAY_E, dashed=True, rad=0.2)
    arrow(ax, (16.9, 6.05), (14.3, 4.65), color=GRAY_E, dashed=True, rad=0.15)
    arrow(ax, (5.2, 8.7), (4.4, 6.95), color=BLUE_E, width=2.0)
    arrow(ax, (13.0, 2.6), (10.2, 2.55), color=GREEN_E, width=1.8, rad=0.12)

    # ---- aggregation ----
    box(
        ax,
        8.9,
        2.15,
        8.6,
        0.95,
        "L_G = sum_k  w_k * L_k        (w_k from gradient-norm balancing, 8:4:2:1)",
        face="#e0e7ff",
        edge="#4f46e5",
        fs=11,
        weight="bold",
    )
    arrow(ax, (5.2, 4.9), (6.4, 2.62), color=RED_E, width=1.8, rad=-0.12)
    box(
        ax,
        8.9,
        0.95,
        9.4,
        0.9,
        "backward -> clip(2.83) -> Schedule-Free AdamW step on theta_G",
        face=BLUE,
        edge=BLUE_E,
        fs=11,
        weight="bold",
    )
    arrow(ax, (8.9, 1.67), (8.9, 1.42), color="#4f46e5", width=2.0)
    arrow(ax, (4.2, 0.95), (2.2, 0.95), color=BLUE_E, width=1.8)
    ax.text(
        0.35, 3.1, "gradient flows\nonly through\nh_live", fontsize=9.5, color=BLUE_E, ha="left"
    )
    arrow(ax, (2.2, 1.05), (0.95, 8.4), color=BLUE_E, width=1.6, rad=0.22)
    arrow(ax, (0.95, 8.4), (0.85, 9.9), color=BLUE_E, width=1.6)
    arrow(ax, (0.85, 9.9), (1.5, 9.9), color=BLUE_E, width=1.6)

    legend = [
        (BLUE, BLUE_E, "differentiable (gradient flows)"),
        (GRAY, GRAY_E, "detached / no_grad (target only)"),
        (RED, RED_E, "STE surrogate (latent)"),
        (GREEN, GREEN_E, "MSE regression (5/15/25)"),
    ]
    for index, (face, edge, label) in enumerate(legend):
        x = 0.65 + index * 4.85
        ax.add_patch(
            FancyBboxPatch(
                (x, 13.32),
                0.34,
                0.28,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
            )
        )
        ax.text(x + 0.48, 13.46, label, fontsize=10, va="center", color="#334155")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight", facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
