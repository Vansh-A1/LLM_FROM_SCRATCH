#!/usr/bin/env python3
"""Render the published training curves from portable CSV files.

Usage:
    python plot_training_metrics.py --data-dir results --output-dir docs/figures

Requires numpy and matplotlib. This script performs no training, checkpoint
loading, network access, or modification of its input CSV files.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#34435A", "text.color": "#152B43",
        "xtick.color": "#526477", "ytick.color": "#526477",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CDDAE4", "axes.grid": True,
        "grid.color": "#E6EDF2", "grid.linewidth": 0.7,
        "axes.axisbelow": True, "svg.hashsalt": "gpt2-training-metrics-v1",
    })


def save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white", metadata={"Date": None})
    svg_path = output_dir / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_losses(train, validation, output_dir):
    step, loss = train["step"], train["train_loss"]
    assert np.array_equal(step, np.arange(1, 17235)), "Expected complete update coverage"
    assert np.isfinite(loss).all(), "Nonfinite training loss"
    window = 100
    smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.25), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.07, right=.975, bottom=.29, top=.77, wspace=.25)
    fig.text(.07, .94, "GPT-2 Small · measured training loss", fontsize=20, weight="bold")
    fig.text(.07, .875, "124.44M parameters  /  17,234 updates  /  9.036B training-token exposures", fontsize=11, color="#526477")
    for ax in axes:
        ax.plot(step, loss, color="#96BBC9", alpha=.35, linewidth=.5, label="Train · individual update", rasterized=True)
        ax.plot(step[window - 1:], smooth, color="#146C82", linewidth=1.8, label="Train · trailing 100-update mean")
        ax.plot(validation["step"], validation["val_loss"], color="#CB6B35", linewidth=1.65,
                marker="o", markersize=3.6, label="Sampled validation · measured points")
        ax.set_xlabel("Completed optimizer updates", labelpad=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylabel("Cross-entropy loss (nats)", labelpad=9)
    axes[0].set_title("Complete run", loc="left", fontsize=11, weight="bold", pad=12)
    axes[0].set_xlim(0, 17500)
    axes[0].set_ylim(2.7, 11.4)
    axes[0].set_xticks([0, 4000, 8000, 12000, 16000])
    axes[1].set_title("Later updates · expanded loss scale", loc="left", fontsize=11, weight="bold", pad=12)
    axes[1].set_xlim(8000, 17700)
    axes[1].set_ylim(2.98, 3.43)
    axes[1].set_xticks([8000, 10000, 12000, 14000, 16000])
    axes[1].annotate(f"Final validation  {validation['val_loss'][-1]:.4f}",
                     xy=(step[-1], validation["val_loss"][-1]), xytext=(10300, 3.35),
                     fontsize=10, color="#A74D24", arrowprops={"arrowstyle": "-", "color": "#CB6B35", "lw": .9})
    axes[1].annotate(f"Last 100 train mean  {smooth[-1]:.4f}",
                     xy=(step[-1], smooth[-1]), xytext=(9300, 3.015),
                     fontsize=10, color="#146C82", arrowprops={"arrowstyle": "-", "color": "#146C82", "lw": .9})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.065, .095), frameon=False,
               ncol=3, fontsize=9.5, handlelength=2.5)
    fig.text(.07, .055, "All 17,234 unique update records; replayed update 5 uses its last record. Validation points are connected only as a visual guide.",
             fontsize=8.6, color="#68798B")
    fig.text(.07, .023, "The update axis does not show interruption duration. Sampled training-time validation is distinct from the separate full-file evaluation.",
             fontsize=8.6, color="#68798B")
    save(fig, output_dir, "training_loss")


def configured_lr(index):
    return np.where(index < 2000, .0006 * (index + 1) / 2001,
                    .00006 + .5 * (1 + np.cos(np.pi * np.minimum((index - 2000) / 598000, 1))) * .00054)


def plot_lr(train, output_dir):
    step, lr = train["step"], train["lr"]
    assert np.allclose(lr, configured_lr(step - 1), rtol=0, atol=1e-15), "Logged LR differs from documented formula"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.07, right=.975, bottom=.29, top=.76, wspace=.25)
    fig.text(.07, .94, "Learning rate · observed run and configured schedule", fontsize=19, weight="bold")
    fig.text(.07, .875, "2,000 warmup updates  /  peak 6 × 10⁻⁴  /  minimum 6 × 10⁻⁵  /  decay horizon 600,000", fontsize=11, color="#526477")
    axes[0].plot(step, lr, color="#146C82", linewidth=2, label="Recorded learning rate")
    axes[0].scatter(step[-1], lr[-1], color="#CB6B35", s=26, zorder=4)
    axes[0].axvline(2000, color="#B6C3CE", ls=":", lw=1)
    axes[0].set_xlim(0, 17700)
    axes[0].set_xticks([0, 4000, 8000, 12000, 16000])
    axes[0].set_title("Observed: completed run", loc="left", fontsize=11, weight="bold", pad=12)
    axes[0].annotate(f"Final LR  {lr[-1]:.8f}", xy=(step[-1], lr[-1]), xytext=(6900, .00043),
                     color="#A74D24", fontsize=10, arrowprops={"arrowstyle": "-", "color": "#CB6B35", "lw": .9})
    future = np.linspace(step[-1], 600000, 1500)
    axes[1].plot(future, configured_lr(future - 1), color="#8D9EAE", ls="--", linewidth=1.6, label="Configured continuation · not run")
    axes[1].plot(step, lr, color="#146C82", linewidth=2, label="Recorded run")
    axes[1].axvspan(0, step[-1], color="#DDECF1", alpha=.8, zorder=-1)
    axes[1].set_title("Full configured horizon · only 2.87% completed", loc="left", fontsize=11, weight="bold", pad=12)
    axes[1].set_xlim(0, 600000)
    axes[1].set_xticks([0, 150000, 300000, 450000, 600000])
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.6)
    for ax in axes:
        ax.set_ylim(0, .00066)
        ax.set_xlabel("Completed optimizer updates", labelpad=9)
        ax.set_ylabel("Learning rate", labelpad=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4), useMathText=True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.text(.07, .105, "The run completed one epoch-equivalent token budget using randomly sampled windows. Dashed continuation: configured but not run.", fontsize=9, color="#526477")
    fig.text(.07, .065, "Update n used schedule index n − 1. Warmup uses (index + 1) / 2,001; cosine decay uses indices 2,000 through 600,000.", fontsize=8.6, color="#68798B")
    fig.text(.07, .025, "The learning-rate configuration is descriptive evidence; it does not establish the cause of generation-quality limitations.", fontsize=8.6, color="#68798B")
    save(fig, output_dir, "learning_rate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = read_csv(args.data_dir / "training_metrics.csv")
    validation = read_csv(args.data_dir / "validation_metrics.csv")
    configure()
    plot_losses(train, validation, args.output_dir)
    plot_lr(train, args.output_dir)


if __name__ == "__main__":
    main()
