#!/usr/bin/env python3
"""Load the PA6_B TSV and plot all measurement points plus the averages."""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_FILE = Path(
    "/mnt/samples/results/fp_batch/20261001_100331/batch/PA6_B_results_20261001_100331.tsv"
)


def load_results(filepath):
    """Read the tab-separated file and return data sorted by frequency."""
    data = pd.read_csv(filepath, sep="\t")
    return data.sort_values("frequency_THz")


def plot_results(data, title="PA6_B"):
    """Return a figure with refractive index and absorption versus frequency."""
    frequency = data["frequency_THz"]
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )

    for point in range(1, 11):
        axes[0].plot(
            frequency, data[f"point{point}_n"],
            linewidth=1, alpha=0.55, label=f"Point {point}"
        )
        axes[1].plot(
            frequency, data[f"point{point}_alpha"],
            linewidth=1, alpha=0.55
        )

    axes[0].plot(
        frequency, data["avg_n"], color="black", linewidth=2.5, label="Average"
    )
    axes[1].plot(
        frequency, data["avg_alpha"], color="black", linewidth=2.5
    )

    axes[0].set_ylabel("Refractive index n")
    # The TSV header does not specify alpha's unit. Set your actual unit here.
    axes[1].set_ylabel(r"Absorption coefficient $\alpha$ (stored units)")
    axes[1].set_xlabel("Frequency (THz)")
    axes[0].legend(ncol=4, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a window.")
    args = parser.parse_args()

    data = load_results(args.file)
    fig = plot_results(data, title=args.file.stem.split("_results")[0])
    output = args.file.with_suffix(".png")
    fig.savefig(output, dpi=300)
    print(f"Loaded {len(data)} frequency rows. Plot saved to: {output}")

    if not args.no_show:
        plt.show()
    plt.close(fig)
