"""
Check THz-TDS alignment methods.

This script:
1. Reads your YAML configuration.
2. Loads reference and sample folders.
3. Compares raw traces vs aligned traces.
4. Tests different alignment methods.
5. Tests different correlation windows for correlation-based methods.
6. Saves every plot as a separate PNG/PDF figure.

Run:
    python examples/check_alignment_methods.py config/my_experiment.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

from thz_tds.dataset import THZDataset


# ============================================================
# User settings
# ============================================================

METHODS_TO_TEST = [

 "correlation_subsample",
    
]

# Used only for correlation and correlation_subsample.
# None means full trace.  Add more values here to sweep windows.
CORR_WINDOWS_PS = [2, 5, 10, 20, None]

# Plot only first N individual traces to avoid overcrowded figures.
MAX_TRACES_TO_PLOT = 20


# ============================================================
# Helpers
# ============================================================

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(text: str) -> str:
    """
    Make a string safe for filenames.
    """
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "p")
    )


def get_dataset_paths(cfg: dict) -> list[tuple[str, str]]:
    """
    Return list of datasets to test:
        [("reference_air3", "/path/to/air3"), ("sample_point3", "/path/to/point3"), ...]
    """
    paths = []

    reference = cfg["paths"]["reference"]
    paths.append((f"reference_{Path(reference).name}", reference))

    samples = cfg["paths"].get("samples", [])
    for sample_path in samples:
        paths.append((f"sample_{Path(sample_path).name}", sample_path))

    return paths


def plot_raw_vs_aligned(
    ds: THZDataset,
    dataset_label: str,
    method: str,
    corr_window_ps: float | None,
) -> plt.Figure:
    """Plot raw traces and aligned traces in one figure."""
    t = ds.x_ps
    raw = ds.all_y_raw
    aligned = ds.all_y
    avg_raw = np.mean(raw, axis=0)
    avg_aligned = ds.y_avg

    n_traces = raw.shape[0]
    n_plot = min(n_traces, MAX_TRACES_TO_PLOT)

    window_text = "full trace" if corr_window_ps is None else f"{corr_window_ps:g} ps"

    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(n_plot):
        ax.plot(t, raw[i],     lw=0.7, alpha=0.35, color="tab:blue",
                label="raw traces" if i == 0 else None)
    for i in range(n_plot):
        ax.plot(t, aligned[i], lw=0.7, alpha=0.35, color="tab:orange", ls="--",
                label="aligned traces" if i == 0 else None)

    ax.plot(t, avg_raw,     lw=2.2, color="tab:blue",   label="raw average")
    ax.plot(t, avg_aligned, lw=2.2, color="tab:orange", ls="--", label="aligned average")

    ax.set_title(
        f"{dataset_label} — method: {method},  window: {window_text},  "
        f"{n_traces} traces"
    )
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Amplitude [nA]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_zoom_main_pulse(
    ds: THZDataset,
    dataset_label: str,
    method: str,
    corr_window_ps: float | None,
    zoom_width_ps: float = 20.0,
) -> plt.Figure:
    """Plot a zoom around the strongest pulse."""
    t = ds.x_ps
    raw = ds.all_y_raw
    aligned = ds.all_y
    avg_raw = np.mean(raw, axis=0)
    avg_aligned = ds.y_avg

    idx_peak = int(np.argmax(np.abs(avg_raw)))
    t0 = float(t[idx_peak])
    mask = (t >= t0 - zoom_width_ps / 2) & (t <= t0 + zoom_width_ps / 2)

    n_plot = min(raw.shape[0], MAX_TRACES_TO_PLOT)
    window_text = "full trace" if corr_window_ps is None else f"{corr_window_ps:g} ps"

    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(n_plot):
        ax.plot(t[mask], raw[i, mask],     lw=0.7, alpha=0.35, color="tab:blue",
                label="raw traces" if i == 0 else None)
    for i in range(n_plot):
        ax.plot(t[mask], aligned[i, mask], lw=0.7, alpha=0.35, color="tab:orange", ls="--",
                label="aligned traces" if i == 0 else None)

    ax.plot(t[mask], avg_raw[mask],     lw=2.2, color="tab:blue",   label="raw average")
    ax.plot(t[mask], avg_aligned[mask], lw=2.2, color="tab:orange", ls="--", label="aligned average")

    ax.set_title(
        f"Zoom — main pulse: {dataset_label}\n"
        f"method: {method},  window: {window_text}"
    )
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Amplitude [nA]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_jitter_shifts(
    ds: THZDataset,
    dataset_label: str,
    method: str,
    corr_window_ps: float | None,
) -> plt.Figure | None:
    """Plot applied jitter shifts for each scan."""
    shifts = ds.jitter_shifts_ps
    if shifts is None:
        return None

    window_text = "full trace" if corr_window_ps is None else f"{corr_window_ps:g} ps"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(shifts)), shifts, marker="o", lw=1.5)
    ax.axhline(0.0, ls="--", lw=1.0, color="black")

    ax.set_title(
        f"Jitter shifts: {dataset_label}\n"
        f"method: {method},  window: {window_text}"
    )
    ax.set_xlabel("Scan index")
    ax.set_ylabel("Applied shift [ps]")
    ax.grid(True, alpha=0.3)

    text = (
        f"mean = {float(np.mean(shifts)):.4f} ps\n"
        f"std  = {float(np.std(shifts)):.4f} ps\n"
        f"min  = {float(np.min(shifts)):.4f} ps\n"
        f"max  = {float(np.max(shifts)):.4f} ps"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", alpha=0.15))
    fig.tight_layout()
    return fig


def plot_alignment_quality_metric(
    results: list[dict],
    dataset_label: str,
) -> plt.Figure:
    """Bar chart comparing peak amplitude and RMS across all alignment methods."""
    labels, peak_values, rms_values = [], [], []

    for item in results:
        ds     = item["dataset"]
        method = item["method"]
        window = item["corr_window_ps"]
        window_text = "full" if window is None else f"{window:g} ps"
        labels.append(f"{method}\n{window_text}")
        peak_values.append(float(np.max(np.abs(ds.y_avg))))
        rms_values.append(float(np.sqrt(np.mean(ds.y_avg ** 2))))

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, peak_values, color="tab:blue", alpha=0.8)
    axes[0].set_title("Peak of average trace")
    axes[0].set_ylabel("max(|average trace|)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, rms_values, color="tab:orange", alpha=0.8)
    axes[1].set_title("RMS of average trace")
    axes[1].set_ylabel("RMS")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Alignment quality comparison — {dataset_label}", fontsize=11)
    fig.tight_layout()
    return fig


def run_for_dataset(
    dataset_label: str,
    dataset_path: str | Path,
    cfg: dict,
) -> list[plt.Figure]:
    correction_factor = cfg["loading"]["correction_factor"]

    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_label}")
    print(f"Path:    {dataset_path}")
    print("=" * 80)

    all_results = []
    figures: list[plt.Figure] = []

    for method in METHODS_TO_TEST:

        if method in ["correlation", "correlation_subsample"]:
            windows_to_test = CORR_WINDOWS_PS
        else:
            windows_to_test = [None]

        for corr_window_ps in windows_to_test:

            window_text = "full_trace" if corr_window_ps is None else f"{corr_window_ps:g} ps"
            print(f"Testing method={method}, corr_window_ps={window_text}")

            try:
                ds = THZDataset(
                    path=dataset_path,
                    correction_factor=correction_factor,
                    align=True,
                    align_method=method,
                    corr_window_ps=corr_window_ps,
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue

            all_results.append(
                {
                    "method": method,
                    "corr_window_ps": corr_window_ps,
                    "dataset": ds,
                }
            )

            figures.append(plot_raw_vs_aligned(
                ds=ds,
                dataset_label=dataset_label,
                method=method,
                corr_window_ps=corr_window_ps,
            ))

            figures.append(plot_zoom_main_pulse(
                ds=ds,
                dataset_label=dataset_label,
                method=method,
                corr_window_ps=corr_window_ps,
                zoom_width_ps=20.0,
            ))

            fig_jitter = plot_jitter_shifts(
                ds=ds,
                dataset_label=dataset_label,
                method=method,
                corr_window_ps=corr_window_ps,
            )
            if fig_jitter is not None:
                figures.append(fig_jitter)

    if all_results:
        figures.append(plot_alignment_quality_metric(
            results=all_results,
            dataset_label=dataset_label,
        ))

    return figures


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) >= 2:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    cfg = load_yaml(config_path)

    dataset_paths = get_dataset_paths(cfg)

    print(f"Config: {config_path}")

    all_figures: list[plt.Figure] = []
    for dataset_label, dataset_path in dataset_paths:
        figs = run_for_dataset(
            dataset_label=dataset_label,
            dataset_path=dataset_path,
            cfg=cfg,
        )
        all_figures.extend(figs)

    print(f"\nDone. Showing {len(all_figures)} figure(s).")
    plt.show()


if __name__ == "__main__":
    main()