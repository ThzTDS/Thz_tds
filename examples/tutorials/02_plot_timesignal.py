"""
02_air4_point4_plot.py — raw time-domain viewer
================================================
Reads all_samples_fp_batch.yaml, discovers air/point subfolder pairs for
each active sample, and plots the RAW (unaligned) time-domain traces.

For every pair two subplots are produced side-by-side:
  Left  — all raw air traces (grey) + average (blue)
  Right — all raw point traces (grey) + average (orange)

Usage
-----
    python examples/02_air4_point4_plot.py
    python examples/02_air4_point4_plot.py config/all_samples_fp_batch.yaml
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

import thz_tds


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


def _label(name: str) -> str:
    """Keep material and variant, drop thickness: 'PA6_B_3' → 'PA6_B'."""
    parts = name.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_pairs(
    sample_dir: Path,
    air_prefix: str,
    point_prefix: str,
    n_points: int | None,
) -> list[tuple[int, Path, Path]]:
    air_re = re.compile(rf"^{re.escape(air_prefix)}(\d+)$", re.IGNORECASE)
    pairs: list[tuple[int, Path, Path]] = []
    for child in sorted(sample_dir.iterdir()):
        if not child.is_dir():
            continue
        m = air_re.match(child.name)
        if not m:
            continue
        idx = int(m.group(1))
        point_path = sample_dir / f"{point_prefix}{idx}"
        if point_path.is_dir():
            pairs.append((idx, child, point_path))
    pairs.sort(key=lambda t: t[0])
    if n_points is not None:
        pairs = pairs[:n_points]
    return pairs


def _plot_pair(
    air_path: Path,
    sam_path: Path,
    title: str,
    ds_kwargs: dict,
    n_traces_plot: int | None,
    figsize: tuple,
) -> None:
    """Load and plot one air/sample pair side by side (raw traces + average)."""
    print(f"  Loading: {air_path.name}  +  {sam_path.name}")
    air_ds = thz_tds.THZDataset(path=air_path, **ds_kwargs)
    sam_ds = thz_tds.THZDataset(path=sam_path, **ds_kwargs)

    t_air   = air_ds.x_ps;  all_air = air_ds.all_y;  avg_air = air_ds.y_avg;  n_air = all_air.shape[0]
    t_sam   = sam_ds.x_ps;  all_sam = sam_ds.all_y;  avg_sam = sam_ds.y_avg;  n_sam = all_sam.shape[0]

    def _idx(n: int) -> list[int]:
        if n_traces_plot is None or n_traces_plot >= n:
            return list(range(n))
        step = max(1, n // n_traces_plot)
        return list(range(0, n, step))

    idx_air = _idx(n_air)
    idx_sam = _idx(n_sam)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    for i in idx_air:
        ax_l.plot(t_air, all_air[i], color="tab:gray", lw=0.6, alpha=0.25,
                  label="raw traces" if i == idx_air[0] else None)
    ax_l.plot(t_air, avg_air, color="tab:blue", lw=2.0, label=f"average  (n={n_air})")
    ax_l.set_title(f"Reference — {air_path.name}  [{n_air} traces]")
    ax_l.set_xlabel("Time (ps)");  ax_l.set_ylabel("Amplitude (a.u.)")
    ax_l.legend(fontsize=8, loc="upper right");  ax_l.grid(True, alpha=0.3)

    for i in idx_sam:
        ax_r.plot(t_sam, all_sam[i], color="tab:gray", lw=0.6, alpha=0.25,
                  label="raw traces" if i == idx_sam[0] else None)
    ax_r.plot(t_sam, avg_sam, color="tab:orange", lw=2.0, label=f"average  (n={n_sam})")
    ax_r.set_title(f"Sample — {sam_path.name}  [{n_sam} traces]")
    ax_r.set_xlabel("Time (ps)");  ax_r.set_ylabel("Amplitude (a.u.)")
    ax_r.legend(fontsize=8, loc="upper right");  ax_r.grid(True, alpha=0.3)

    fig.suptitle(f"Raw time-domain — {title}", fontsize=11)
    fig.tight_layout()

    # ── Figure 2: individual labeled traces ───────────────────────────────────
    cmap_air = plt.get_cmap("Blues")
    cmap_sam = plt.get_cmap("Oranges")

    fig2, (ax2_l, ax2_r) = plt.subplots(1, 2, figsize=figsize)

    for j, i in enumerate(idx_air):
        color = cmap_air(0.35 + 0.55 * j / max(len(idx_air) - 1, 1))
        y = all_air[i]
        ax2_l.plot(t_air, y, color=color, lw=0.9)
        pk = int(np.argmax(np.abs(y)))
        ax2_l.text(t_air[pk], y[pk], f" {i}", color=color,
                   fontsize=7, va="bottom", ha="center", clip_on=True)

    ax2_l.set_title(f"Reference — {air_path.name}  [{n_air} traces, {len(idx_air)} shown]")
    ax2_l.set_xlabel("Time (ps)");  ax2_l.set_ylabel("Amplitude (a.u.)")
    ax2_l.grid(True, alpha=0.3)

    for j, i in enumerate(idx_sam):
        color = cmap_sam(0.35 + 0.55 * j / max(len(idx_sam) - 1, 1))
        y = all_sam[i]
        ax2_r.plot(t_sam, y, color=color, lw=0.9)
        pk = int(np.argmax(np.abs(y)))
        ax2_r.text(t_sam[pk], y[pk], f" {i}", color=color,
                   fontsize=7, va="bottom", ha="center", clip_on=True)

    ax2_r.set_title(f"Sample — {sam_path.name}  [{n_sam} traces, {len(idx_sam)} shown]")
    ax2_r.set_xlabel("Time (ps)");  ax2_r.set_ylabel("Amplitude (a.u.)")
    ax2_r.grid(True, alpha=0.3)

    fig2.suptitle(f"Individual traces — {title}", fontsize=11)
    fig2.tight_layout()
    print(f"    reference: {n_air} traces   sample: {n_sam} traces")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) >= 2 else str(CONFIG_DEFAULT)
    cfg = load_yaml(config_path)

    load_cfg  = cfg.get("loading",   {})
    align_cfg = cfg.get("alignment", {})
    plot_cfg  = cfg.get("plot",      {})

    correction_factor = load_cfg.get("correction_factor", 1.0)
    max_traces        = load_cfg.get("max_traces")
    n_traces_plot     = plot_cfg.get("n_traces_plot")
    figsize           = tuple(plot_cfg.get("figure_size", [14, 5]))

    ds_kwargs = dict(
        correction_factor=correction_factor,
        max_traces=max_traces,
        align=False,
    )

    # ── Format A: explicit pairs  {reference, sample} ────────────────────────
    if "pairs" in cfg:
        pairs = cfg["pairs"] or []
        if not pairs:
            print("No pairs defined in config.")
            return
        for entry in pairs:
            ref_path = Path(entry["reference"])
            sam_path = Path(entry["sample"])
            name     = sam_path.parent.name
            print(f"\n{'='*60}\n  {name}\n{'='*60}")
            _plot_pair(ref_path, sam_path, _label(name), ds_kwargs, n_traces_plot, figsize)

    # ── Format B: sample dirs with auto-discovered air/point pairs ────────────
    elif "samples" in cfg:
        samples      = cfg["samples"] or []
        air_prefix   = align_cfg.get("air_prefix",   "air")
        point_prefix = align_cfg.get("point_prefix", "point")
        n_points     = align_cfg.get("n_points",     None)

        if not samples:
            print("No samples found in config.")
            return

        for entry in samples:
            name       = entry["name"]
            sample_dir = Path(entry["dir"])
            print(f"\n{'='*60}\n  {name}\n{'='*60}")

            pairs = discover_pairs(sample_dir, air_prefix, point_prefix, n_points)
            if not pairs:
                print("  No air/point pairs found — skipping.")
                continue

            print(f"  {len(pairs)} position(s) found")
            for idx, air_path, point_path in pairs:
                _plot_pair(
                    air_path, point_path,
                    f"{_label(name)}  |  position {idx}",
                    ds_kwargs, n_traces_plot, figsize,
                )

    else:
        print("ERROR: config must have 'pairs' or 'samples' — see file header.")
        return

    plt.show()


if __name__ == "__main__":
    main()
