"""
plot_time_domain_fft.py
=======================
For each reference/sample pair produces TWO figures:
  1. Time-domain: reference + sample averages with peak-to-peak annotation
  2. FFT amplitude spectrum: reference + sample (log scale)

Supported config formats (auto-detected):

  default_config.yaml  (default)
      pairs:
        - reference: /path/to/air5
          sample:    /path/to/point5
        - ...          (add as many as needed)

  all_samples_fp_batch.yaml
      samples[].dir  — auto-discovers air*/point* sub-folder pairs

Usage
-----
    python examples/plot_time_domain_fft.py
    python examples/plot_time_domain_fft.py config/default_config.yaml
    python examples/plot_time_domain_fft.py config/all_samples_fp_batch.yaml
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

import thz_tds
from thz_tds.spectral import fft_field


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


def _label(name: str) -> str:
    """Keep material and variant, drop thickness: 'PA6_B_3' → 'PA6_B'."""
    parts = name.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _discover_pairs(
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


def _annotate_p2p(ax: plt.Axes, t: np.ndarray, y: np.ndarray, color: str, label: str) -> float:
    p2p = float(y.max() - y.min())
    t_at_max = t[int(np.argmax(y))]
    ax.annotate(
        "",
        xy=(t_at_max, float(y.max())),
        xytext=(t_at_max, float(y.min())),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.6),
    )
    ax.text(
        t_at_max + 0.25,
        (float(y.max()) + float(y.min())) / 2,
        f"{label}: {p2p:.4f}",
        color=color, fontsize=8, va="center",
    )
    return p2p


def _build_ds_kwargs(load_cfg: dict, align_cfg: dict) -> dict:
    return dict(
        correction_factor       = load_cfg.get("correction_factor", 1.0),
        max_traces              = load_cfg.get("max_traces"),
        align                   = align_cfg.get("enabled", True),
        align_method            = align_cfg.get("method", "gaussian_median_integer"),
        corr_window_ps          = align_cfg.get("corr_window_ps", None),
        adaptive_fit_window     = align_cfg.get("adaptive_fit_window", True),
        fit_window_sigma_factor = align_cfg.get("fit_window_sigma_factor", 5.0),
        fallback_fit_window_ps  = align_cfg.get("fallback_fit_window_ps", 15.0),
        min_sigma_ps            = align_cfg.get("min_sigma_ps", 0.05),
        max_sigma_ps            = align_cfg.get("max_sigma_ps", 10.0),
        min_fit_points          = align_cfg.get("min_fit_points", 10),
        max_center_shift_ps     = align_cfg.get("max_center_shift_ps", 5.0),
    )


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_pair(
    sample_name: str,
    pos_label: str,
    ref_ds: thz_tds.THZDataset,
    sam_ds: thz_tds.THZDataset,
    out_dir: Path,
    plot_cfg: dict,
    fft_pad: int = 4,
    fft_max_THz: float = 3.0,
    n_traces_plot: int | None = None,
) -> None:
    """Save two figures — time domain and FFT — for one reference/sample pair."""
    figsize = tuple(plot_cfg.get("figure_size", [12, 5]))
    dpi     = plot_cfg.get("dpi", 150)
    formats = plot_cfg.get("formats", ["png"])

    t_ref = ref_ds.x_ps
    y_ref = ref_ds.y_avg
    t_sam = sam_ds.x_ps
    y_sam = sam_ds.y_avg

    all_ref     = ref_ds.all_y
    all_sam     = sam_ds.all_y
    n_ref_total = all_ref.shape[0]
    n_sam_total = all_sam.shape[0]

    def _idx(n: int) -> list[int]:
        if n_traces_plot is None or n_traces_plot >= n:
            return list(range(n))
        step = max(1, n // n_traces_plot)
        return list(range(0, n, step))

    idx_ref = _idx(n_ref_total)
    idx_sam = _idx(n_sam_total)

    f_ref_Hz, E_ref = fft_field(t_ref, y_ref, pad_factor=fft_pad)
    f_sam_Hz, E_sam = fft_field(t_sam, y_sam, pad_factor=fft_pad)
    f_ref_THz = f_ref_Hz * 1e-12
    f_sam_THz = f_sam_Hz * 1e-12

    title_base = f"{sample_name} — {pos_label}" if pos_label else sample_name
    stem = out_dir / (f"{sample_name}_{pos_label}" if pos_label else sample_name)

    # ── Figure 1: Time domain ─────────────────────────────────────────────────
    fig_td, ax_td = plt.subplots(figsize=figsize)

    for i in idx_ref:
        ax_td.plot(t_ref, all_ref[i], color="tab:blue", lw=0.5, alpha=0.20,
                   label="Ref traces" if i == idx_ref[0] else None)
    for i in idx_sam:
        ax_td.plot(t_sam, all_sam[i], color="tab:orange", lw=0.5, alpha=0.20,
                   label="Sample traces" if i == idx_sam[0] else None)

    ax_td.plot(t_ref, y_ref, color="tab:blue",   lw=2.0, label="Reference (avg)")
    ax_td.plot(t_sam, y_sam, color="tab:orange", lw=2.0, label="Sample (avg)")

    p2p_r = _annotate_p2p(ax_td, t_ref, y_ref, color="tab:blue",   label="Ref p2p")
    p2p_s = _annotate_p2p(ax_td, t_sam, y_sam, color="tab:orange", label="Sam p2p")

    ax_td.set_title(
        f"{title_base}\n"
        f"Ref p2p = {p2p_r:.4f}   Sam p2p = {p2p_s:.4f}   "
        f"({n_ref_total} ref / {n_sam_total} sam traces)"
    )
    ax_td.set_xlabel("Time (ps)")
    ax_td.set_ylabel("Amplitude (a.u.)")
    ax_td.legend(fontsize=8, loc="upper right")
    ax_td.grid(True, alpha=0.3)
    fig_td.tight_layout()

    for fmt in formats:
        fig_td.savefig(f"{stem}_td.{fmt}", dpi=dpi)
    print(f"  Saved TD : {stem}_td.{formats[0]}")

    # ── Figure 2: FFT amplitude spectrum ─────────────────────────────────────
    fig_fft, ax_fft = plt.subplots(figsize=figsize)

    ax_fft.plot(f_ref_THz, np.abs(E_ref), color="tab:blue",   lw=1.8, label="Reference (avg)")
    ax_fft.plot(f_sam_THz, np.abs(E_sam), color="tab:orange", lw=1.8, label="Sample (avg)")
    ax_fft.set_title(f"Amplitude spectrum — {title_base}")
    ax_fft.set_xlabel("Frequency (THz)")
    ax_fft.set_ylabel("|E(f)|  (a.u.)")
    ax_fft.set_xlim(0, fft_max_THz)
    ax_fft.set_yscale("log")
    ax_fft.legend(fontsize=8, loc="upper right")
    ax_fft.grid(True, alpha=0.3, which="both")
    fig_fft.tight_layout()

    for fmt in formats:
        fig_fft.savefig(f"{stem}_fft.{fmt}", dpi=dpi)
    print(f"  Saved FFT: {stem}_fft.{formats[0]}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) >= 2 else str(CONFIG_DEFAULT)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw       = _load_yaml(config_path)
    load_cfg  = raw.get("loading",   {})
    align_cfg = raw.get("alignment", {})
    plot_cfg  = raw.get("plot",      {})
    fft_pad   = raw.get("fft", {}).get("pad_factor", 4)
    out_root  = Path(raw.get("output", "output"))
    kw        = _build_ds_kwargs(load_cfg, align_cfg)
    n_traces_plot = plot_cfg.get("n_traces_plot")

    out_dir = out_root / f"td_fft_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config : {config_path}")
    print(f"Output : {out_dir}\n")

    # ── Format A: pairs list  ─────────────────────────────────────────────────
    if "pairs" in raw:
        pairs = raw["pairs"]
        if not pairs:
            print("No pairs defined in config — nothing to do.")
            return

        ref_cache: dict[str, thz_tds.THZDataset] = {}
        for entry in pairs:
            ref_path = entry["reference"]
            sam_path = entry["sample"]

            if ref_path not in ref_cache:
                print(f"Loading reference: {ref_path}")
                ref_cache[ref_path] = thz_tds.THZDataset(path=ref_path, **kw)
            ref_ds = ref_cache[ref_path]

            # Label from the parent folder, trimmed to material type (e.g. ABS_3 → ABS)
            name = _label(Path(sam_path).parent.name)
            print(f"\n{'='*60}\n  {name}\n{'='*60}")
            print(f"  sample : {sam_path}")
            sam_ds = thz_tds.THZDataset(path=sam_path, **kw)
            plot_pair(
                name, "",
                ref_ds, sam_ds,
                out_dir, plot_cfg,
                fft_pad=fft_pad,
                n_traces_plot=n_traces_plot,
            )

    # ── Format B: batch  samples[].dir  ──────────────────────────────────────
    elif "samples" in raw:
        samples      = raw["samples"]
        air_prefix   = align_cfg.get("air_prefix",   "air")
        point_prefix = align_cfg.get("point_prefix", "point")
        n_points     = align_cfg.get("n_points",     None)

        if not samples:
            print("No samples found in config — nothing to do.")
            return

        for entry in samples:
            name       = entry["name"]
            sample_dir = Path(entry["dir"])
            print(f"{'='*60}\n  {name}\n{'='*60}")

            pairs = _discover_pairs(sample_dir, air_prefix, point_prefix, n_points)
            if not pairs:
                print("  No air/point pairs found — skipping.\n")
                continue

            print(f"  {len(pairs)} position(s) found")
            for idx, air_path, point_path in pairs:
                pos_name = f"{point_prefix}{idx}"
                print(f"  {pos_name}: loading {air_path.name} + {point_path.name} ...")
                ref_ds = thz_tds.THZDataset(path=air_path,   **kw)
                sam_ds = thz_tds.THZDataset(path=point_path, **kw)
                plot_pair(
                    _label(name), pos_name,
                    ref_ds, sam_ds,
                    out_dir, plot_cfg,
                    fft_pad=fft_pad,
                    n_traces_plot=n_traces_plot,
                )
            print()

    else:
        print("ERROR: config must contain either 'pairs' or 'samples' — see docstring.")
        return

    plt.show()
    print(f"\nDone.  All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
