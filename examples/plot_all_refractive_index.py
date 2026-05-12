"""
plot_all_refractive_index.py
============================
Loads all samples from all_samples_config.yaml, computes the refractive
index and absorption coefficient for each one, then plots them together
in a single figure with distinct colours and labels.

Usage
-----
    python plot_all_refractive_index.py
    python plot_all_refractive_index.py /path/to/all_samples_config.yaml
"""

import sys

import yaml
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import thz_tds


CONFIG_DEFAULT = Path(__file__).parent.parent / "config" / "all_samples_config.yaml"


def load_yaml_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml_config(config_path)

    align_cfg = cfg["alignment"]
    load_cfg = cfg["loading"]
    ri_cfg = cfg["refractive_index"]
    plot_cfg = cfg["plot"]
    output_dir = cfg["output"]

    samples_cfg = cfg["samples"]

    # Skip samples with unknown thickness
    valid = [s for s in samples_cfg if s["thickness_m"] is not None]
    skipped = [s["name"] for s in samples_cfg if s["thickness_m"] is None]
    if skipped:
        print(f"Skipping (no thickness): {', '.join(skipped)}")

    # Assign one colour per sample from a qualitative colormap
    colors = cm.tab20(np.linspace(0, 1, len(valid)))

    results_n = []      # for plot_refractive_index
    results_a = []      # for plot_absorption
    summary = []        # for end-of-run table

    align_tag = align_cfg["method"] if align_cfg["enabled"] else "no alignment"

    for entry, color in tqdm(zip(valid, colors), total=len(valid), desc="Samples", unit="sample"):
        name = entry["name"]

        ref = thz_tds.THZDataset(
            path=entry["reference"],
            correction_factor=load_cfg["correction_factor"],
            align=align_cfg["enabled"],
            align_method=align_cfg["method"],
            corr_window_ps=align_cfg["corr_window_ps"],
        )

        sam = thz_tds.THZDataset(
            path=entry["sample_folder"],
            correction_factor=load_cfg["correction_factor"],
            align=align_cfg["enabled"],
            align_method=align_cfg["method"],
            corr_window_ps=align_cfg["corr_window_ps"],
        )

        f_THz, n_f, _, _, alpha_cm = thz_tds.compute_refractive_index(
            ref_ds=ref,
            sam_ds=sam,
            thickness_m=entry["thickness_m"],
            f_low_THz=ri_cfg["f_low_THz"],
            f_high_THz=ri_cfg["f_high_THz"],
        )

        idx = np.argmin(np.abs(f_THz - 1.0))
        n_at_1 = n_f[idx]

        thickness_mm = entry["thickness_m"] * 1e3
        results_n.append({"name": name, "f_THz": f_THz, "n_f": n_f, "color": color})
        results_a.append({"name": name, "f_THz": f_THz, "alpha_cm": alpha_cm, "color": color})
        summary.append((name, thickness_mm, n_at_1))
        tqdm.write(f"  {name}: thickness = {thickness_mm:.3f} mm  n(1.0 THz) ≈ {n_at_1:.3f}  [{align_tag}]")

    # ── Summary table ──────────────────────────────────────────────────────
    col_w = max(len(name) for name, _, _ in summary)
    print(f"\n{'Sample':<{col_w}}  {'thickness (mm)':>14}  {'n(1.0 THz)':>10}  alignment")
    print("-" * (col_w + 42))
    for name, t_mm, n_val in summary:
        print(f"{name:<{col_w}}  {t_mm:>14.3f}  {n_val:>10.3f}  {align_tag}")
    print()

    f_lo = ri_cfg["f_low_THz"]
    f_hi = ri_cfg["f_high_THz"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def add_inline_labels(ax, results, y_key):
        """Place each sample name at the right end of its curve."""
        for r in results:
            mask = (r["f_THz"] >= f_lo) & (r["f_THz"] <= f_hi)
            f = r["f_THz"][mask]
            y = r[y_key][mask]
            if len(f) == 0:
                continue
            ax.annotate(
                r["name"],
                xy=(f[-1], y[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=r["color"],
                fontsize=7,
                va="center",
                clip_on=False,
            )

    def style_ax(ax, ylabel, title):
        ax.set_xlabel("Frequency [THz]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        # extra right margin so inline labels aren't clipped
        ax.margins(x=0.05)

    # ── Figure 1: Refractive index ──────────────────────────────────────────
    fig_n, ax_n = plt.subplots(figsize=plot_cfg["figure_size"])

    for r in results_n:
        mask = (r["f_THz"] >= f_lo) & (r["f_THz"] <= f_hi)
        ax_n.plot(r["f_THz"][mask], r["n_f"][mask],
                  color=r["color"], linewidth=2)

    add_inline_labels(ax_n, results_n, "n_f")
    style_ax(ax_n, "Refractive index  n(f)", "Refractive Index — all samples")
    fig_n.tight_layout()

    for p in thz_tds.viz.save_figure(fig_n, output_dir=output_dir,
                                      stem="all_samples_refractive_index",
                                      dpi=plot_cfg["dpi"], formats=plot_cfg["formats"]):
        print(f"Saved: {p}")

    # ── Figure 2: Refractive index — zoomed [1.6, 1.7] ─────────────────────
    fig_nz, ax_nz = plt.subplots(figsize=plot_cfg["figure_size"])

    for r in results_n:
        mask = (r["f_THz"] >= f_lo) & (r["f_THz"] <= f_hi)
        ax_nz.plot(r["f_THz"][mask], r["n_f"][mask],
                   color=r["color"], linewidth=2)

    add_inline_labels(ax_nz, results_n, "n_f")
    style_ax(ax_nz, "Refractive index  n(f)", "Refractive Index — all samples  [zoom 1.6 – 1.7]")
    ax_nz.set_ylim(1.6, 1.7)
    fig_nz.tight_layout()

    for p in thz_tds.viz.save_figure(fig_nz, output_dir=output_dir,
                                      stem="all_samples_refractive_index_zoom",
                                      dpi=plot_cfg["dpi"], formats=plot_cfg["formats"]):
        print(f"Saved: {p}")

    # ── Figure 3: Absorption coefficient ───────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=plot_cfg["figure_size"])

    for r in results_a:
        mask = (r["f_THz"] >= f_lo) & (r["f_THz"] <= f_hi)
        ax_a.plot(r["f_THz"][mask], r["alpha_cm"][mask],
                  color=r["color"], linewidth=2)

    add_inline_labels(ax_a, results_a, "alpha_cm")
    style_ax(ax_a, "Absorption coefficient  α [cm⁻¹]", "Absorption Coefficient — all samples")
    fig_a.tight_layout()

    for p in thz_tds.viz.save_figure(fig_a, output_dir=output_dir,
                                      stem="all_samples_absorption",
                                      dpi=plot_cfg["dpi"], formats=plot_cfg["formats"]):
        print(f"Saved: {p}")

    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
