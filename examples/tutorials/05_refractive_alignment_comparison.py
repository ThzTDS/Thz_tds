"""
Example 05 — Refractive index comparison across alignment methods
=================================================================
Demonstrates:
  - Loading the same reference/sample with three alignment methods:
      * "minimum"               — align to the trace minimum
      * "correlation"           — integer-sample cross-correlation
      * "correlation_subsample" — cross-correlation with sub-sample refinement
  - Computing n(f) and α(f) for each alignment method
  - Overlaying all three results on the same axes for direct comparison

Usage
-----
    python examples/05_refractive_alignment_comparison.py my_experiment.yaml
"""

import sys
import matplotlib.pyplot as plt
import thz_tds


ALIGN_METHODS = [
    "minimum",
    "correlation",
    "correlation_subsample",
]


def load_pair(cfg, align_method: str):
    """Return (ref_dataset, sam_dataset) loaded with the given alignment method."""
    ref = thz_tds.THZDataset(
        path=cfg.paths.reference,
        correction_factor=cfg.loading.correction_factor,
        align=True,
        align_method=align_method,
        corr_window_ps=cfg.alignment.corr_window_ps,
    )

    if cfg.paths.samples:
        sam = thz_tds.THZDataset(
            path=cfg.paths.samples[0],
            correction_factor=cfg.loading.correction_factor,
            align=True,
            align_method=align_method,
            corr_window_ps=cfg.alignment.corr_window_ps,
        )
    else:
        samples = thz_tds.scan_samples(
            base_dir=cfg.paths.base_dir,
            correction_factor=cfg.loading.correction_factor,
            align=True,
            align_method=align_method,
            corr_window_ps=cfg.alignment.corr_window_ps,
        )
        if not samples:
            raise RuntimeError(
                "No samples found. Check paths.samples or paths.base_dir in your config."
            )
        sam = samples[0]

    return ref, sam


def main(config_path: str) -> None:
    cfg = thz_tds.load_config(config_path)
    ri = cfg.refractive_index

    print(f"Thickness : {ri.thickness_m * 1e3:.3f} mm")
    print(f"Band      : {ri.f_low_THz} – {ri.f_high_THz} THz")
    print()

    # ── Compute refractive index for every alignment method ─────────────
    results = []
    for method in ALIGN_METHODS:
        print(f"Loading with align_method='{method}' ...")
        ref, sam = load_pair(cfg, method)
        print(f"  Reference : {ref.name}  |  {ref.all_y.shape[0]} traces")
        print(f"  Sample    : {sam.name}  |  {sam.all_y.shape[0]} traces")

        f_THz, n_f, _T, _phi, alpha_cm = thz_tds.compute_refractive_index(
            ref_ds=ref,
            sam_ds=sam,
            thickness_m=ri.thickness_m,
            f_low_THz=ri.f_low_THz,
            f_high_THz=ri.f_high_THz,
        )
        results.append(
            {
                "name": method,
                "f_THz": f_THz,
                "n_f": n_f,
                "alpha_cm": alpha_cm,
                "ref": ref,
                "sam": sam,
            }
        )
        print(f"  Done.\n")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    thz_tds.viz.plot_refractive_index(
        results=results,
        f_min_THz=ri.f_low_THz,
        f_max_THz=ri.f_high_THz,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Refractive Index — alignment method comparison\n"
        f"Sample: {results[0]['name'].split('/')[0]}  |  "
        f"band {ri.f_low_THz}–{ri.f_high_THz} THz"
    )

    thz_tds.viz.plot_absorption(
        results=results,
        f_min_THz=ri.f_low_THz,
        f_max_THz=ri.f_high_THz,
        ax=axes[1],
    )
    axes[1].set_title("Absorption Coefficient — alignment method comparison")

    # Use the sample folder name as the figure title
    sam_name = cfg.paths.samples[0].rstrip("/\\").split("/")[-1] if cfg.paths.samples else "sample"
    fig.suptitle(
        f"Optical constants — {sam_name} | alignment comparison",
        fontsize=13,
    )
    fig.tight_layout()

    thz_tds.viz.save_figure(
        fig,
        cfg.paths.output,
        stem=f"refractive_alignment_comparison_{sam_name}",
        dpi=cfg.plot.dpi,
        formats=cfg.plot.formats,
    )

    import numpy as np

    # ── Figure: reference average trace ──────────────────────────────────
    ref0 = results[0]["ref"]
    fig_ref, ax_ref = plt.subplots(figsize=(10, 4))
    ax_ref.plot(ref0.x_ps, ref0.y_avg, color="tab:blue", lw=1.5)
    ax_ref.set_xlabel("Time (ps)")
    ax_ref.set_ylabel("Amplitude [nA]")
    ax_ref.set_title(f"Reference trace — {ref0.name}")
    ax_ref.grid(True, alpha=0.3)
    fig_ref.tight_layout()

    # ── One figure per alignment method: all aligned traces + average ─────
    for r in results:
        method = r["name"]
        sam_ds = r["sam"]
        n_traces = sam_ds.all_y.shape[0]
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / max(n_traces - 1, 1)) for i in range(n_traces)]

        fig_m, axes_m = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Top: individual aligned traces + average
        for i, y in enumerate(sam_ds.all_y):
            axes_m[0].plot(sam_ds.x_ps, y, color=colors[i], lw=0.8, alpha=0.5,
                           label=f"Trace {i+1}" if n_traces <= 10 else None)
        axes_m[0].plot(sam_ds.x_ps, sam_ds.y_avg, color="black", lw=2,
                       label="Average", zorder=10)
        axes_m[0].set_ylabel("Amplitude [nA]")
        axes_m[0].set_title(f"Aligned traces — method: {method}")
        axes_m[0].legend(fontsize=8, ncol=2)
        axes_m[0].grid(True, alpha=0.3)

        # Bottom: per-trace residual from average
        for i, y in enumerate(sam_ds.all_y):
            axes_m[1].plot(sam_ds.x_ps, y - sam_ds.y_avg,
                           color=colors[i], lw=0.8, alpha=0.6,
                           label=f"Trace {i+1}" if n_traces <= 10 else None)
        axes_m[1].axhline(0, color="black", lw=0.8)
        rms = float(np.sqrt(np.mean((sam_ds.all_y - sam_ds.y_avg[None, :]) ** 2)))
        axes_m[1].set_ylabel("ΔAmplitude [nA]")
        axes_m[1].set_xlabel("Time (ps)")
        axes_m[1].set_title(f"Residuals from average  (RMS = {rms:.4e})")
        if n_traces <= 10:
            axes_m[1].legend(fontsize=8, ncol=2)
        axes_m[1].grid(True, alpha=0.3)

        fig_m.suptitle(
            f"Sample: {sam_ds.name}  |  alignment: {method}  |  {n_traces} traces",
            fontsize=11,
        )
        fig_m.tight_layout()

    plt.show()


if __name__ == "__main__":
    default_config = "/mnt/Code/thz_tds/config/default_config.yaml" \
    ""
    config_path = sys.argv[1] if len(sys.argv) >= 2 else default_config
    main(config_path)
