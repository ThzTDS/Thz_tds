"""
Example 03 — Fabry-Perot echo removal
======================================
Demonstrates:
  - Running the Liu model grid search for one sample
  - Plotting the time-domain fit (measured vs simulated vs cleaned)
  - Extracting optical constants from the FP-cleaned signal

Usage
-----
    python examples/03_fabry_perot_removal.py my_experiment.yaml

Note: the thickness grid search can take several minutes depending on the
step size and echo count.  Reduce thickness_step_m in your config (e.g. to
5e-6) for a faster run during testing.
"""

import sys
import matplotlib.pyplot as plt
import thz_tds
from thz_tds.fabry_perot import THzTrace


def main(config_path: str) -> None:
    cfg = thz_tds.load_config(config_path)

    # ── Load reference and first sample ────────────────────────────────
    ref = thz_tds.THZDataset(
        path=cfg.paths.reference,
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
    )

    samples = thz_tds.scan_samples(
        base_dir=cfg.paths.base_dir,
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
    )
    if not samples:
        print(f"No samples found in {cfg.paths.base_dir!r}.")
        return

    sam = samples[0]
    fp = cfg.fabry_perot
    print(f"Sample     : {sam.name}")
    print(f"Grid       : {fp.thickness_min_m*1e3:.3f} – {fp.thickness_max_m*1e3:.3f} mm "
          f"step {fp.thickness_step_m*1e6:.1f} µm")
    print(f"Echo count : {fp.echo_count}")
    print("Running Liu grid search …  (this may take a few minutes)")

    # ── Run FP removal ──────────────────────────────────────────────────
    ref_trace = THzTrace(time_ps=ref.x_ps, y=ref.y_avg)
    sam_trace = THzTrace(time_ps=sam.x_ps, y=sam.y_avg)

    result = thz_tds.remove_fabry_perot(
        ref=ref_trace,
        sam=sam_trace,
        thickness_min_m=fp.thickness_min_m,
        thickness_max_m=fp.thickness_max_m,
        thickness_step_m=fp.thickness_step_m,
        echo_count=fp.echo_count,
        f_min_THz=fp.f_min_THz,
        f_max_THz=fp.f_max_THz,
    )

    fit = result["fit"]
    spec = result["spectrum"]

    print(f"\nBest fit:")
    print(f"  thickness : {fit.thickness_m * 1e3:.4f} mm")
    print(f"  nav       : {fit.nav:.4f}")
    print(f"  gamma     : {fit.gamma:.4f}")
    print(f"  delta     : {fit.delta:.4f}")
    print(f"  R²        : {fit.r2:.4f}")

    # ── Plot time-domain fit ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    thz_tds.viz.plot_fp_fit(ref_trace, sam_trace, fit, ax=ax1)
    fig1.tight_layout()
    thz_tds.viz.save_figure(fig1, cfg.paths.output, stem=f"fp_fit_{sam.name}",
                             dpi=cfg.plot.dpi, formats=cfg.plot.formats)

    # ── Plot optical constants after FP removal ─────────────────────────
    fig2, axes = plt.subplots(2, 1, figsize=(9, 8))
    thz_tds.viz.plot_refractive_index(
        results=[{"name": sam.name, "f_THz": spec["f_THz"], "n_f": spec["n"]}],
        f_min_THz=fp.f_min_THz,
        f_max_THz=fp.f_max_THz,
        ax=axes[0],
    )
    thz_tds.viz.plot_absorption(
        results=[{"name": sam.name, "f_THz": spec["f_THz"], "alpha_cm": spec["alpha_1_per_cm"]}],
        f_min_THz=fp.f_min_THz,
        f_max_THz=fp.f_max_THz,
        ax=axes[1],
    )
    fig2.suptitle(f"Optical constants after FP removal — {sam.name}", fontsize=13)
    fig2.tight_layout()
    thz_tds.viz.save_figure(fig2, cfg.paths.output, stem=f"fp_optics_{sam.name}",
                             dpi=cfg.plot.dpi, formats=cfg.plot.formats)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Provide a config file path as the first argument.")
        sys.exit(1)
    main(sys.argv[1])
