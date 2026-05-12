"""
Example 02 — Refractive index and absorption coefficient
=========================================================
Demonstrates:
  - Computing n(f) and α(f) for a single sample
  - Plotting the results

Usage
-----
    python examples/02_refractive_index.py my_experiment.yaml
"""

import sys
import matplotlib.pyplot as plt
import thz_tds



def main(config_path: str) -> None:
    cfg = thz_tds.load_config(config_path)

    # ── Load reference and first sample ────────────────────────────────
    ref = thz_tds.THZDataset(
        path=cfg.paths.reference,
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
        corr_window_ps=cfg.alignment.corr_window_ps,
    )

    if cfg.paths.samples:
        samples = [
            thz_tds.THZDataset(
                path=p,
                correction_factor=cfg.loading.correction_factor,
                align=cfg.alignment.enabled,
                align_method=cfg.alignment.method,
                corr_window_ps=cfg.alignment.corr_window_ps,
            )
            for p in cfg.paths.samples
        ]
    else:
        samples = thz_tds.scan_samples(
            base_dir=cfg.paths.base_dir,
            correction_factor=cfg.loading.correction_factor,
            align=cfg.alignment.enabled,
            align_method=cfg.alignment.method,
            corr_window_ps=cfg.alignment.corr_window_ps,
        )
    if not samples:
        print(f"No samples found. Check paths.samples or paths.base_dir in your config.")
        return

    sam = samples[0]
    ri = cfg.refractive_index
    print(f"Sample: {sam.name}")
    print(f"Thickness: {ri.thickness_m * 1e3:.3f} mm")
    print(f"Frequency band: {ri.f_low_THz} – {ri.f_high_THz} THz")

    # ── Compute optical constants ───────────────────────────────────────
    f_THz, n_f, T, phi_full, alpha_cm = thz_tds.compute_refractive_index(
        ref_ds=ref,
        sam_ds=sam,
        thickness_m=ri.thickness_m,
        f_low_THz=ri.f_low_THz,
        f_high_THz=ri.f_high_THz,
    )

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    thz_tds.viz.plot_refractive_index(
        results=[{"name": sam.name, "f_THz": f_THz, "n_f": n_f}],
        f_min_THz=ri.f_low_THz,
        f_max_THz=ri.f_high_THz,
        ax=axes[0],
    )

    thz_tds.viz.plot_absorption(
        results=[{"name": sam.name, "f_THz": f_THz, "alpha_cm": alpha_cm}],
        f_min_THz=ri.f_low_THz,
        f_max_THz=ri.f_high_THz,
        ax=axes[1],
    )

    fig.suptitle(f"Optical constants — {sam.name}", fontsize=13)
    fig.tight_layout()

    thz_tds.viz.save_figure(fig, cfg.paths.output, stem=f"refractive_{sam.name}",
                             dpi=cfg.plot.dpi, formats=cfg.plot.formats)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        import os
        default = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
        main(os.path.abspath(default))
