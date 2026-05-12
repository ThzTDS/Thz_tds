"""
Example 04 — Multi-sample spectral survey
==========================================
Demonstrates:
  - Loading all samples from base_dir automatically
  - Running the full spectral survey (power spectra + differences)
  - Generating and saving the standard 4-panel survey figure

Usage
-----
    python examples/04_multi_sample_survey.py my_experiment.yaml
"""

import sys
import matplotlib.pyplot as plt
import thz_tds


def main(config_path: str) -> None:
    cfg = thz_tds.load_config(config_path)

    print(f"Reference : {cfg.paths.reference}")
    print(f"Base dir  : {cfg.paths.base_dir}")
    print("Running spectral survey …")

    # ── Run the batch workflow ──────────────────────────────────────────
    results = thz_tds.run_spectral_survey(cfg)

    n_samples = sum(1 for r in results if not r.get("is_reference"))
    print(f"Loaded {n_samples} sample(s).")
    for r in results:
        n = r["all_y"].shape[0] if r.get("all_y") is not None else "?"
        tag = " [ref]" if r.get("is_reference") else ""
        print(f"  {r['name']}{tag}  ({n} traces)")

    # ── Generate the standard 4-panel survey figure ─────────────────────
    fig = thz_tds.viz.plot_survey_figure(results, cfg)
    fig.suptitle("THz-TDS Multi-Sample Survey", fontsize=14, y=1.01)

    saved = thz_tds.viz.save_figure(
        fig,
        output_dir=cfg.paths.output,
        stem="survey",
        dpi=cfg.plot.dpi,
        formats=cfg.plot.formats,
    )
    for p in saved:
        print(f"Saved: {p}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Provide a config file path as the first argument.")
        sys.exit(1)
    main(sys.argv[1])
