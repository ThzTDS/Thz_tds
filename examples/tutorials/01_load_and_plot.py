"""
Example 01 — Load data and plot time-domain signals
====================================================
Demonstrates:
  - Loading a config file
  - Creating THZDataset objects for reference and one sample
  - Plotting all raw traces + the average (faint-grey + red)
  - Plotting reference and sample averages side by side
  - Computing and plotting the power spectrum

Usage
-----
    python examples/01_load_and_plot.py my_experiment.yaml
"""

import sys
import matplotlib.pyplot as plt
import thz_tds



def main(config_path: str) -> None:
    # ── Load configuration ──────────────────────────────────────────────
    cfg = thz_tds.load_config(config_path)

    # ── Load reference dataset ──────────────────────────────────────────
    ref = thz_tds.THZDataset(
        path=cfg.paths.reference,
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
        corr_window_ps=cfg.alignment.corr_window_ps,
    )
    print(f"Reference  : {ref.name}  |  {ref.all_y.shape[0]} traces")

    # ── Load first sample ───────────────────────────────────────────────
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
    print(f"Sample     : {sam.name}  |  {sam.all_y.shape[0]} traces")

    # ── Figure 1: raw pulses + average for the reference ────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    thz_tds.viz.plot_raw_and_average(
        ref,
        zoom_xlim=cfg.plot.zoom_xlim_ps,
        ax=ax1,
    )
    fig1.tight_layout()

    # ── Figure 2: reference vs sample averages ──────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    thz_tds.viz.plot_time_domain(
        datasets=[ref, sam],
        zoom_xlim=cfg.plot.zoom_xlim_ps,
        ax=ax2,
    )
    fig2.tight_layout()

    # ── Figure 3: power spectrum ─────────────────────────────────────────
    ref.compute_fft(pad_factor=cfg.fft.pad_factor)
    sam.compute_fft(pad_factor=cfg.fft.pad_factor)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(ref.freq_THz, ref.spectrum_dB, label=f"{ref.name} (ref)", linewidth=2)
    ax3.plot(sam.freq_THz, sam.spectrum_dB, label=sam.name, linewidth=2)
    ax3.set_xlim(0, 2.0)
    ax3.set_xlabel("Frequency [THz]")
    ax3.set_ylabel("Power [dB]")
    ax3.set_title("Normalised Power Spectrum")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    default_config = "/mnt/Code/thz_tds/config/default_config.yaml"
    config_path = sys.argv[1] if len(sys.argv) >= 2 else default_config
    main(config_path)
