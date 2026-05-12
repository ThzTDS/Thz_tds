"""
Example 07 — Fabry–Pérot-aware refractive index extraction
===========================================================
Demonstrates:
  - Computing n(f), κ(f), and α(f) using the full FP slab model
  - Comparing FP-corrected results against the simple (no-FP) initial guess
  - Reconstructing and comparing the modelled sample time-domain trace

Usage
-----
    python examples/07_refractive_index_fp.py my_experiment.yaml

The YAML config must have a ``refractive_index`` section with at least:
    thickness_m, f_low_THz, f_high_THz
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import thz_tds
from thz_tds.refractive_fp import alpha_cm_to_kappa, compute_refractive_index_fp
from thz_tds.spectral import ensure_common_time_axis

_C0 = 2.99792458e8


def _load_pair(cfg, corr_window_ps):
    """Load reference + first sample with the given corr_window_ps."""
    _kw = dict(
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
        corr_window_ps=corr_window_ps,
    )
    ref = thz_tds.THZDataset(path=cfg.paths.reference, **_kw)
    if cfg.paths.samples:
        sam = thz_tds.THZDataset(path=cfg.paths.samples[0], **_kw)
    else:
        sams = thz_tds.scan_samples(base_dir=cfg.paths.base_dir, **_kw)
        if not sams:
            raise RuntimeError("No samples found. Check paths in your config.")
        sam = sams[0] if not isinstance(sams[0], str) else thz_tds.THZDataset(
            path=sams[0], **_kw,
        )
    return ref, sam


def _window_label(w):
    return "full" if w is None else f"{w:g} ps"


def main(config_path: str) -> None:
    cfg = thz_tds.load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ri = cfg.refractive_index

    # Derive sample label from path
    _first_path = (cfg.paths.samples[0] if cfg.paths.samples else cfg.paths.base_dir or "")
    _sp = _first_path.rstrip("/\\")
    sam_label = f"{os.path.basename(os.path.dirname(_sp))} / {os.path.basename(_sp)}"

    corr_windows = cfg.alignment.corr_window_sweep or [cfg.alignment.corr_window_ps]

    print(f"Sample   : {sam_label}")
    print(f"Thickness: {ri.thickness_m * 1e3:.3f} mm")
    print(f"Fit band : {ri.f_low_THz} – {ri.f_high_THz} THz")
    print(f"Windows  : {corr_windows}\n")

    f_low  = ri.f_low_THz
    f_high = ri.f_high_THz

    sweep_results = []

    for win in tqdm(corr_windows, desc="corr_window_ps sweep"):
        label = _window_label(win)
        tqdm.write(f"\n── window = {label} ──")

        ref, sam = _load_pair(cfg, win)

        t_ref_peak = ref.x_ps[np.argmax(np.abs(ref.y_avg))]
        t_sam_peak = sam.x_ps[np.argmax(np.abs(sam.y_avg))]
        dt_peak_ps = t_sam_peak - t_ref_peak
        n_peak = 1.0 + (dt_peak_ps * 1e-12 * _C0) / ri.thickness_m
        tqdm.write(f"  n_peak = {n_peak:.4f}  (Δt = {dt_peak_ps:.4f} ps)")

        result = compute_refractive_index_fp(
            ref_ds=ref,
            sam_ds=sam,
            thickness_m=ri.thickness_m,
            f_low_THz=f_low,
            f_high_THz=f_high,
        )

        band = (result.f_THz >= f_low) & (result.f_THz <= f_high)
        tqdm.write(
            f"  n_fp    = {result.n_fp[band].mean():.4f}"
            f"  α_fp    = {result.alpha_cm_fp[band].mean():.2f} cm⁻¹"
        )
        tqdm.write(
            f"  n_nofp  = {result.n_init[band].mean():.4f}"
            f"  α_nofp  = {result.alpha_init_cm[band].mean():.2f} cm⁻¹"
        )

        _, y_ref_c, y_sam_c = ensure_common_time_axis(
            ref.x_ps, ref.y_avg, sam.x_ps, sam.y_avg
        )
        time_ps = ref.x_ps

        sweep_results.append(dict(
            win=win, label=label,
            ref=ref, sam=sam,
            result=result,
            band=band,
            y_ref_c=y_ref_c, y_sam_c=y_sam_c,
            time_ps=time_ps,
            n_nofp=result.n_init,
            alpha_nofp=result.alpha_init_cm,
        ))

    title_suffix = f" — {sam_label}"

    # ── Comparison figure: n(f) for all windows ──────────────────────────
    tqdm.write("\nBuilding figures...")
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sweep_results))]

    fig_n, ax_n = plt.subplots(figsize=(9, 5))
    fig_a, ax_a = plt.subplots(figsize=(9, 5))

    for i, r in enumerate(sweep_results):
        res   = r["result"]
        band  = r["band"]
        lbl   = r["label"]
        col   = colors[i]
        # FP-corrected — solid
        ax_n.plot(res.f_THz[band], res.n_fp[band],        color=col, lw=1.8,
                  label=f"{lbl}  FP")
        ax_a.plot(res.f_THz[band], res.alpha_cm_fp[band], color=col, lw=1.8,
                  label=f"{lbl}  FP")
        # No-FP — dashed, same colour
        ax_n.plot(res.f_THz[band], r["n_nofp"][band],     color=col, lw=1.2, ls="--", alpha=0.7,
                  label=f"{lbl}  no-FP")
        ax_a.plot(res.f_THz[band], r["alpha_nofp"][band], color=col, lw=1.2, ls="--", alpha=0.7,
                  label=f"{lbl}  no-FP")

    for ax, title, ylabel in [
        (ax_n, "n(f) — FP vs no-FP, corr_window_ps sweep", "Refractive index  n"),
        (ax_a, "α(f) — FP vs no-FP, corr_window_ps sweep", "Absorption coefficient  α (cm⁻¹)"),
    ]:
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title + title_suffix)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig_n.tight_layout()
    fig_a.tight_layout()

    # ── Detail figures for each window ───────────────────────────────────
    detail_figs: list[tuple] = []

    for r in sweep_results:
        res        = r["result"]
        band       = r["band"]
        lbl        = r["label"]
        time_ps    = r["time_ps"]
        y_ref_c    = r["y_ref_c"]
        y_sam_c    = r["y_sam_c"]
        y_sam_mod  = res.y_sam_model
        n_len      = len(time_ps)
        ref_peak   = np.max(np.abs(y_ref_c))
        n_avg      = res.n_fp[band].mean()
        echo_delay = 2.0 * n_avg * ri.thickness_m / _C0 * 1e12

        # Build no-FP model trace for residual
        f_hz, E_ref = thz_tds.fft_field(time_ps, y_ref_c, pad_factor=1)
        f_hz  = f_hz[1:];  E_ref = E_ref[1:]
        omega = 2.0 * np.pi * f_hz
        kappa_i   = alpha_cm_to_kappa(f_hz, res.alpha_init_cm)
        n_tilde_i = res.n_init - 1j * kappa_i
        H_nofp    = ((2.0 / (1.0 + n_tilde_i)) *
                     (2.0 * n_tilde_i / (1.0 + n_tilde_i)) *
                     np.exp(-1j * omega * (n_tilde_i - 1.0) * ri.thickness_m / _C0))
        y_nofp = np.fft.irfft(np.concatenate(([0j], H_nofp * E_ref)), n=n_len)

        # Normalized overlay + echo markers
        fig_ov, ax_ov = plt.subplots(figsize=(10, 5))
        ax_ov.plot(time_ps, y_ref_c / ref_peak,       label="Reference",     color="tab:blue",   lw=1.5)
        ax_ov.plot(time_ps, y_sam_c / ref_peak,       label="Sample (meas)", color="tab:orange", lw=1.5)
        ax_ov.plot(time_ps, y_sam_mod[:n_len] / ref_peak, label="FP model",  color="tab:green",  lw=1.5, ls="--")
        t_main = time_ps[np.argmax(np.abs(y_sam_c))]
        for k in range(1, 4):
            ax_ov.axvline(t_main + k * echo_delay, color="red", ls=":", alpha=0.5,
                          label=f"Echo {k}" if k == 1 else None)
        ax_ov.set_xlabel("Time (ps)")
        ax_ov.set_ylabel("Amplitude (normalised) [nA]")
        ax_ov.set_title(f"Echo overlay — window = {lbl}" + title_suffix)
        ax_ov.legend()
        ax_ov.grid(True, alpha=0.3)
        fig_ov.tight_layout()

        # Residual (measured − no-FP model)
        fig_res, axes_r = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes_r[0].plot(time_ps, y_sam_c,              label="Measured",   color="tab:gray",  lw=1.5)
        axes_r[0].plot(time_ps, y_nofp,               label="No-FP",      color="tab:blue",  lw=1.5, ls="--")
        axes_r[0].plot(time_ps, y_sam_mod[:n_len],    label="FP model",   color="tab:green", lw=1.5, ls="--")
        axes_r[0].set_ylabel("Amplitude [nA]")
        axes_r[0].set_title(f"Models — window = {lbl}" + title_suffix)
        axes_r[0].legend()
        axes_r[0].grid(True, alpha=0.3)

        resid = y_sam_c - y_nofp
        axes_r[1].plot(time_ps, resid, color="tab:red", lw=1.2, label="Residual (FP echo content)")
        axes_r[1].axhline(0, color="k", lw=0.8)
        axes_r[1].set_xlabel("Time (ps)")
        axes_r[1].set_ylabel("ΔAmplitude [nA]")
        axes_r[1].set_title(f"Residual: measured − no-FP  (window = {lbl})")
        axes_r[1].legend()
        axes_r[1].grid(True, alpha=0.3)
        fig_res.tight_layout()

        # n and α: FP vs no-FP for this window
        fig_ri, axes_ri = plt.subplots(1, 2, figsize=(12, 5))
        axes_ri[0].plot(res.f_THz[band], res.n_fp[band],       color="tab:blue",   lw=2,   label="FP-corrected")
        axes_ri[0].plot(res.f_THz[band], r["n_nofp"][band],    color="tab:gray",   lw=1.5, ls="--", label="No-FP")
        axes_ri[0].set_xlabel("Frequency (THz)")
        axes_ri[0].set_ylabel("Refractive index  n")
        axes_ri[0].set_title(f"n(f) — window = {lbl}")
        axes_ri[0].legend()
        axes_ri[0].grid(True, alpha=0.3)

        axes_ri[1].plot(res.f_THz[band], res.alpha_cm_fp[band],  color="tab:red",   lw=2,   label="FP-corrected")
        axes_ri[1].plot(res.f_THz[band], r["alpha_nofp"][band],  color="tab:gray",  lw=1.5, ls="--", label="No-FP")
        axes_ri[1].set_xlabel("Frequency (THz)")
        axes_ri[1].set_ylabel("Absorption coefficient  α (cm⁻¹)")
        axes_ri[1].set_title(f"α(f) — window = {lbl}")
        axes_ri[1].legend()
        axes_ri[1].grid(True, alpha=0.3)

        fig_ri.suptitle(f"FP vs no-FP — window = {lbl}" + title_suffix, fontsize=11)
        fig_ri.tight_layout()

        detail_figs.append((lbl, fig_ov, fig_res, fig_ri))

    # ── Show all figures ─────────────────────────────────────────────────
    plt.show()

    # ── Save ─────────────────────────────────────────────────────────────
    safe_label = sam_label.replace(" / ", "_").replace("/", "_")
    save_pairs = [
        (fig_n, f"fp_n_window_sweep_{safe_label}_{timestamp}"),
        (fig_a, f"fp_alpha_window_sweep_{safe_label}_{timestamp}"),
    ]
    for lbl, fig_ov, fig_res, fig_ri in detail_figs:
        w_tag = lbl.replace(" ", "")
        save_pairs += [
            (fig_ov,  f"fp_echo_overlay_{w_tag}_{safe_label}_{timestamp}"),
            (fig_res, f"fp_residual_{w_tag}_{safe_label}_{timestamp}"),
            (fig_ri,  f"fp_vs_nofp_{w_tag}_{safe_label}_{timestamp}"),
        ]
    for fig, stem in tqdm(save_pairs, desc="Saving figures", unit="fig"):
        thz_tds.viz.save_figure(fig, cfg.paths.output,
                                stem=stem,
                                dpi=cfg.plot.dpi, formats=cfg.plot.formats)


if __name__ == "__main__":
    
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        import os
        default = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
        main(os.path.abspath(default))
