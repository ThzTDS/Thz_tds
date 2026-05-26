"""
examples/diagnose_dataset_timing.py
====================================
Diagnostic script for THz-TDS dataset timing, alignment quality, and
reference–sample pairing checks.

Physics reminder
----------------
Alignment is applied *within* each folder to reduce scan-to-scan jitter.
The reference–sample delay is NOT aligned away — it encodes the refractive
index of the sample and must be preserved.

Usage
-----
Minimal::

    python examples/diagnose_dataset_timing.py \\
        --ref  /data/PP_B_1.3/air1 \\
        --sam  /data/PP_B_1.3/point1 \\
        --thickness 0.5e-3

With expected n and full directory scan::

    python examples/diagnose_dataset_timing.py \\
        --ref       /data/PP_B_1.3/air1 \\
        --sam       /data/PP_B_1.3/point1 \\
        --thickness 0.5e-3 \\
        --n-expected 3.1 \\
        --base-dir  /data/PP_B_1.3

Headless (no plot windows)::

    python examples/diagnose_dataset_timing.py ... --no-plot

Save diagnostic files::

    python examples/diagnose_dataset_timing.py ... --output-dir ./reports
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from thz_tds.dataset_diag import (
    THZDatasetDiag,
    check_fp_echo_window,
    compare_alignment_methods,
    compare_reference_sample_timing,
    plot_aligned_traces,
    plot_alignment_comparison,
    plot_alignment_method_detail,
    plot_average_trace,
    plot_fft_magnitude,
    plot_jitter_shifts,
    plot_raw_traces,
    plot_ref_sample_overlay,
    plot_trace_residuals,
    plot_transfer_function,
    print_dataset_report,
    save_dataset_report_json,
    save_timing_report_csv,
    scan_samples,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: Dict, *keys, default=None):
    """Drill into nested dict: _cfg_get(cfg, 'paths', 'reference')."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, None)
        if node is None:
            return default
    return node


def main() -> None:
    parser = argparse.ArgumentParser(
        description="THz-TDS dataset diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config",        nargs="?", default=None,
                        help="Path to a YAML config file (positional, e.g. config/diagnose_config.yaml). "
                             "Command-line flags override YAML values.")
    parser.add_argument("--ref",        default=None,
                        help="Path to reference (air) folder")
    parser.add_argument("--sam",        default=None,
                        help="Path to sample folder")
    parser.add_argument("--thickness",  type=float, default=None,
                        help="Sample thickness in metres (e.g. 0.5e-3)")
    parser.add_argument("--n-expected", type=float, default=None,
                        help="Expected refractive index (enables FP echo check)")
    parser.add_argument("--base-dir",   default=None,
                        help="Scan all sample sub-folders in this directory")
    parser.add_argument("--align-method", default=None,
                        help="Internal alignment method (default: correlation_subsample)")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip all plot windows (batch / headless mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for saved reports")
    args = parser.parse_args()

    # ── Merge YAML → CLI (CLI wins) ──────────────────────────────────────
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)
        print(f"Config loaded: {args.config}")

    ref_path     = args.ref          or _cfg_get(cfg, "paths", "reference")
    # Accept both paths.sample (singular) and paths.samples[0] (list, matches main config format)
    _samples_list = _cfg_get(cfg, "paths", "samples")
    sam_path     = (args.sam
                    or _cfg_get(cfg, "paths", "sample")
                    or (_samples_list[0] if isinstance(_samples_list, list) and _samples_list else None))
    thickness    = args.thickness    or _cfg_get(cfg, "sample", "thickness_m")
    n_expected   = args.n_expected   or _cfg_get(cfg, "sample", "expected_n")
    base_dir     = args.base_dir     or _cfg_get(cfg, "paths", "base_dir")
    align_method = (args.align_method
                    or _cfg_get(cfg, "alignment", "method")
                    or "correlation_subsample")
    output_dir   = (args.output_dir
                    or _cfg_get(cfg, "paths", "output")
                    or ".")
    show_plots   = (not args.no_plot) and _cfg_get(cfg, "plot", "show", default=True)
    n_fp_echoes  = int(_cfg_get(cfg, "sample", "n_fp_echoes", default=3))
    compare_all  = bool(_cfg_get(cfg, "alignment", "compare_all", default=False))

    if not ref_path:
        parser.error("Reference path required: use --ref or set paths.reference in YAML.")
    if not sam_path:
        parser.error("Sample path required: use --sam or set paths.sample in YAML.")
    if thickness is None:
        parser.error("Thickness required: use --thickness or set sample.thickness_m in YAML.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference ──────────────────────────────────────────────────
    _section("Loading reference")
    ref = THZDatasetDiag(path=ref_path, align=True, align_method=align_method)
    print_dataset_report(ref)
    save_dataset_report_json(ref, output_dir / f"report_ref_{ref.name}.json")

    # ── Load sample ─────────────────────────────────────────────────────
    _section("Loading sample")
    sam = THZDatasetDiag(path=sam_path, align=True, align_method=align_method)
    print_dataset_report(sam)
    save_dataset_report_json(sam, output_dir / f"report_sam_{sam.name}.json")

    # ── Reference–sample timing comparison ──────────────────────────────
    _section("Reference–sample timing")
    timing = compare_reference_sample_timing(
        ref, sam, thickness, expected_n=n_expected
    )
    print(timing)
    save_timing_report_csv(
        [timing], output_dir / "timing_report.csv", labels=[sam.name]
    )
    print(f"\nTiming report → {output_dir / 'timing_report.csv'}")

    # ── Fabry–Pérot echo window check ────────────────────────────────────
    if n_expected is not None:
        _section("Fabry–Pérot echo window")
        fp = check_fp_echo_window(sam, n_expected, thickness, n_echoes=n_fp_echoes)
        print(f"  Round-trip echo delay : {fp['echo_delay_ps']:.4f} ps")
        for k, (t_echo, ok) in enumerate(
            zip(fp["echo_times_ps"], fp["echoes_in_window"]), start=1
        ):
            print(f"  Echo {k} at {t_echo:.2f} ps : {'OK' if ok else 'OUTSIDE scan window'}")
        for w in fp["warnings"]:
            print(f"  WARNING: {w}")

    # ── Alignment method comparison ──────────────────────────────────────
    if compare_all:
        _section("Alignment method comparison (reference folder)")
        method_results = compare_alignment_methods(
            ref_path,
            corr_window_ps=_cfg_get(cfg, "alignment", "corr_window_ps"),
        )
        print(f"  {'Method':<25}  {'mean (ps)':>10}  {'std (ps)':>10}  {'max|Δ| (ps)':>12}")
        print("  " + "-" * 62)
        for m, ds in method_results.items():
            if ds.alignment:
                al = ds.alignment
                print(f"  {m:<25}  {al.mean_ps:>10.5f}  {al.std_ps:>10.5f}  {al.max_abs_ps:>12.5f}")
            else:
                print(f"  {m:<25}  {'—':>10}  {'—':>10}  {'—':>12}")
        if show_plots:
            fig_tr, fig_jit = plot_alignment_comparison(
                method_results, title_prefix=f"{ref.name} — "
            )
            plt.show(block=False)
            for m, ds in method_results.items():
                plot_alignment_method_detail(ds, method_name=m)
            plt.show(block=False)

    # ── Scan all samples in base_dir ─────────────────────────────────────
    if base_dir:
        _section(f"Scanning all samples in {base_dir}")
        all_samples = scan_samples(base_dir, align=True, align_method=align_method)
        all_reports, all_labels = [], []
        for s in all_samples:
            r = compare_reference_sample_timing(
                ref, s, thickness, expected_n=n_expected
            )
            all_reports.append(r)
            all_labels.append(s.name)
            warn_str = "  |  ".join(r.warnings) if r.warnings else "OK"
            print(
                f"  {s.name:30s}  n_peak = {r.n_from_delay:.4f}"
                f"  Δt = {r.delay_ps:.3f} ps  {warn_str}"
            )
        if all_reports:
            save_timing_report_csv(
                all_reports,
                output_dir / "timing_report_all.csv",
                labels=all_labels,
            )
            print(f"\nFull timing report → {output_dir / 'timing_report_all.csv'}")

    # ── Plots ────────────────────────────────────────────────────────────
    if not show_plots:
        return

    _section("Generating plots")

    # Uncomment any of these when needed:
    # plot_raw_traces(ref, title=f"Reference — raw traces ({ref.name})")
    # plot_aligned_traces(ref, title=f"Reference — aligned traces ({ref.name})")
    # plot_jitter_shifts(ref, title=f"Reference — alignment shifts ({ref.name})")
    # plot_trace_residuals(ref, title=f"Reference — residuals ({ref.name})")
    # plot_average_trace(ref, title=f"Reference — average ({ref.name})")
    # plot_fft_magnitude(ref, title=f"Reference — FFT ({ref.name})")
    # plot_raw_traces(sam, title=f"Sample — raw traces ({sam.name})")
    # plot_aligned_traces(sam, title=f"Sample — aligned traces ({sam.name})")
    # plot_jitter_shifts(sam, title=f"Sample — alignment shifts ({sam.name})")
    # plot_trace_residuals(sam, title=f"Sample — residuals ({sam.name})")
    # plot_average_trace(sam, title=f"Sample — average ({sam.name})")
    # plot_fft_magnitude(sam, title=f"Sample — FFT ({sam.name})")
    # plot_ref_sample_overlay(ref, sam, normalize=True)
    # plot_transfer_function(ref, sam)

    plt.show()


if __name__ == "__main__":
    import os as _os
    _default = _os.path.join(
        _os.path.dirname(__file__), "..", "config", "diagnose_config.yaml"
    )
    if len(sys.argv) == 1 and _os.path.exists(_default):
        sys.argv.append(_os.path.abspath(_default))
    main()
