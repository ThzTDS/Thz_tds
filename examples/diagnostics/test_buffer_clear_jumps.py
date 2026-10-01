"""
Buffer-clear timing check — are the ~±18 ps alignment shifts tied to
``rd.discardBufferedData()``?

The acquisition saves 1000 traces per ``airN`` / ``pointN`` folder and calls
``discardBufferedData()`` before saved traces 181, 361, 541, 721, 901 (then
discards 5 warm-up traces, which are NOT saved and NOT counted here).

For every selected folder this script, using ALL traces in saved order:

  1. plots raw t₀ and the applied alignment shift vs saved-trace number, with
     a labelled vertical line before every buffer-clear trace,
  2. plots histograms of t₀ and of the shift (two timing groups?),
  3. measures the jump at each buffer-clear boundary and compares it with the
     same statistic at every non-boundary position (null distribution), and
     looks for jumps inside blocks, alternating timing groups and drift.

Definitions (same as thz_tds.dataset.THZDataset)
-----------------------------------------------
t₀     Centre of a Gaussian fitted to the strongest |peak| of the RAW trace
       (``fit_gaussian_to_trace``; ``fit_gaussian_to_minimum`` for
       "gaussian_minimum_*" methods).  Time on the recorded axis, in ps.
shift  Time shift APPLIED to the trace by the alignment.  Positive = trace
       moved LATER.  For "gaussian_median_integer":
           shift = −round((t₀ − median t₀) / dt) · dt   ≈  median t₀ − t₀

Trace order
-----------
THZDataset reads files in os.listdir order (not saved order), so its trace
index is NOT the saved-trace number.  This script sorts files by the last
integer in the file name; the per-trace CSV lists the file name of every
trace so the ordering can be verified.

Outputs  (in <alignment_check.output>/<sample>_<YYYY-MM-DD_HH-MM-SS>/)
-------
<folder>/bufchk_t0_vs_trace.png     t₀, within-group residual, shift, raw argmin/argmax vs trace no.
<folder>/bufchk_histograms.png      t₀ and shift histograms
<folder>/bufchk_boundary_zoom.png   ±20 traces around every buffer clear
<folder>/bufchk_boundaries.csv      one row per buffer clear
<folder>/bufchk_blocks.csv          one row per 180-trace block
<folder>/bufchk_traces.csv          one row per trace
buffer_check_summary.csv            one row per folder + verdict

Usage
-----
    python3 examples/diagnostics/test_buffer_clear_jumps.py
    python3 examples/diagnostics/test_buffer_clear_jumps.py config/other.yaml
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # plots off — save only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import hypergeom

from thz_tds.dataset import THZDataset


# ── Config ────────────────────────────────────────────────────────────────────
_default = Path(__file__).resolve().parent.parent.parent / "config" / "all_samples_fp_batch.yaml"
config_path = sys.argv[1] if len(sys.argv) >= 2 else str(_default)

with open(config_path) as _f:
    _raw = yaml.safe_load(_f)

_loading = _raw.get("loading", {}) or {}
_align   = _raw.get("alignment", {}) or {}
_check   = _raw.get("alignment_check", {}) or {}
_buf     = _raw.get("buffer_check", {}) or {}

correction_factor = _loading.get("correction_factor", 1.0)
align_method      = _align.get("method", "gaussian_median_integer")
FIT_KWARGS = dict(
    corr_window_ps=_align.get("corr_window_ps", None),
    adaptive_fit_window=_align.get("adaptive_fit_window", True),
    fit_window_sigma_factor=_align.get("fit_window_sigma_factor", 5.0),
    fallback_fit_window_ps=_align.get("fallback_fit_window_ps", 15.0),
    min_sigma_ps=_align.get("min_sigma_ps", 0.05),
    max_sigma_ps=_align.get("max_sigma_ps", 10.0),
    min_fit_points=_align.get("min_fit_points", 10),
    max_center_shift_ps=_align.get("max_center_shift_ps", 5.0),
)

out_root      = Path(_check.get("output", "/mnt/Data/results"))
folder_select = _buf.get("folders") or _check.get("folders") or {"air": None, "point": None}
CLEAR_BEFORE  = [int(b) for b in _buf.get("clear_before", [181, 361, 541, 721, 901])]
EDGE_N        = int(_buf.get("edge_n", 5))
JUMP_THR_CFG  = _buf.get("jump_threshold_ps", None)
ZOOM_HALF     = 20   # traces each side in the boundary zoom figure

# Discover selected folders of every sample
jobs: list[tuple[str, Path]] = []
for _s in _raw.get("samples") or []:
    _sample_dir = Path(_s["dir"])
    _name       = _s.get("name", _sample_dir.name)
    if not _sample_dir.is_dir():
        print(f"WARNING: sample dir {_sample_dir} not found — skipped")
        continue
    for _prefix, _wanted in folder_select.items():
        _re    = re.compile(rf"^{re.escape(_prefix)}(\d+)$", re.IGNORECASE)
        _found = {int(_re.match(p.name).group(1)): p
                  for p in _sample_dir.iterdir() if p.is_dir() and _re.match(p.name)}
        _nums  = sorted(_found) if _wanted is None else [int(n) for n in _wanted]
        for _n in _nums:
            if _n in _found:
                jobs.append((_name, _found[_n]))
            else:
                print(f"WARNING: {_sample_dir / f'{_prefix}{_n}'} not found — skipped")

if not jobs:
    raise RuntimeError(f"No folders matching {folder_select} found.")

print(f"Config        : {config_path}")
print(f"Output        : {out_root}")
print(f"Method        : {align_method}")
print(f"Clear before  : {CLEAR_BEFORE}   edge_n = {EDGE_N}")
print(f"Folders ({len(jobs):>2})  : {', '.join(f'{s}/{p.name}' for s, p in jobs)}")
print()


# ── Loading in saved order ────────────────────────────────────────────────────
_NUM_RE = re.compile(r"(\d+)(?!.*\d)")   # last integer in the name


def load_in_saved_order(path: Path):
    """Return (t_ps, Y, trace_no, filenames) with traces in saved order.

    trace_no is the 1-based rank in the sorted file list, so an unreadable
    file leaves a gap instead of renumbering the rest.
    """
    files = [f for f in os.listdir(path) if (path / f).is_file()]
    nums  = [_NUM_RE.search(Path(f).stem) for f in files]
    if all(nums):
        order = sorted(range(len(files)), key=lambda i: (int(nums[i].group(1)), files[i]))
        keys  = [int(nums[i].group(1)) for i in order]
        if len(set(keys)) != len(keys):
            print("  WARNING: duplicate trace numbers in file names — check ordering")
    else:
        print("  WARNING: some file names have no number — sorting by name")
        order = sorted(range(len(files)), key=lambda i: files[i])
    files = [files[i] for i in order]

    x_ref, ys, tnos, names = None, [], [], []
    for rank, fname in enumerate(files, start=1):
        try:
            df = pd.read_csv(path / fname, sep="\t", header=0, comment="#")
            x = df.iloc[:, 0].to_numpy(dtype=float)
            y = df.iloc[:, 1].to_numpy(dtype=float) * correction_factor
        except Exception as exc:
            print(f"  skipping {fname}: {exc}")
            continue
        if x_ref is None:
            x_ref = x
        elif len(x) != len(x_ref) or not np.allclose(x, x_ref, atol=1e-9):
            y = np.interp(x_ref, x, y, left=0.0, right=0.0)
        ys.append(y)
        tnos.append(rank)
        names.append(fname)

    if x_ref is None:
        raise RuntimeError(f"No readable traces in {path}")
    return x_ref, np.vstack(ys), np.array(tnos), names


def fit_all(tool: THZDataset, t: np.ndarray, Y: np.ndarray):
    """Gaussian fit on every raw trace → t₀, amplitude, sigma (NaN on failure)."""
    fit = tool.fit_gaussian_to_minimum if "minimum" in align_method else tool.fit_gaussian_to_trace
    n = Y.shape[0]
    t0, amp, sig = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        try:
            f = fit(t, Y[i])
            t0[i], amp[i], sig[i] = f["t0_ps"], f["A"], f["sigma_ps"]
        except Exception:
            pass
    return t0, amp, sig


def applied_shift(tool: THZDataset, t: np.ndarray, Y: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """Shift applied by the alignment (+ = trace moved later), per trace.

    The Gaussian methods use the same formulas as THZDataset on the t₀ fitted
    above (no second fit); correlation methods call the library directly.
    """
    if align_method == "gaussian_median_integer":
        dt  = float(np.median(np.diff(t)))
        ref = float(np.nanmedian(t0))
        return -np.round((t0 - ref) / dt) * dt
    if align_method == "gaussian_minimum_mean":
        return float(np.nanmean(t0)) - t0
    tool.align_method = align_method
    _, shifts, _ = tool._run_alignment(t, Y)
    return np.asarray(shifts, dtype=float)


# ── Analysis helpers ──────────────────────────────────────────────────────────
def block_of(tno: np.ndarray) -> np.ndarray:
    """Block index 0..len(CLEAR_BEFORE): block k starts at CLEAR_BEFORE[k-1]."""
    return np.searchsorted(np.array(CLEAR_BEFORE), tno, side="right")


def edge_jump(series: dict[int, float], b: int) -> tuple[float, float]:
    """(median(after) − median(before), single-step value[b] − value[b−1])."""
    before = [series[k] for k in range(b - EDGE_N, b) if k in series]
    after  = [series[k] for k in range(b, b + EDGE_N) if k in series]
    med = (np.nanmedian(after) - np.nanmedian(before)) if before and after else np.nan
    step = (series[b] - series[b - 1]) if (b in series and b - 1 in series) else np.nan
    return float(med), float(step)


def find_groups(values: np.ndarray, thr: float):
    """Split values into clusters separated by gaps > thr (clusters ≥ 3 pts)."""
    v = np.sort(values[np.isfinite(values)])
    if len(v) < 6:
        return np.array([]), [float(np.median(v))] if len(v) else []
    gaps  = np.diff(v)
    cuts  = [0] + [i + 1 for i in np.where(gaps > thr)[0]] + [len(v)]
    clus  = [v[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 3]
    if len(clus) < 2:
        return np.array([]), [float(np.median(v))]
    splits = np.array([(clus[k][-1] + clus[k + 1][0]) / 2 for k in range(len(clus) - 1)])
    return splits, [float(np.median(c)) for c in clus]


def slope_per_100(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0] * 100.0)


def add_clear_lines(ax, label: bool) -> None:
    for b in CLEAR_BEFORE:
        ax.axvline(b - 0.5, color="tab:red", lw=1.0, ls="--", alpha=0.8)
        if label:
            ax.text(b - 0.5, 1.01, f"clear before #{b}", transform=ax.get_xaxis_transform(),
                    rotation=0, ha="center", va="bottom", fontsize=7, color="tab:red")


# ── Per-folder analysis ───────────────────────────────────────────────────────
def analyse_folder(path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{path.parent.name}/{path.name}"
    print(f"══ {label} " + "═" * max(0, 60 - len(label)))

    t, Y, tno, names = load_in_saved_order(path)
    n = len(tno)
    dt = float(np.median(np.diff(t)))
    print(f"  {n} traces (last saved #{tno[-1]}), dt = {dt:.4f} ps, first/last file: {names[0]} / {names[-1]}")
    if tno[-1] < CLEAR_BEFORE[-1]:
        print(f"  WARNING: fewer traces than the last buffer clear (#{CLEAR_BEFORE[-1]})")

    tool = THZDataset(path=str(path), correction_factor=correction_factor,
                      align=False, max_traces=1, **FIT_KWARGS)

    print(f"  fitting {n} traces ...")
    t0, amp, sig = fit_all(tool, t, Y)
    shift = applied_shift(tool, t, Y, t0)
    t_argmin = t[np.argmin(Y, axis=1)]
    t_argmax = t[np.argmax(Y, axis=1)]
    n_fail = int(np.isnan(t0).sum())
    if n_fail:
        print(f"  WARNING: {n_fail} Gaussian fits failed (NaN)")

    # ── Noise and jump threshold from consecutive-trace differences ──────────
    consec = np.diff(tno) == 1
    d_t0   = np.diff(t0)[consec]
    d_ok   = d_t0[np.isfinite(d_t0)]
    noise  = float(1.4826 * np.median(np.abs(d_ok - np.median(d_ok)))) if len(d_ok) else float("nan")
    thr    = float(JUMP_THR_CFG) if JUMP_THR_CFG is not None else max(8.0 * noise, 3.0 * dt)

    # ── Timing groups ────────────────────────────────────────────────────────
    splits, centers = find_groups(t0, thr)
    group = np.where(np.isfinite(t0), np.searchsorted(splits, t0), -1)
    n_groups = len(centers)

    # ── Boundary jumps + null distribution at non-boundary positions ─────────
    s_t0  = {int(k): float(v) for k, v in zip(tno, t0) if np.isfinite(v)}
    s_min = {int(k): float(v) for k, v in zip(tno, t_argmin)}
    null_pos = [p for p in range(EDGE_N + 1, int(tno[-1]) - EDGE_N + 2)
                if all(abs(p - b) > EDGE_N for b in CLEAR_BEFORE)]
    null_abs = np.array([abs(edge_jump(s_t0, p)[0]) for p in null_pos])
    null_abs = null_abs[np.isfinite(null_abs)]

    # Within-group jump: last EDGE_N traces of a group before the clear vs the
    # first EDGE_N after it.  Unaffected by alternation between groups.
    def group_jump(g: int, b: int) -> float:
        m_g = group == g
        before = t0[m_g & (tno < b)][-EDGE_N:]
        after  = t0[m_g & (tno >= b)][:EDGE_N]
        if len(before) == 0 or len(after) == 0:
            return float("nan")
        return float(np.nanmedian(after) - np.nanmedian(before))

    # Same statistic at every non-boundary position (null distribution)
    null_g = np.array([abs(group_jump(g, p)) for p in null_pos for g in range(n_groups)])
    null_g = null_g[np.isfinite(null_g)]

    boundary_rows = []
    for b in CLEAR_BEFORE:
        jm, js = edge_jump(s_t0, b)
        am, _  = edge_jump(s_min, b)
        pct = float(np.mean(null_abs < abs(jm)) * 100) if len(null_abs) and np.isfinite(jm) else np.nan
        n_ge = int(np.sum(null_abs >= abs(jm))) if np.isfinite(jm) else -1
        g_b  = group[tno == b - 1]
        g_a  = group[tno == b]
        switch = bool(len(g_b) and len(g_a) and g_b[0] >= 0 and g_a[0] >= 0 and g_b[0] != g_a[0])
        gj   = {f"jump_within_group{g}_ps": group_jump(g, b) for g in range(n_groups)}
        gmax = max((abs(v) for v in gj.values() if np.isfinite(v)), default=np.nan)
        boundary_rows.append(dict(
            clear_before=b, t0_jump_median_ps=jm, t0_step_ps=js, argmin_jump_median_ps=am,
            is_jump=bool(np.isfinite(jm) and abs(jm) > thr), group_switch=switch,
            null_percentile=pct, n_nonboundary_positions_ge=n_ge,
            **gj,
            within_group_null_percentile=(float(np.mean(null_g < gmax) * 100)
                                          if len(null_g) and np.isfinite(gmax) else np.nan),
        ))

    # ── Consecutive jumps / group switches inside vs at boundaries ───────────
    idx_pairs = np.where(consec)[0]
    later     = tno[idx_pairs + 1]
    at_b      = np.isin(later, CLEAR_BEFORE)
    big       = np.abs(np.diff(t0)[idx_pairs]) > thr
    sw        = (group[idx_pairs] != group[idx_pairs + 1]) & (group[idx_pairs] >= 0) & (group[idx_pairs + 1] >= 0)
    within_jump_tnos = later[big & ~at_b].tolist()
    n_sw_b, n_sw_w   = int((sw & at_b).sum()), int((sw & ~at_b).sum())
    n_pairs_w        = int((~at_b).sum())
    # Chance that ≥ n_sw_b of all switches land on the boundary pairs if switches were random
    n_sw = n_sw_b + n_sw_w
    p_chance = float(hypergeom.sf(n_sw_b - 1, len(idx_pairs), int(at_b.sum()), n_sw)) if n_sw else float("nan")

    # ── Per-block stats + drift (t₀ minus its group centre) ──────────────────
    blk = block_of(tno)
    t0_detr = t0 - np.array([centers[g] if g >= 0 else np.nan for g in group]) if n_groups else t0 - np.nanmedian(t0)
    block_rows = []
    edges = [1] + CLEAR_BEFORE + [int(tno[-1]) + 1]
    for k in range(len(edges) - 1):
        m = blk == k
        if not m.any():
            continue
        gk = group[m]
        block_rows.append(dict(
            block=k + 1, first_trace=edges[k], last_trace=edges[k + 1] - 1, n=int(m.sum()),
            t0_median_ps=float(np.nanmedian(t0[m])), t0_std_ps=float(np.nanstd(t0[m])),
            shift_median_ps=float(np.nanmedian(shift[m])),
            **{f"frac_group{g}": float(np.mean(gk == g)) for g in range(n_groups)} if n_groups > 1 else {},
            group_switches_within=int((sw & ~at_b & (blk[idx_pairs] == k)).sum()),
            jumps_within=int((big & ~at_b & (blk[idx_pairs] == k)).sum()),
            drift_ps_per_100=slope_per_100(tno[m].astype(float), t0_detr[m]),
        ))
    drift_all = slope_per_100(tno.astype(float), t0_detr)

    # ── Does the whole waveform move, or only the fitted feature? ────────────
    feat = {}
    if n_groups >= 2:
        g_lo, g_hi = group == 0, group == n_groups - 1
        feat = dict(
            group_sep_t0_ps=centers[-1] - centers[0],
            group_sep_argmin_ps=float(np.median(t_argmin[g_hi]) - np.median(t_argmin[g_lo])),
            group_sep_argmax_ps=float(np.median(t_argmax[g_hi]) - np.median(t_argmax[g_lo])),
            amp_sign_lo=float(np.mean(np.sign(amp[g_lo]))),
            amp_sign_hi=float(np.mean(np.sign(amp[g_hi]))),
        )

    # ── Alternation: switch rate and odd/even pattern ────────────────────────
    alt_rate = n_sw / max(len(idx_pairs), 1)
    parity_match = float("nan")
    if n_groups == 2:
        valid = group >= 0
        par   = tno % 2
        maj   = {p: np.bincount(group[valid & (par == p)], minlength=2).argmax() for p in (0, 1)}
        parity_match = float(np.mean(group[valid] == np.array([maj[p] for p in par[valid]])))
    resid_std = float(np.nanstd(t0_detr))
    gj_all = [abs(v) for r in boundary_rows for k, v in r.items()
              if k.startswith("jump_within_group") and np.isfinite(v)]
    gj_max = max(gj_all, default=float("nan"))

    # ── Verdict (evidence, not proof) ────────────────────────────────────────
    n_bjump = sum(r["is_jump"] for r in boundary_rows)
    notes = []
    if n_groups >= 2 and alt_rate > 0.9:
        verdict = (f"Groups alternate trace by trace (switch rate {alt_rate:.3f}, odd/even pattern match "
                   f"{parity_match:.1%}) with the same phase before and after every clear; within-group "
                   f"jump at clears ≤ {gj_max:.3f} ps (within-group scatter {resid_std:.3f} ps): "
                   f"not caused by buffer clearing.")
    elif n_groups < 2 and n_bjump == 0:
        verdict = "No distinct timing groups and no boundary jumps: no evidence of a buffer-clear effect."
    elif n_groups < 2:
        verdict = (f"No distinct groups, but {n_bjump}/{len(CLEAR_BEFORE)} boundaries exceed the jump "
                   f"threshold; see null_percentile.")
    elif n_sw_w == 0 and n_sw_b > 0:
        verdict = (f"All {n_sw} group switches happen exactly at buffer clears "
                   f"({n_sw_b}/{len(CLEAR_BEFORE)} clears switch; chance p = {p_chance:.2g}): "
                   f"strongly consistent with buffer clearing (correlation, not proof).")
    elif n_sw_b > 0 and n_sw_w <= 2:
        verdict = (f"{n_sw_b} switches at clears, {n_sw_w} inside blocks (chance p = {p_chance:.2g}): "
                   f"mostly consistent with buffer clearing.")
    else:
        verdict = (f"{n_sw_w} group switches INSIDE blocks vs {n_sw_b} at clears "
                   f"(switch rate inside blocks {n_sw_w / max(n_pairs_w, 1):.3f}): "
                   f"not explained by buffer clearing alone.")
    if feat:
        if abs(feat["group_sep_argmin_ps"] - feat["group_sep_t0_ps"]) > max(3 * dt, 0.2 * abs(feat["group_sep_t0_ps"])):
            notes.append(f"raw argmin moves {feat['group_sep_argmin_ps']:.2f} ps vs t0 {feat['group_sep_t0_ps']:.2f} ps "
                         f"-> fit may be locking onto different features, not a whole-waveform shift")
        if np.sign(feat["amp_sign_lo"]) != np.sign(feat["amp_sign_hi"]):
            notes.append("fitted peak sign differs between groups -> strongest-|peak| picking flips lobe")
    print(f"  noise = {noise:.4f} ps   threshold = {thr:.3f} ps   groups = {n_groups} {np.round(centers, 3).tolist()}")
    for r in boundary_rows:
        gtxt = "  ".join(f"g{g} {r[f'jump_within_group{g}_ps']:+.3f}" for g in range(n_groups))
        print(f"  clear before #{r['clear_before']:>4}: jump(median ±{EDGE_N}) = {r['t0_jump_median_ps']:+9.3f} ps"
              f"  step = {r['t0_step_ps']:+9.3f} ps  switch = {r['group_switch']!s:5}  "
              f"within-group [{gtxt}] ps (null pct {r['within_group_null_percentile']:.1f})")
    print(f"  switches at clears = {n_sw_b}, inside blocks = {n_sw_w} (switch rate {alt_rate:.3f}, "
          f"odd/even match {parity_match:.1%}); within-block jumps: {len(within_jump_tnos)} "
          f"{within_jump_tnos[:20]}{' ...' if len(within_jump_tnos) > 20 else ''}")
    print(f"  drift = {drift_all:+.4f} ps / 100 traces")
    print(f"  VERDICT: {verdict}")
    for s in notes:
        print(f"  NOTE: {s}")

    # ── Figures ──────────────────────────────────────────────────────────────
    gcols = plt.get_cmap("tab10")
    t0_def = ("t₀ = fitted Gaussian centre of strongest |peak| on the RAW trace; "
              "shift = applied alignment shift (+ = trace moved later)")

    # 1) t₀ / shift / raw extrema vs saved-trace number
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    ax = axes[0]
    if n_groups > 1:
        for g in range(n_groups):
            m = group == g
            ax.plot(tno[m], t0[m], ".", ms=3, color=gcols(g), label=f"group {g}: {m.sum()} traces, median {centers[g]:.3f} ps")
    else:
        ax.plot(tno, t0, ".", ms=3, color="tab:blue", label="raw t₀")
    if n_groups < 2:   # a block median between two groups would be meaningless
        for r in block_rows:
            ax.hlines(r["t0_median_ps"], r["first_trace"], r["last_trace"], color="black", lw=1.2)
    ax.set_ylabel("raw t₀ [ps]")
    ax.legend(fontsize=8, loc="best")
    add_clear_lines(ax, label=True)
    ax.set_title(f"{label} — {align_method} — all {n} traces in saved order\n{t0_def}", fontsize=9, pad=16)

    for g in range(max(n_groups, 1)):
        m = group == g
        axes[1].plot(tno[m], t0_detr[m], ".", ms=3, color=gcols(g) if n_groups > 1 else "tab:blue")
    axes[1].set_ylabel("t₀ − group median [ps]\n(within-group residual)")
    add_clear_lines(axes[1], label=False)

    axes[2].plot(tno, shift, ".", ms=3, color="tab:green")
    axes[2].set_ylabel("applied shift [ps]\n(+ = moved later)")
    add_clear_lines(axes[2], label=False)

    axes[3].plot(tno, t_argmin, ".", ms=2, color="tab:purple", label="raw argmin time")
    axes[3].plot(tno, t_argmax, ".", ms=2, color="tab:orange", label="raw argmax time")
    axes[3].set_ylabel("raw extremum time [ps]\n(no fit)")
    axes[3].set_xlabel("saved trace number (warm-up traces excluded)")
    axes[3].legend(fontsize=8, loc="best")
    add_clear_lines(axes[3], label=False)
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bufchk_t0_vs_trace.png", dpi=150)
    plt.close(fig)

    # 2) histograms
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ok = np.isfinite(t0)
    a1.hist(t0[ok], bins=min(200, max(20, int(np.ptp(t0[ok]) / max(dt / 2, 1e-6)) + 1)), color="tab:blue")
    for c in centers:
        a1.axvline(c, color="black", ls="--", lw=1)
    a1.set_xlabel("raw t₀ [ps]")
    a1.set_ylabel("count")
    a1.set_title(f"raw t₀ — {n_groups} group(s) (gap > {thr:.2f} ps)")
    a2.hist(shift[np.isfinite(shift)], bins=100, color="tab:green")
    a2.set_xlabel("applied shift [ps]  (+ = moved later)")
    a2.set_title(f"applied shift  mean {np.nanmean(shift):+.4f}  std {np.nanstd(shift):.4f} ps")
    for a in (a1, a2):
        a.grid(True, alpha=0.3)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_dir / "bufchk_histograms.png", dpi=150)
    plt.close(fig)

    # 3) zoom ±ZOOM_HALF traces around every clear
    fig, axes = plt.subplots(1, len(CLEAR_BEFORE), figsize=(3.2 * len(CLEAR_BEFORE), 3.8), sharey=True, squeeze=False)
    for a, r in zip(axes[0], boundary_rows):
        b = r["clear_before"]
        m = (tno >= b - ZOOM_HALF) & (tno < b + ZOOM_HALF)
        a.plot(tno[m], t0[m], "-", lw=0.4, color="tab:gray")
        for g in range(max(n_groups, 1)):
            mg = m & (group == g)
            a.plot(tno[mg], t0[mg], "o", ms=3, color=gcols(g) if n_groups > 1 else "tab:blue")
        a.axvline(b - 0.5, color="tab:red", ls="--")
        gtxt = ", ".join(f"g{g} {r[f'jump_within_group{g}_ps']:+.3f}" for g in range(n_groups))
        a.set_title(f"clear before #{b}\nstep {r['t0_step_ps']:+.2f} ps\nwithin-group: {gtxt} ps", fontsize=8)
        a.set_xlabel("saved trace #")
        a.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("raw t₀ [ps]")
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "bufchk_boundary_zoom.png", dpi=150)
    plt.close(fig)

    # ── CSVs ─────────────────────────────────────────────────────────────────
    pd.DataFrame(boundary_rows).to_csv(out_dir / "bufchk_boundaries.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out_dir / "bufchk_blocks.csv", index=False)
    pd.DataFrame(dict(
        trace_no=tno, file=names, block=blk + 1, t0_ps=t0, shift_ps=shift,
        t_argmin_ps=t_argmin, t_argmax_ps=t_argmax, amp=amp, sigma_ps=sig, group=group,
    )).to_csv(out_dir / "bufchk_traces.csv", index=False)
    print(f"  saved to {out_dir}\n")

    return dict(
        folder=path.name, n_traces=n, method=align_method, dt_ps=dt,
        shift_mean_ps=float(np.nanmean(shift)), shift_std_ps=float(np.nanstd(shift)),
        shift_min_ps=float(np.nanmin(shift)), shift_max_ps=float(np.nanmax(shift)),
        t0_noise_ps=noise, jump_threshold_ps=thr, n_groups=n_groups,
        group_centers_ps=";".join(f"{c:.3f}" for c in centers),
        boundary_jumps_ps=";".join(f"{r['t0_jump_median_ps']:+.3f}" for r in boundary_rows),
        n_boundary_jumps=n_bjump, switches_at_clears=n_sw_b, switches_inside_blocks=n_sw_w,
        switch_rate=alt_rate, odd_even_match=parity_match,
        p_chance_switches_at_clears=p_chance,
        max_within_group_jump_at_clears_ps=gj_max, within_group_scatter_ps=resid_std,
        n_within_block_jumps=len(within_jump_tnos),
        drift_ps_per_100=drift_all, **feat,
        verdict=verdict, notes=" | ".join(notes),
    )


# ── Run ───────────────────────────────────────────────────────────────────────
# One timestamp per run → new folder each run, nothing gets overwritten
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

summary: dict[str, list[dict]] = {}
for sample_name, folder in jobs:
    try:
        row = analyse_folder(folder, out_root / f"{sample_name}_{RUN_STAMP}" / folder.name)
        summary.setdefault(sample_name, []).append(row)
    except Exception as exc:
        print(f"  FAILED on {folder}: {exc}\n")

for sample_name, rows in summary.items():
    sample_dir = out_root / f"{sample_name}_{RUN_STAMP}"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(sample_dir / "buffer_check_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Summary: {sample_dir / 'buffer_check_summary.csv'}")
