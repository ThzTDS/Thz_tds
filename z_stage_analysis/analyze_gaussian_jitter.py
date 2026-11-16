"""Gaussian-median integer jitter correction and optical Z-stage diagnostics.

No cross-correlation or correlation_subsample is used. Each stationary group of
traces is processed independently (one group in memory per worker):

1. Gaussian fit (THZDataset.fit_gaussian_to_trace) of every trace -> t0_i.
2. QC: failed fits, fits at parameter bounds, robust timing/amplitude outliers.
3. Group arrival time = median(t0_i of valid traces); integer-sample shifts toward
   it only sharpen the averaged waveform and do not change the group arrival time.
4. Timestamp-aware within-group trend (trace_log.csv).
5. Across groups: apparent displacement, reference returns, drift diagnostics and
   an optional, clearly labelled assumption-based drift correction.

Run:  python3 analyze_gaussian_jitter.py --config config.yaml
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thz_tds.dataset import THZDataset
import zstage_core as core

OUT_SUBDIR = 'gaussian_median_integer'


def save_with_zoom(fig, ax, path, zoom, dpi):
    """Save full figure; if zoom=[t_min, t_max] (ps), also save <name>_zoom.png
    with the x-axis limited to that window and y rescaled to the visible data."""
    fig.tight_layout(); fig.savefig(path, dpi=dpi)
    if zoom:
        lo, hi = float(zoom[0]), float(zoom[1])
        ys = [l.get_ydata()[(l.get_xdata() >= lo) & (l.get_xdata() <= hi)] for l in ax.get_lines()]
        ys = np.concatenate([y for y in ys if len(y)]) if any(len(y) for y in ys) else np.array([])
        ys = ys[np.isfinite(ys)]
        ax.set_xlim(lo, hi)
        if len(ys):
            pad = 0.05 * (ys.max() - ys.min() or 1); ax.set_ylim(ys.min() - pad, ys.max() + pad)
        fig.tight_layout(); fig.savefig(path.with_name(path.stem + '_zoom' + path.suffix), dpi=dpi)
    plt.close(fig)


def z_ticks(ax, idx, z, unit):
    """Label a measurement-order x-axis with the commanded Z of each group.
    Z repeats (reference returns), so the order is kept and only the labels change."""
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f'{v:.4f}' for v in z], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(f'Commanded Z ({unit}), in measurement order')


def to_builtin(x):
    """numpy scalars/containers -> plain Python for yaml.safe_dump (NaN -> None)."""
    if isinstance(x, dict): return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [to_builtin(v) for v in x]
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return None if not np.isfinite(x) else float(x)
    return x


def read_experiment_config(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding='utf8', errors='replace').splitlines():
            if '=' in line:
                k, v = line.split('=', 1); out[k.strip()] = v.strip()
    return out


def load_trace_log(base: Path):
    p = base / 'trace_log.csv'
    if not p.is_file():
        warnings.warn('trace_log.csv not found: timestamp analysis disabled')
        return None
    tl = pd.read_csv(p)
    tl['ts'] = pd.to_datetime(tl['acquired_utc'], errors='coerce', utc=True, format='ISO8601')
    tl['file'] = tl['relative_file'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[-1]
    return tl


def process_folder(job):
    """Worker: load one folder with THZDataset (align=False) and process it."""
    folder, cfg, ts_map, t_ref = job
    ds = THZDataset(folder, correction_factor=float(cfg.get('correction_factor', 0.0414)), align=False,
                    fit_method=None, max_traces=cfg['analysis'].get('max_traces_per_position'))
    ds.all_y = None  # unaligned copy not needed; halves memory
    t = core.time_ps(ds.x_ps, cfg)
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'Non-increasing time axis in {folder}')
    if t_ref is not None and (len(t) != len(t_ref) or
                              np.max(np.abs(t - t_ref)) > float(cfg['analysis'].get('max_time_grid_error_ps', 1e-5))):
        raise ValueError(f'Time grid differs from first group: {folder}')
    elapsed = None
    if ts_map is not None:
        elapsed = np.array([ts_map.get(f, np.nan) for f in ds.loaded_files], float)
        if not np.isfinite(elapsed).any():
            warnings.warn(f'{folder}: no timestamps matched; timing-trend analysis skipped')
            elapsed = None
    fitter = core.configure_fitter(ds, cfg.get('gaussian_fit', {}))
    res = core.process_group(t, ds.all_y_raw, fitter, cfg, elapsed)
    res['traces'].insert(1, 'file', ds.loaded_files)
    res['t'] = t
    return res


def main(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding='utf8'))
    base = Path(cfg['data_dir']).expanduser()
    if not base.is_absolute(): base = (path.parent / base).resolve()
    out = Path(cfg['output_dir']).expanduser()
    if not out.is_absolute(): out = (path.parent / out).resolve()
    out = out / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    if cfg['alignment']['method'] != 'gaussian_median_integer':
        raise ValueError('Only gaussian_median_integer is supported')
    factor = core.um_per_ps(cfg['incidence_angle_deg'])
    target = float(cfg.get('analysis', {}).get('precision_target_um', 1.0))
    zoom = cfg.get('plot', {}).get('zoom_window_ps')
    issues = []

    manifest = pd.read_csv(base / 'manifest.csv').reset_index(drop=True)
    required = {'cycle', 'step', 'z_command', 'folder'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Missing manifest columns: {required - set(manifest.columns)}')

    # Units sanity against the acquisition record (never silently converted).
    exp = read_experiment_config(base / 'experiment_config.txt')
    for k in ('stage_coordinate_unit', 'thz_time_axis_unit'):
        if exp.get(k, '').upper() == 'UNVERIFIED':
            issues.append(f'experiment_config.txt: {k}=UNVERIFIED; analysis assumes '
                          f'{cfg["stage_unit"] if "stage" in k else cfg["time_unit"]} from config.yaml.')
    if exp.get('incidence_angle_deg', '').lower() in ('nan', ''):
        issues.append(f'experiment_config.txt has no incidence angle; config uses {cfg["incidence_angle_deg"]} deg.')
    if exp.get('z_reference') and abs(float(exp['z_reference']) - float(cfg['z_reference'])) > 1e-9:
        issues.append(f'z_reference differs: experiment {exp["z_reference"]} vs config {cfg["z_reference"]}.')

    tl = load_trace_log(base) if cfg.get('timing', {}).get('use_trace_log', True) else None
    t_start = tl['ts'].min() if tl is not None else None

    jobs, meta = [], []
    req_status = cfg['analysis'].get('require_status', 'completed')
    max_tr = cfg['analysis'].get('max_traces_per_position')
    for idx, row in manifest.iterrows():
        folder = base / str(row['folder'])
        if req_status and 'status' in row and str(row['status']) != req_status:
            issues.append(f'Skipped {row["folder"]}: status={row["status"]}'); continue
        if not folder.is_dir():
            issues.append(f'Skipped {row["folder"]}: folder missing'); continue
        n_files = len([f for f in folder.iterdir() if f.is_file()])
        if 'trace_count' in row and pd.notna(row['trace_count']) and int(row['trace_count']) != n_files:
            issues.append(f'{row["folder"]}: manifest trace_count={int(row["trace_count"])} but {n_files} files')
        ts_map = None
        if tl is not None:
            g = tl[(tl['cycle'] == row['cycle']) & (tl['step'] == row['step'])].sort_values('trace_number')
            if len(g) != n_files:
                issues.append(f'{row["folder"]}: trace_log has {len(g)} rows, folder has {n_files} files')
            expect = [f'Trace_{n:04d}.tsv' for n in g['trace_number']]
            if list(g['file']) != expect:
                issues.append(f'{row["folder"]}: trace_log file names do not follow trace_number order')
            ts_map = dict(zip(g['file'], (g['ts'] - t_start).dt.total_seconds()))
        jobs.append([folder, cfg, ts_map, None]); meta.append((idx, row))
    if not jobs:
        raise RuntimeError('No valid measurement folders')

    # First group sequentially (defines the time grid), the rest optionally in parallel.
    n_workers = max(1, int(cfg['analysis'].get('n_workers', 1)))
    results = [None] * len(jobs)
    first_ok = None
    for k, job in enumerate(jobs):
        try:
            results[k] = process_folder(job); first_ok = k; break
        except (RuntimeError, ValueError) as exc:
            issues.append(f'Skipped {meta[k][1]["folder"]}: {exc}')
    if first_ok is None:
        raise RuntimeError('No group could be processed')
    t = results[first_ok]['t']
    rest = list(range(first_ok + 1, len(jobs)))
    for k in rest: jobs[k][3] = t

    def handle(k, fut_or_res):
        try:
            results[k] = fut_or_res() if callable(fut_or_res) else fut_or_res
        except (RuntimeError, ValueError) as exc:
            issues.append(f'Skipped {meta[k][1]["folder"]}: {exc}')
    if n_workers > 1 and rest:
        with ProcessPoolExecutor(n_workers) as pool:
            futs = {k: pool.submit(process_folder, jobs[k]) for k in rest}
            for k in rest: handle(k, futs[k].result)
    else:
        for k in rest: handle(k, lambda k=k: process_folder(jobs[k]))

    groups, trace_tabs, averages = [], [], {}
    for k, res in enumerate(results):
        if res is None: continue
        idx, row = meta[k]
        st = dict(measurement_index=int(idx), cycle=row['cycle'], step=row['step'],
                  folder=str(row['folder']), z_command=float(row['z_command']))
        for col in ('previous_z_command', 'approach_direction', 'acquisition_start_utc', 'acquisition_end_utc'):
            if col in row.index: st[col] = row[col]
        st.update(res['stats'])
        if not st.get('baseline_window_ok', True):
            issues.append(f'{row["folder"]}: baseline window contains signal '
                          f'(RMS ratio {st["baseline_rms_ratio"]:.1f}); consider baseline_correction: false')
        groups.append(st)
        tr = res['traces']
        tr.insert(0, 'measurement_index', int(idx)); tr.insert(1, 'step', row['step'])
        tr.insert(2, 'trace_number', tr.pop('trace_index') + 1)
        trace_tabs.append(tr)
        averages[int(idx)] = (res['raw_avg'], res['corrected_avg'], row['z_command'])
        print(f'Position {idx}: {st["n_traces"]} traces; {st["n_valid"]} valid; '
              f'median={st["median_arrival_ps"]:.6f} ps; SD={st["jitter_std_fs"]:.1f} fs; '
              f'slope={st.get("slope_fs_per_s", np.nan):.2f} fs/s', flush=True)

    ana = core.analyze_positions(pd.DataFrame(groups), cfg)
    df, ref, drift = ana['positions'], ana['references'], ana['drift']
    issues += ana['notes']
    tr = pd.concat(trace_tabs, ignore_index=True)
    tr['arrival_rel_fs'] = (tr['arrival_ps'] - df['median_arrival_ps'].iloc[0]) * 1000
    tr = tr.merge(df[['measurement_index', 'expected_um']], on='measurement_index')
    # Arrival minus the nominal commanded delay: a continuous curve here = drift through moves.
    tr['timing_offset_from_commanded_fs'] = tr['arrival_rel_fs'] - \
        int(cfg['analysis']['delay_sign_for_positive_z']) * tr['expected_um'] / factor * 1000

    # ---- CSV outputs ----
    df.to_csv(out / 'position_results.csv', index=False)
    tr.to_csv(out / 'trace_timing.csv', index=False)
    ref.to_csv(out / 'reference_returns.csv', index=False)
    drift_cols = ['measurement_index', 'step', 'z_command', 'is_reference', 'elapsed_start_s', 'elapsed_mid_s',
                  'elapsed_end_s', 'duration_s', 'n_valid', 'slope_fs_per_s', 'slope_ci_low_fs_per_s',
                  'slope_ci_high_fs_per_s', 'trend_over_group_fs', 'sd_raw_fs', 'sd_detrended_fs',
                  'lag1_autocorr_detrended', 'n_eff', 'se_median_fs', 'n_missing_timestamps',
                  'median_interval_s', 'max_interval_s', 'n_irregular_intervals', 'irregular_after_traces',
                  'n_nonmonotonic', 'n_timing_outliers', 'n_amplitude_outliers', 'n_polarity_outliers', 'n_at_bound']
    td = df[[c for c in drift_cols if c in df]].copy()
    if 'slope_fs_per_s' in td:
        td['slope_fs_per_min'] = td['slope_fs_per_s'] * 60
        td['trend_over_group_um'] = td['trend_over_group_fs'] * factor / 1000
        td['interpretation'] = 'descriptive within-group trend (stage commanded static); not an independent ECOPS drift estimate'
    td.to_csv(out / 'timing_drift_analysis.csv', index=False)

    # ---- Plots ----
    make_plots(out, t, df, ref, tr, averages, cfg, factor, zoom)

    # ---- Summary ----
    warn_prec = core.precision_assessment(df, ana['summary'], target, factor)
    summary = {
        'alignment_method': 'gaussian_median_integer',
        'data_dir': str(base), 'positions': int(len(df)), 'trace_count': int(len(tr)),
        'valid_traces': int(tr['valid'].sum()),
        'conversion': {'incidence_angle_deg': float(cfg['incidence_angle_deg']),
                       'um_per_ps': factor, 'fs_per_um': 1000 / factor,
                       'delay_sign_for_positive_z': int(cfg['analysis']['delay_sign_for_positive_z']),
                       'stage_unit': cfg['stage_unit'], 'time_unit': cfg['time_unit']},
        'baseline': {'baseline_correction': bool(cfg['preprocessing'].get('baseline_correction', True)),
                     'baseline_samples': int(cfg['preprocessing'].get('baseline_samples', 30)),
                     'median_baseline_level': float(df['baseline_level_median'].median()),
                     'max_abs_effect_on_group_median_fs': float(df['baseline_effect_on_median_fs'].abs().max())
                     if 'baseline_effect_on_median_fs' in df else None},
        'jitter': {'median_within_group_sd_fs': float(df['jitter_std_fs'].median()),
                   'median_within_group_sd_detrended_fs': float(df['sd_detrended_fs'].median()) if 'sd_detrended_fs' in df else None,
                   'median_se_group_median_um': float(df['se_displacement_um'].median())},
        'reference_returns': ana['summary'],
        'drift_diagnostics': drift,
        'estimator_cross_check_max_abs_um': float(df['gauss_minus_zero_crossing_um'].abs().max()),
        'precision_target_um': target,
        'precision_warnings': warn_prec,
        'data_issues': issues,
        'notes': ['Apparent displacement = optical path change seen by the THz pulse; it combines stage '
                  'motion, ECOPS time-base drift and any other optical-path drift.',
                  'Within-group slopes are measured while the stage is commanded static; they are '
                  'descriptive and do not by themselves identify the source.',
                  'drift_corrected_* columns assume the timing baseline varies linearly between '
                  'neighbouring reference groups; they are NOT a measurement of stage accuracy.',
                  'The first reference group defines zero; it is not an absolute position standard.',
                  'Gaussian fit of the main lobe of a bipolar pulse has a shape-dependent offset that '
                  'cancels only if the pulse shape is unchanged; see gauss_minus_zero_crossing_um.']}
    (out / 'summary.yaml').write_text(yaml.safe_dump(to_builtin(summary), sort_keys=False, allow_unicode=True), encoding='utf8')
    print('\n'.join(['', 'WARNINGS:'] + [' - ' + w for w in warn_prec + issues]) if warn_prec or issues else '')
    print('Saved results to', out)
    return out


def make_plots(out, t, df, ref, tr, averages, cfg, factor, zoom):
    unit = cfg['stage_unit']
    zt = lambda ax, d: z_ticks(ax, d['measurement_index'], d['z_command'], unit)
    has_time = 'elapsed_s' in tr and tr['elapsed_s'].notna().any()
    v = tr[tr['valid']]

    # 01 raw vs corrected averages (per group)
    for idx, (raw_avg, corr, z) in averages.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, raw_avg, label='Raw average (all traces)', alpha=.7)
        ax.plot(t, corr, label='Gaussian-median integer corrected (valid traces)', alpha=.8)
        ax.set(xlabel='Time (ps)', ylabel='Signal (scaled)', title=f'Z = {float(z):.4f} {cfg["stage_unit"]} (group {idx})')
        ax.legend(); ax.grid(alpha=.25)
        save_with_zoom(fig, ax, out / f'01_waveform_{idx:03d}.png', zoom, 150)

    # 02 arrival time per trace (relative to group median), outliers marked
    n = len(df); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), sharey=True, squeeze=False)
    for ax, (_, g) in zip(axes.flat, df.iterrows()):
        s = tr[tr['measurement_index'] == g['measurement_index']]
        ax.plot(s['trace_number'], s['jitter_from_group_median_fs'], '.', ms=2, alpha=.5)
        bad = s[~s['valid']]
        if len(bad): ax.plot(bad['trace_number'], bad['jitter_from_group_median_fs'], 'rx', ms=4, label='rejected')
        ax.set_title(f'Z = {g.z_command:.4f} ({g.measurement_index})', fontsize=9); ax.grid(alpha=.25)
    for ax in axes.flat[n:]: ax.axis('off')
    fig.supxlabel('Trace number'); fig.supylabel('t0 − group median (fs)')
    fig.suptitle('Gaussian-fitted arrival time of individual traces')
    fig.tight_layout(); fig.savefig(out / '02_trace_arrival_times.png', dpi=150); plt.close(fig)

    if has_time:
        # 03 timing vs acquisition time across the whole experiment
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sc = ax.scatter(v['elapsed_s'] / 60, v['timing_offset_from_commanded_fs'], c=v['measurement_index'],
                        s=2, cmap='tab20', alpha=.6)
        ax.plot(df['elapsed_mid_s'] / 60, (df['relative_delay_fs'] - int(cfg['analysis']['delay_sign_for_positive_z'])
                * df['expected_um'] / factor * 1000), 'k^', ms=6, label='group median')
        ax.set(xlabel='Elapsed time (min)', ylabel='Arrival − nominal commanded delay (fs)',
               title='Timing vs acquisition time (commanded displacement removed with nominal c/2 conversion)')
        sec = ax.secondary_yaxis('right', functions=(lambda x: x * factor / 1000, lambda x: x / factor * 1000))
        sec.set_ylabel('µm equivalent'); ax.grid(alpha=.25); ax.legend(loc='upper left')
        cb = fig.colorbar(sc, ax=ax, pad=.08, label=f'Commanded Z ({unit})')
        cb.set_ticks(list(df['measurement_index']))
        cb.set_ticklabels([f'{i}: {z:.4f}' for i, z in zip(df['measurement_index'], df['z_command'])])
        cb.ax.tick_params(labelsize=7)
        fig.tight_layout(); fig.savefig(out / '03_timing_vs_time.png', dpi=160); plt.close(fig)

        # 04 within-group trends
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), sharey=True, squeeze=False)
        for ax, (_, g) in zip(axes.flat, df.iterrows()):
            s = v[v['measurement_index'] == g['measurement_index']]
            x = s['elapsed_s'] - s['elapsed_s'].min()
            ax.plot(x, s['jitter_from_group_median_fs'], '.', ms=2, alpha=.4)
            if np.isfinite(g.slope_fs_per_s):
                yfit = s['jitter_from_group_median_fs'] - s['detrended_residual_fs']
                ax.plot(x, yfit, 'r-', lw=1.5)
            ax.set_title(f'Z = {g.z_command:.4f}: {g.slope_fs_per_s:+.2f} fs/s '
                         f'({g.trend_over_group_fs * factor / 1000:+.1f} µm/group)', fontsize=9)
            ax.grid(alpha=.25)
        for ax in axes.flat[n:]: ax.axis('off')
        fig.supxlabel('Time within group (s)'); fig.supylabel('t0 − group median (fs)')
        fig.suptitle('Within-group timing trend (Theil–Sen), stage commanded static')
        fig.tight_layout(); fig.savefig(out / '04_within_group_trends.png', dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        yerr = np.vstack([df.slope_fs_per_s - df.slope_ci_low_fs_per_s, df.slope_ci_high_fs_per_s - df.slope_fs_per_s])
        ax.errorbar(df.measurement_index, df.slope_fs_per_s, yerr=yerr, fmt='o', capsize=3, label='within group (95% CI)')
        if len(ref) >= 2:
            sl = theilslopes(ref['relative_delay_fs'], ref['elapsed_mid_s'])[0]
            ax.axhline(sl, color='r', ls='--', label=f'between reference returns: {sl:.2f} fs/s')
        ax.axhline(0, color='gray', lw=1)
        ax.set(ylabel='Timing slope (fs/s)', title='Within-group vs between-return timing drift rate'); zt(ax, df)
        ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(out / '04b_trend_slopes.png', dpi=160); plt.close(fig)

    # 05 reference returns
    fig, axes = plt.subplots(1, 2 if has_time else 1, figsize=(12 if has_time else 7, 4.2), squeeze=False)
    markers = {'increasing_Z': ('^', 'tab:blue'), 'decreasing_Z': ('v', 'tab:orange'), 'initial': ('s', 'k')}
    for ax, xcol, xl in zip(axes.flat, ['measurement_index', 'elapsed_mid_s'], [None, 'Elapsed time (min)']):
        xs = ref[xcol] / (60 if xcol == 'elapsed_mid_s' else 1)
        ax.plot(xs, ref['apparent_displacement_um'], '-', color='gray', lw=1)
        for d, g in ref.groupby(ref.get('approach_direction', pd.Series('n/a', index=ref.index))):
            m, c = markers.get(d, ('o', 'tab:green'))
            ax.errorbar(g[xcol] / (60 if xcol == 'elapsed_mid_s' else 1), g['apparent_displacement_um'],
                        yerr=g['se_displacement_um'], fmt=m, color=c, ms=7, capsize=3, label=str(d))
        ax.axhline(0, color='gray', ls='--', lw=1); ax.grid(alpha=.25)
        ax.set(ylabel='Apparent displacement (µm)')
        if xl: ax.set_xlabel(xl)
        else:
            z_ticks(ax, ref['measurement_index'], ref['z_command'], unit)
            if 'previous_z_command' in ref:  # all returns share Z; show where each came from
                ax.set_xticklabels(['initial' if i == ref['measurement_index'].iloc[0] else f'← {p:.4f}'
                                    for i, p in zip(ref['measurement_index'], ref['previous_z_command'])])
                ax.set_xlabel(f'Return to Z = {float(cfg["z_reference"]):.4f} {unit}, coming from')
    axes.flat[0].legend(title='approach'); fig.suptitle(f'Reference returns to Z = {cfg["z_reference"]} (uncorrected)')
    fig.tight_layout(); fig.savefig(out / '05_reference_returns.png', dpi=160); plt.close(fig)

    # 06 apparent vs commanded displacement
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(df.expected_um, df.apparent_displacement_um, 'o')
    lim = [df.expected_um.min() * 1.05, df.expected_um.max() * 1.05]
    a1.plot(lim, lim, 'k--', lw=1); a1.grid(alpha=.25)
    a1.set(xlabel='Commanded displacement (µm)', ylabel='Apparent displacement (µm)', title='Identity check (uncorrected)')
    a2.plot(df.measurement_index, df.apparent_error_um, 'o-', label='uncorrected')
    a2.axhline(0, color='gray', ls='--', lw=1); a2.grid(alpha=.25); a2.legend()
    a2.set(ylabel='Apparent − commanded (µm)', title='Apparent error (stage + timing, not separated)'); zt(a2, df)
    fig.tight_layout(); fig.savefig(out / '06_displacement_vs_commanded.png', dpi=160); plt.close(fig)

    # 07 jitter SD by group
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.measurement_index, df.jitter_std_fs, 'o-', label='SD (raw)')
    if 'sd_detrended_fs' in df: ax.plot(df.measurement_index, df.sd_detrended_fs, 's-', label='SD after removing linear trend')
    ax.plot(df.measurement_index, df.jitter_mad_fs, 'd-', label='1.4826·MAD')
    sec = ax.secondary_yaxis('right', functions=(lambda x: x * factor / 1000, lambda x: x / factor * 1000))
    sec.set_ylabel('µm equivalent (single trace)')
    ax.set(ylabel='Within-group timing spread (fs)', title='Per-trace timing jitter by group'); zt(ax, df)
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / '07_jitter_std.png', dpi=160); plt.close(fig)

    # 08 baseline sensitivity
    if 'baseline_effect_on_median_fs' in df:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(df.measurement_index, df.baseline_effect_on_median_fs, 'o-')
        a1.set(ylabel='Δ median t0 (fs)', title='Baseline ON − OFF: effect on group arrival'); zt(a1, df)
        a1.grid(alpha=.25)
        xcol = 'elapsed_s' if has_time else 'trace_number'
        a2.plot(tr[xcol] / (60 if has_time else 1), tr['baseline_level'], '.', ms=1.5, alpha=.5)
        a2.set(xlabel='Elapsed time (min)' if has_time else 'Trace', ylabel='Baseline level (scaled)',
               title=f'Median of first {cfg["preprocessing"].get("baseline_samples", 30)} samples per trace')
        a2.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / '08_baseline_sensitivity.png', dpi=160); plt.close(fig)

    # 09 assumption-based drift correction (clearly separated)
    if 'drift_corrected_error_um' in df:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df.measurement_index, df.apparent_error_um, 'o-', color='gray', label='UNCORRECTED apparent error')
        nr = df[~df.is_reference]
        ax.plot(nr.measurement_index, nr.drift_corrected_error_um, 's', color='tab:red', ms=8,
                label='ASSUMPTION: linear timing drift between neighbouring references')
        for _, r in nr.iterrows():
            if np.isfinite(r.drift_corrected_error_um):
                ax.annotate(f'{r.drift_corrected_error_um:+.1f}', (r.measurement_index, r.drift_corrected_error_um),
                            textcoords='offset points', xytext=(5, 5), fontsize=8, color='tab:red')
        ax.axhline(0, color='gray', ls='--', lw=1)
        zt(ax, df)
        ax.set(ylabel='Displacement error (µm)',
               title='Assumption-based drift correction — NOT a measurement of stage accuracy')
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(out / '09_drift_corrected_ASSUMPTION.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    args = parser.parse_args()
    main(args.config.resolve())
