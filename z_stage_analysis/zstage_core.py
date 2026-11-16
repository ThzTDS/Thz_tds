"""Core, testable functions for the Gaussian-median-integer Z-stage analysis.

Everything here works on plain arrays so that the synthetic tests exercise the
same code path as analyze_gaussian_jitter.py. No cross-correlation is used.

Conventions
-----------
* Time in ps, stage coordinates converted to µm via ``stage_um``.
* ``arrival_ps`` is the Gaussian-fitted pulse centre t0 of the strongest lobe.
* Positive delay = pulse arrives later. With ``delay_sign_for_positive_z=+1``,
  increasing Z lengthens the optical path (later arrival).
* Displacement at incidence angle θ (measured from the surface normal):
  Δz = c·Δt / (2·cos θ);  1 µm ↔ 6.671 fs round trip at normal incidence.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from thz_tds.dataset import THZDataset

C_UM_PER_PS = 299.792458


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def stage_um(v, cfg) -> float:
    unit = str(cfg['stage_unit']).lower()
    if unit == 'mm':
        return float(v) * 1000
    if unit in ('um', 'µm'):
        return float(v)
    if unit == 'native':
        return float(v) * 1000 / float(cfg['stage_units_per_mm'])
    raise ValueError('stage_unit must be mm, um or native')


def time_ps(t, cfg) -> np.ndarray:
    factors = {'ps': 1., 'fs': 1e-3, 's': 1e12}
    unit = str(cfg['time_unit']).lower()
    if unit not in factors:
        raise ValueError('time_unit must be ps, fs or s')
    return np.asarray(t, float) * factors[unit]


def um_per_ps(incidence_angle_deg: float) -> float:
    """Displacement per ps of round-trip delay: c / (2 cos θ)."""
    cos = np.cos(np.deg2rad(float(incidence_angle_deg)))
    if not np.isfinite(cos) or cos <= 0.05:
        raise ValueError('incidence_angle_deg invalid for the reflection model')
    return C_UM_PER_PS / (2 * cos)


def uniform_dt(t: np.ndarray, tol_fraction: float = 0.01) -> float:
    """Sample interval from a linear fit; exported time values are rounded,
    so per-step differences are not a valid uniformity test."""
    n = np.arange(len(t))
    dt, t_start = np.polyfit(n, t, 1)
    if np.max(np.abs(t - (t_start + n * dt))) > tol_fraction * dt:
        raise ValueError('Time axis must be uniform for integer-sample shifts')
    return float(dt)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def baseline_levels(raw: np.ndarray, n: int) -> np.ndarray:
    """Median of the first n samples of each trace (0 if n <= 0)."""
    raw = np.asarray(raw, float)
    if n <= 0:
        return np.zeros(raw.shape[0])
    if n >= raw.shape[-1]:
        raise ValueError('baseline_samples must be smaller than waveform length')
    return np.median(raw[:, :n], axis=1)


def baseline_region_check(t: np.ndarray, y_avg: np.ndarray, n: int,
                          pulse_t_ps: float, warn_ratio: float = 3.0,
                          min_gap_ps: float = 5.0) -> Dict:
    """Check that the first n samples are pulse-free.

    Compares the RMS of the baseline window (after removing its median) with the
    RMS of a pulse-free reference region (n .. pulse - min_gap_ps). Also fails if
    the window reaches within min_gap_ps of the fitted pulse.
    """
    out = {'baseline_window_end_ps': float(t[max(n - 1, 0)]) if n > 0 else np.nan,
           'baseline_rms': np.nan, 'reference_rms': np.nan, 'baseline_rms_ratio': np.nan,
           'baseline_window_ok': True}
    if n <= 0:
        return out
    win = y_avg[:n] - np.median(y_avg[:n])
    ref_mask = (np.arange(len(t)) >= n) & (t < pulse_t_ps - min_gap_ps)
    out['baseline_rms'] = float(np.sqrt(np.mean(win ** 2)))
    if ref_mask.sum() >= 10:
        ref = y_avg[ref_mask] - np.median(y_avg[ref_mask])
        out['reference_rms'] = float(np.sqrt(np.mean(ref ** 2)))
        if out['reference_rms'] > 0:
            out['baseline_rms_ratio'] = out['baseline_rms'] / out['reference_rms']
    too_close = t[n - 1] > pulse_t_ps - min_gap_ps
    too_noisy = np.isfinite(out['baseline_rms_ratio']) and out['baseline_rms_ratio'] > warn_ratio
    out['baseline_window_ok'] = not (too_close or too_noisy)
    return out


# ---------------------------------------------------------------------------
# Gaussian fitting (THZDataset implementation)
# ---------------------------------------------------------------------------

def configure_fitter(ds, fit_cfg: Dict):
    """Apply YAML gaussian_fit settings to a THZDataset instance."""
    ds.adaptive_fit_window = bool(fit_cfg.get('adaptive_fit_window', True))
    ds.fit_window_sigma_factor = float(fit_cfg.get('fit_window_sigma_factor', 5.0))
    ds.fit_window_ps = float(fit_cfg.get('fit_window_ps', 8.0))
    ds.fallback_fit_window_ps = ds.fit_window_ps
    ds.min_sigma_ps = float(fit_cfg.get('min_sigma_ps', 0.05))
    ds.max_sigma_ps = float(fit_cfg.get('max_sigma_ps', 10.0))
    ds.min_fit_points = int(fit_cfg.get('min_fit_points', 10))
    ds.max_center_shift_ps = float(fit_cfg.get('max_center_shift_ps', 5.0))
    return ds


def gaussian_fitter(fit_cfg: Dict):
    """A THZDataset used only for its fit_gaussian_to_trace method (no data)."""
    return configure_fitter(THZDataset.__new__(THZDataset), fit_cfg)


def fit_arrivals(fitter, t: np.ndarray, traces: np.ndarray) -> pd.DataFrame:
    """Gaussian fit of every trace. A fit that converges onto a parameter bound
    is kept but flagged (at_bound) because its t0 is not a free estimate."""
    rows = []
    for y in traces:
        r = {'arrival_ps': np.nan, 'sigma_ps': np.nan, 'amplitude': np.nan,
             'offset': np.nan, 'fit_ok': False, 'at_bound': False, 'fit_error': ''}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # OptimizeWarning: covariance not estimated
                fit = fitter.fit_gaussian_to_trace(t, y)
            r.update(arrival_ps=fit['t0_ps'], sigma_ps=fit['sigma_ps'], amplitude=fit['A'],
                     offset=fit['C'], fit_ok=bool(np.isfinite(fit['t0_ps'])))
            t_lo = max(fit['x0_ini_ps'] - fitter.max_center_shift_ps, fit['fit_window_left_ps'])
            t_hi = min(fit['x0_ini_ps'] + fitter.max_center_shift_ps, fit['fit_window_right_ps'])
            eps = 1e-6
            r['at_bound'] = bool(abs(fit['t0_ps'] - t_lo) < eps or abs(fit['t0_ps'] - t_hi) < eps
                                 or fit['sigma_ps'] <= fitter.min_sigma_ps * (1 + eps)
                                 or fit['sigma_ps'] >= fitter.max_sigma_ps * (1 - eps))
        except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
            r['fit_error'] = type(exc).__name__
        rows.append(r)
    return pd.DataFrame(rows)


def zero_crossing_time(t: np.ndarray, y: np.ndarray) -> float:
    """Model-free timing marker: zero crossing between the main positive and
    negative lobes (linear interpolation). Diagnostic only - not used to align.
    Requires a baseline-corrected trace."""
    ip, im = int(np.argmax(y)), int(np.argmin(y))
    lo, hi = sorted((ip, im))
    seg = y[lo:hi + 1]
    s = np.flatnonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))
    if not len(s):
        return np.nan
    k = lo + int(s[0])
    return float(t[k] - y[k] * (t[k + 1] - t[k]) / (y[k + 1] - y[k]))


def robust_outliers(values: np.ndarray, usable: np.ndarray, k: float) -> np.ndarray:
    """|x - median| > k * 1.4826 * MAD among usable values."""
    out = np.zeros(len(values), bool)
    v = values[usable]
    if len(v) < 3 or not k:
        return out
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    if mad > 0:
        out[usable] = np.abs(v - med) > k * mad
    return out


# ---------------------------------------------------------------------------
# Integer-sample alignment
# ---------------------------------------------------------------------------

def shift_integer(y: np.ndarray, lag: int, fill=np.nan) -> np.ndarray:
    """Return z with z[n] = y[n + lag]: lag > 0 moves the waveform EARLIER.
    Vacated samples are filled with `fill` (NaN by default, so padding never
    enters an average as a fake zero signal)."""
    y = np.asarray(y, float)
    z = np.full_like(y, fill)
    n = len(y)
    if lag == 0:
        z[:] = y
    elif 0 < lag < n:
        z[:-lag] = y[lag:]
    elif -n < lag < 0:
        z[-lag:] = y[:lag]
    return z


def integer_align_average(traces: np.ndarray, centers: np.ndarray, valid: np.ndarray,
                          dt: float, reference: float):
    """Shift each valid trace by round((t0_i - reference)/dt) samples toward the
    reference and average. Returns (average, lags, applied_shift_ps, residual_ps).

    The reference (group median) itself is not changed, so the group arrival
    time is preserved; only within-group spread is reduced.
    """
    n_tr, n_s = traces.shape
    total = np.zeros(n_s)
    count = np.zeros(n_s)
    lags = np.zeros(n_tr, int)
    shift_ps = np.full(n_tr, np.nan)
    resid = np.full(n_tr, np.nan)
    for j in np.flatnonzero(valid):
        lag = int(np.rint((centers[j] - reference) / dt))
        lags[j] = lag
        shift_ps[j] = -lag * dt
        resid[j] = centers[j] - lag * dt - reference
        z = shift_integer(traces[j], lag)
        ok = np.isfinite(z)
        total[ok] += z[ok]
        count[ok] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return avg, lags, shift_ps, resid


# ---------------------------------------------------------------------------
# Timestamps and trends
# ---------------------------------------------------------------------------

def interval_check(elapsed_s: np.ndarray, factor: float = 10.0) -> Dict:
    """Acquisition intervals; an interval > factor x median is 'irregular'
    (e.g. the 10 s buffer clears every 180 traces)."""
    e = np.asarray(elapsed_s, float)
    out = {'n_missing_timestamps': int(np.sum(~np.isfinite(e))), 'median_interval_s': np.nan,
           'max_interval_s': np.nan, 'n_irregular_intervals': 0, 'irregular_after_traces': [],
           'n_nonmonotonic': 0}
    d = np.diff(e)
    good = np.isfinite(d)
    if good.sum() == 0:
        return out
    med = float(np.median(d[good]))
    out.update(median_interval_s=med, max_interval_s=float(np.max(d[good])),
               n_nonmonotonic=int(np.sum(d[good] <= 0)))
    irr = np.flatnonzero(good & (d > factor * med))
    out['n_irregular_intervals'] = int(len(irr))
    out['irregular_after_traces'] = [int(i + 1) for i in irr]  # 1-based trace number before gap
    return out


def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


def effective_n(n: int, rho: float) -> float:
    """AR(1) effective sample size n(1-ρ)/(1+ρ) (ρ<=0 → n)."""
    if not np.isfinite(rho) or rho <= 0:
        return float(n)
    return float(max(1.0, n * (1 - rho) / (1 + rho)))


def timing_trend(elapsed_s: np.ndarray, arrival_ps: np.ndarray) -> Dict:
    """Robust (Theil-Sen) linear trend of arrival time vs acquisition time.

    Descriptive only: a within-group slope says the measured timing changed
    while the stage was commanded static. It cannot tell whether the ECOPS
    time base, the optical path (stage creep, thermal expansion) or both moved.
    """
    x = np.asarray(elapsed_s, float)
    y = np.asarray(arrival_ps, float) * 1000  # fs
    ok = np.isfinite(x) & np.isfinite(y)
    out = {k: np.nan for k in ('slope_fs_per_s', 'slope_ci_low_fs_per_s', 'slope_ci_high_fs_per_s',
                               'trend_over_group_fs', 'sd_raw_fs', 'sd_detrended_fs',
                               'lag1_autocorr_detrended', 'n_eff', 'duration_s')}
    out['residual_fs'] = np.full(len(x), np.nan)
    if ok.sum() < 5:
        return out
    xs, ys = x[ok], y[ok]
    slope, icpt, lo, hi = theilslopes(ys, xs, 0.95)
    res = ys - (icpt + slope * xs)
    rho = lag1_autocorr(res)
    out.update(slope_fs_per_s=float(slope), slope_ci_low_fs_per_s=float(lo),
               slope_ci_high_fs_per_s=float(hi), duration_s=float(xs.max() - xs.min()),
               trend_over_group_fs=float(slope * (xs.max() - xs.min())),
               sd_raw_fs=float(np.std(ys, ddof=1)), sd_detrended_fs=float(np.std(res, ddof=1)),
               lag1_autocorr_detrended=rho, n_eff=effective_n(int(ok.sum()), rho))
    out['residual_fs'][ok] = res
    return out


# ---------------------------------------------------------------------------
# Group processing
# ---------------------------------------------------------------------------

def process_group(t: np.ndarray, raw: np.ndarray, fitter, cfg: Dict,
                  elapsed_s: Optional[np.ndarray] = None) -> Dict:
    """Gaussian-median-integer processing of one stationary group.

    Returns dict with 'traces' (DataFrame), 'stats' (dict), 'raw_avg',
    'corrected_avg'. raw: (n_traces, n_samples), amplitude-scaled, unaligned.
    """
    pre = cfg.get('preprocessing', {})
    fit_cfg = cfg.get('gaussian_fit', {})
    dt = uniform_dt(t)
    n_base = int(pre.get('baseline_samples', 30))
    use_bc = bool(pre.get('baseline_correction', True))
    levels = baseline_levels(raw, n_base)
    y_bc = raw - levels[:, None]
    y = y_bc if use_bc else raw

    fits = fit_arrivals(fitter, t, y)
    centers = fits['arrival_ps'].to_numpy()
    ok = fits['fit_ok'].to_numpy()

    # Quality control: bound hits, timing outliers, amplitude outliers.
    reject_bound = bool(fit_cfg.get('reject_at_bounds', True))
    bound = fits['at_bound'].to_numpy() & reject_bound
    usable = ok & ~bound
    # Lobe hopping: with similar positive/negative lobes the strongest |lobe| can
    # switch between traces (~0.9 ps jump). Keep only the majority polarity.
    sign = np.sign(fits['amplitude'].to_numpy())
    major = np.sign(np.sum(sign[usable])) or 1.0
    pol_out = usable & (sign != major)
    usable = usable & ~pol_out
    amp = np.abs(fits['amplitude'].to_numpy())
    amp_out = np.zeros(len(amp), bool)
    max_dev = fit_cfg.get('max_amplitude_deviation', 0.5)
    if max_dev is not None and usable.sum():
        med_a = np.median(amp[usable])
        amp_out = usable & (np.abs(amp - med_a) > float(max_dev) * med_a)
    k_mad = fit_cfg.get('outlier_mad_threshold', 6.0)
    t_out = robust_outliers(centers, usable & ~amp_out, float(k_mad) if k_mad else 0)
    valid = usable & ~t_out & ~amp_out

    if valid.sum() < 2:
        raise RuntimeError('fewer than 2 valid Gaussian fits')

    median = float(np.median(centers[valid]))
    corrected, lags, shift_ps, resid = integer_align_average(y, centers, valid, dt, median)
    raw_avg = np.mean(y, axis=0)
    jitter = centers[valid] - median

    traces = fits.copy()
    traces.insert(0, 'trace_index', np.arange(len(raw)))
    traces['baseline_level'] = levels
    traces['valid'] = valid
    traces['timing_outlier'] = t_out
    traces['amplitude_outlier'] = amp_out
    traces['polarity_outlier'] = pol_out
    traces['jitter_from_group_median_fs'] = (centers - median) * 1000
    traces['integer_lag_samples'] = np.where(valid, lags, 0)
    traces['applied_integer_shift_ps'] = shift_ps
    traces['residual_after_integer_shift_fs'] = resid * 1000
    traces['zero_crossing_ps'] = [zero_crossing_time(t, yy) for yy in y_bc]

    stats = {
        'dt_ps': dt, 'n_traces': int(len(raw)), 'n_fit_ok': int(ok.sum()),
        'n_at_bound': int(fits['at_bound'].sum()), 'n_timing_outliers': int(t_out.sum()),
        'n_amplitude_outliers': int(amp_out.sum()), 'n_polarity_outliers': int(pol_out.sum()),
        'n_valid': int(valid.sum()),
        'median_arrival_ps': median,
        'jitter_std_fs': float(np.std(jitter, ddof=1) * 1000),
        'jitter_mad_fs': float(1.4826 * np.median(np.abs(jitter)) * 1000),
        'jitter_p95_abs_fs': float(np.percentile(np.abs(jitter), 95) * 1000),
        # After integer shifts each trace is within ±dt/2 of the median, so this
        # is ~dt/sqrt(12); it is the alignment quantisation, not physical jitter.
        'residual_after_integer_shift_std_fs': float(np.nanstd(resid[valid], ddof=1) * 1000),
        'median_sigma_ps': float(np.median(fits['sigma_ps'].to_numpy()[valid])),
        'median_amplitude': float(np.median(amp[valid])),
        'baseline_level_median': float(np.median(levels)),
        'baseline_level_std': float(np.std(levels, ddof=1)) if len(levels) > 1 else np.nan,
        'zero_crossing_median_ps': float(np.nanmedian(traces['zero_crossing_ps'].to_numpy()[valid])),
    }
    try:
        stats['corrected_average_peak_ps'] = float(
            fitter.fit_gaussian_to_trace(t, np.nan_to_num(corrected, nan=float(np.nanmedian(corrected))))['t0_ps'])
    except (ValueError, RuntimeError):
        stats['corrected_average_peak_ps'] = np.nan

    stats.update(baseline_region_check(t, np.mean(y_bc, axis=0), n_base, median,
                                       float(pre.get('baseline_warn_ratio', 3.0))))

    # Sensitivity: refit with the opposite baseline choice.
    if pre.get('baseline_sensitivity', True):
        alt = fit_arrivals(fitter, t, raw if use_bc else y_bc)['arrival_ps'].to_numpy()
        both = valid & np.isfinite(alt)
        traces['arrival_other_baseline_ps'] = alt
        stats['median_arrival_other_baseline_ps'] = float(np.median(alt[both])) if both.any() else np.nan
        stats['baseline_effect_on_median_fs'] = (median - stats['median_arrival_other_baseline_ps']) * 1000 \
            * (1 if use_bc else -1)  # (with baseline) - (without baseline)
        stats['baseline_effect_max_abs_trace_fs'] = float(np.max(np.abs(centers[both] - alt[both])) * 1000) if both.any() else np.nan

    # Timestamp-aware trend.
    if elapsed_s is not None:
        e = np.asarray(elapsed_s, float)
        traces['elapsed_s'] = e
        tr = timing_trend(e[valid], centers[valid])
        res = np.full(len(raw), np.nan)
        res[valid] = tr.pop('residual_fs')
        traces['detrended_residual_fs'] = res
        stats.update(tr)
        stats.update({k: v for k, v in interval_check(e, float(cfg.get('timing', {}).get(
            'interval_outlier_factor', 10.0))).items()})
        stats['elapsed_mid_s'] = float(np.nanmedian(e[valid]))
        stats['elapsed_start_s'] = float(np.nanmin(e))
        stats['elapsed_end_s'] = float(np.nanmax(e))
        rho = stats.get('lag1_autocorr_detrended', np.nan)
        neff = stats.get('n_eff', np.nan)
        sd = stats.get('sd_detrended_fs', np.nan)
        stats['se_median_fs'] = float(1.2533 * sd / np.sqrt(neff)) if np.isfinite(neff) and neff > 0 else np.nan
    else:
        stats['se_median_fs'] = float(1.2533 * stats['jitter_std_fs'] / np.sqrt(stats['n_valid']))

    return {'traces': traces, 'stats': stats, 'raw_avg': raw_avg, 'corrected_avg': corrected}


# ---------------------------------------------------------------------------
# Across-group analysis
# ---------------------------------------------------------------------------

def interpolate_reference_timing(times: np.ndarray, ref_times: np.ndarray,
                                 ref_delays: np.ndarray) -> np.ndarray:
    """ASSUMPTION-BASED local timing baseline: linear interpolation in time
    between the neighbouring reference groups. NaN outside the reference span."""
    order = np.argsort(ref_times)
    rt, rd = np.asarray(ref_times)[order], np.asarray(ref_delays)[order]
    out = np.interp(times, rt, rd, left=np.nan, right=np.nan)
    return out


def analyze_positions(groups: pd.DataFrame, cfg: Dict) -> Dict:
    """Relative delay, apparent displacement, reference-return statistics,
    between-group drift, optional assumption-based drift correction."""
    df = groups.sort_values('measurement_index').reset_index(drop=True).copy()
    factor = um_per_ps(cfg['incidence_angle_deg'])
    orient = int(cfg['analysis']['delay_sign_for_positive_z'])
    if orient not in (-1, 1):
        raise ValueError('delay_sign_for_positive_z must be +1 or -1')
    refz = float(cfg['z_reference'])
    tol_um = stage_um(float(cfg.get('reference', {}).get('z_tolerance', 1e-4)), cfg)

    df['expected_um'] = [stage_um(z - refz, cfg) for z in df['z_command']]
    df['is_reference'] = np.abs(df['expected_um']) <= tol_um
    if not df['is_reference'].iloc[0]:
        warnings.warn('First group is not at z_reference; zero delay is defined by a non-reference group.')
    tzero = float(df['median_arrival_ps'].iloc[0])
    df['relative_delay_fs'] = (df['median_arrival_ps'] - tzero) * 1000
    df['apparent_displacement_um'] = orient * factor * df['relative_delay_fs'] / 1000
    df['apparent_error_um'] = df['apparent_displacement_um'] - df['expected_um']
    df['se_displacement_um'] = factor * df['se_median_fs'] / 1000
    df['delta_delay_from_previous_fs'] = df['relative_delay_fs'].diff()
    df['delta_commanded_um'] = df['expected_um'].diff()
    df['apparent_incremental_error_um'] = orient * factor * df['delta_delay_from_previous_fs'] / 1000 \
        - df['delta_commanded_um']
    # Model-free cross-check of the Gaussian timing (zero crossing).
    zc0 = float(df['zero_crossing_median_ps'].iloc[0])
    df['zero_crossing_displacement_um'] = orient * factor * (df['zero_crossing_median_ps'] - zc0)
    df['gauss_minus_zero_crossing_um'] = df['apparent_displacement_um'] - df['zero_crossing_displacement_um']

    # Sign check (do not change the configured sign; only warn).
    notes = []
    big = df[np.abs(df['expected_um']) >= 50]
    if len(big):
        agree = np.sign(big['apparent_displacement_um']) == np.sign(big['expected_um'])
        if agree.mean() < 0.5:
            notes.append('Measured delay sign disagrees with commanded Z for most excursions: '
                         'check delay_sign_for_positive_z (+1 means larger Z = later pulse).')

    ref = df[df['is_reference']].copy()
    has_time = 'elapsed_mid_s' in df and df['elapsed_mid_s'].notna().all()

    # Leave-one-out: predict each interior return from its neighbouring returns.
    ref['loo_residual_um'] = np.nan
    if has_time and len(ref) >= 3:
        for i in range(1, len(ref) - 1):
            r0, r1 = ref.iloc[i - 1], ref.iloc[i + 1]
            pred = r0.apparent_displacement_um + (r1.apparent_displacement_um - r0.apparent_displacement_um) * \
                (ref.iloc[i].elapsed_mid_s - r0.elapsed_mid_s) / (r1.elapsed_mid_s - r0.elapsed_mid_s)
            ref.iloc[i, ref.columns.get_loc('loo_residual_um')] = ref.iloc[i].apparent_displacement_um - pred

    returns = ref.iloc[1:]  # the first reference defines zero
    vals = returns['apparent_displacement_um']
    allv = ref['apparent_displacement_um']
    summary = {
        'reference_group_count': int(len(ref)),
        'return_count_excluding_initial': int(len(returns)),
        'return_mean_um': float(vals.mean()) if len(vals) else None,
        'return_std_um': float(vals.std(ddof=1)) if len(vals) > 1 else None,
        'return_max_abs_um': float(vals.abs().max()) if len(vals) else None,
        'reference_peak_to_peak_um': float(allv.max() - allv.min()) if len(allv) else None,
        'loo_residual_std_um': float(ref['loo_residual_um'].std(ddof=1)) if ref['loo_residual_um'].notna().sum() > 1 else None,
        'loo_residual_max_abs_um': float(ref['loo_residual_um'].abs().max()) if ref['loo_residual_um'].notna().any() else None,
    }
    if 'approach_direction' in ref:
        by_dir = {}
        for d, g in returns.groupby('approach_direction'):
            v, l = g['apparent_displacement_um'], g['loo_residual_um'].dropna()
            by_dir[str(d)] = {'n': int(len(g)), 'mean_um': float(v.mean()),
                              'loo_residual_mean_um': float(l.mean()) if len(l) else None}
        summary['returns_by_approach_direction'] = by_dir

    drift = {}
    if has_time and len(ref) >= 2:
        sl, ic, lo, hi = theilslopes(ref['relative_delay_fs'], ref['elapsed_mid_s'], 0.95)
        drift = {'between_reference_slope_fs_per_s': float(sl),
                 'between_reference_slope_ci_fs_per_s': [float(lo), float(hi)],
                 'between_reference_slope_um_per_min': float(sl * 60 * factor / 1000),
                 'median_within_group_slope_fs_per_s': float(df['slope_fs_per_s'].median()),
                 'groups_with_positive_within_slope': int((df['slope_fs_per_s'] > 0).sum()),
                 'groups_with_slope_ci_excluding_zero': int(((df['slope_ci_low_fs_per_s'] > 0) |
                                                             (df['slope_ci_high_fs_per_s'] < 0)).sum())}
        # Assumption-based correction for every group (references interpolate onto themselves).
        base = interpolate_reference_timing(df['elapsed_mid_s'].to_numpy(), ref['elapsed_mid_s'].to_numpy(),
                                            ref['relative_delay_fs'].to_numpy())
        df['assumed_drift_baseline_fs'] = base
        df['drift_corrected_displacement_um'] = orient * factor * (df['relative_delay_fs'] - base) / 1000
        df['drift_corrected_error_um'] = df['drift_corrected_displacement_um'] - df['expected_um']
        moved = ~df['is_reference'] & (df['expected_um'].abs() > 0)
        df['drift_corrected_scale'] = np.nan
        df.loc[moved, 'drift_corrected_scale'] = df.loc[moved, 'drift_corrected_displacement_um'] / df.loc[moved, 'expected_um']
        nonref = df[moved & df['drift_corrected_displacement_um'].notna()]
        large = nonref[np.abs(nonref['expected_um']) >= float(cfg.get('reference', {}).get('scale_check_min_um', 1000))]
        if len(large):
            drift['drift_corrected_scale_large_moves'] = [float(x) for x in large['drift_corrected_scale']]

    return {'positions': df, 'references': ref, 'summary': summary, 'drift': drift, 'notes': notes}


def precision_assessment(df: pd.DataFrame, summary: Dict, target_um: float, factor: float) -> list:
    """Plain-language warnings when the data cannot support a target_um claim."""
    w = []
    se = df['se_displacement_um'].max()
    if np.isfinite(se) and se > target_um / 3:
        w.append(f'Statistical uncertainty of a group median is up to {se:.2f} µm (target {target_um} µm).')
    if 'trend_over_group_fs' in df:
        tr = (df['trend_over_group_fs'].abs() * factor / 1000).median()
        if np.isfinite(tr) and tr > target_um:
            w.append(f'Median timing trend WITHIN a stationary group is {tr:.1f} µm-equivalent; '
                     f'drift during one acquisition exceeds the {target_um} µm target.')
    if summary.get('return_std_um') is not None and summary['return_std_um'] > target_um:
        w.append(f'Uncorrected reference-return SD = {summary["return_std_um"]:.1f} µm > {target_um} µm: '
                 f'{target_um} µm repeatability is NOT demonstrated by this data.')
    if summary.get('loo_residual_std_um') is not None and summary['loo_residual_std_um'] > target_um:
        w.append(f'Even under a linear-drift assumption, return residuals scatter by '
                 f'{summary["loo_residual_std_um"]:.1f} µm (SD).')
    return w
