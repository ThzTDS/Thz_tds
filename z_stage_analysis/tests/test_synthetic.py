"""Synthetic validation of the Gaussian-median-integer Z-stage pipeline.

Run from z_stage_analysis/:
    python3 -m unittest discover -s tests -v       (or: python3 -m pytest tests)

Synthetic pulse: positive lobe followed by a half-amplitude negative lobe, like
the measured ECOPS waveform. Both the zero crossing and the Gaussian fit of the positive lobe have constant offsets
from the true tc that cancel in differences as long as the pulse shape is unchanged.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import zstage_core as core  # noqa: E402
import analyze_gaussian_jitter as agj  # noqa: E402

DT = 0.033372          # ps, as measured
N = 3000               # 100 ps record
T = np.arange(N) * DT
TC = 50.0              # nominal pulse position (ps)
SIG = 0.24             # lobe width (measured fitted sigma 0.237 ps)
FS_PER_UM = 1000 / core.um_per_ps(0.0)   # 6.671 fs per µm (normal incidence)
FIT = {'adaptive_fit_window': True, 'fit_window_sigma_factor': 5.0, 'fit_window_ps': 8.0,
       'min_sigma_ps': 0.05, 'max_sigma_ps': 10.0, 'max_center_shift_ps': 5.0,
       'outlier_mad_threshold': 6.0, 'max_amplitude_deviation': 0.5, 'reject_at_bounds': True}


def pulse(tc, amp=100.0, sig=SIG, ratio=0.5, half_sep=0.46):
    """Positive lobe at tc-0.46 ps, negative lobe (ratio × amplitude) at tc+0.46 ps,
    matching the measured 2852 / -1423 lobes 0.93 ps apart."""
    return amp * (np.exp(-0.5 * ((T - tc + half_sep) / sig) ** 2)
                  - ratio * np.exp(-0.5 * ((T - tc - half_sep) / sig) ** 2))


def make_traces(centers_ps, rng, noise=0.5, dc=0.0, sig=SIG, ratio=0.5):
    return np.array([pulse(c, sig=sig, ratio=ratio) + dc + rng.normal(0, noise, N) for c in centers_ps])


def base_cfg(**over):
    cfg = {'preprocessing': {'baseline_correction': True, 'baseline_samples': 30, 'baseline_sensitivity': True,
                             'baseline_warn_ratio': 3.0},
           'gaussian_fit': dict(FIT), 'timing': {'interval_outlier_factor': 3.0}}
    for k, v in over.items():
        cfg[k].update(v)
    return cfg


class TestUnitsAndSign(unittest.TestCase):
    def test_conversion(self):
        self.assertAlmostEqual(FS_PER_UM, 6.6713, places=3)
        self.assertAlmostEqual(core.um_per_ps(60.0), 2 * core.um_per_ps(0.0), places=9)
        self.assertEqual(core.stage_um(0.1, {'stage_unit': 'mm'}), 100.0)
        self.assertEqual(core.stage_um(5, {'stage_unit': 'um'}), 5.0)
        self.assertAlmostEqual(core.stage_um(2, {'stage_unit': 'native', 'stage_units_per_mm': 4}), 500.0)
        np.testing.assert_allclose(core.time_ps([1000.0], {'time_unit': 'fs'}), [1.0])
        with self.assertRaises(ValueError):
            core.um_per_ps(90.0)

    def test_integer_shift_direction(self):
        y = np.zeros(20); y[10] = 1
        self.assertEqual(int(np.nanargmax(core.shift_integer(y, 3))), 7)    # lag>0 → earlier
        self.assertEqual(int(np.nanargmax(core.shift_integer(y, -3))), 13)  # lag<0 → later
        z = core.shift_integer(y, 3)
        self.assertTrue(np.all(np.isnan(z[-3:])))                          # padding is NaN, not 0
        self.assertTrue(np.all(np.isnan(core.shift_integer(y, -2)[:2])))

    def test_align_moves_late_trace_to_reference(self):
        rng = np.random.default_rng(0)
        traces = make_traces([TC, TC + 10 * DT], rng, noise=0)
        avg, lags, shift_ps, resid = core.integer_align_average(traces, np.array([TC, TC + 10 * DT]),
                                                                np.array([True, True]), DT, TC)
        self.assertEqual(list(lags), [0, 10])
        self.assertAlmostEqual(shift_ps[1], -10 * DT)
        np.testing.assert_allclose(resid, 0, atol=1e-12)
        np.testing.assert_allclose(avg[100:-100], traces[0][100:-100], atol=1e-9)


class TestGroupProcessing(unittest.TestCase):
    def setUp(self):
        self.fitter = core.gaussian_fitter(FIT)
        self.rng = np.random.default_rng(1)

    def test_median_preserved_and_average_sharpened(self):
        true = TC + 0.2
        c = true + self.rng.normal(0, 0.07, 200)                   # 70 fs jitter
        res = core.process_group(T, make_traces(c, self.rng), self.fitter, base_cfg())
        st = res['stats']
        # Gaussian offset for this pulse shape, measured on a noise-free pulse:
        off = self.fitter.fit_gaussian_to_trace(T, pulse(true))['t0_ps'] - true
        self.assertLess(abs(st['median_arrival_ps'] - off - np.median(c)) * 1000, 3.0)   # fs
        zoff = core.zero_crossing_time(T, pulse(true)) - true
        self.assertLess(abs(st['zero_crossing_median_ps'] - zoff - np.median(c)) * 1000, 5.0)
        self.assertAlmostEqual(st['jitter_std_fs'], 70, delta=10)
        # Residual after integer shifts ≈ uniform quantisation, dt/sqrt(12) ≈ 9.6 fs
        self.assertAlmostEqual(st['residual_after_integer_shift_std_fs'], DT * 1000 / np.sqrt(12), delta=2)
        self.assertGreater(np.nanmax(res['corrected_avg']), np.max(res['raw_avg']))
        self.assertEqual(st['n_valid'], 200)

    def test_dc_offset_does_not_move_gaussian_center(self):
        c = TC + self.rng.normal(0, 0.05, 50)
        raw = make_traces(c, self.rng, dc=5.0)                      # 5 % DC offset
        on = core.process_group(T, raw, self.fitter, base_cfg())['stats']
        off = core.process_group(T, raw, self.fitter, base_cfg(preprocessing={'baseline_correction': False}))['stats']
        self.assertLess(abs(on['median_arrival_ps'] - off['median_arrival_ps']) * 1000, 0.01)
        self.assertLess(abs(on['baseline_effect_on_median_fs']), 0.01)
        self.assertAlmostEqual(on['baseline_level_median'], 5.0, delta=0.3)
        self.assertTrue(on['baseline_window_ok'])

    def test_baseline_window_containing_pulse_is_flagged(self):
        raw = make_traces(TC + self.rng.normal(0, 0.05, 20), self.rng)
        cfg = base_cfg(preprocessing={'baseline_samples': int(49 / DT), 'baseline_sensitivity': False})
        self.assertFalse(core.process_group(T, raw, self.fitter, cfg)['stats']['baseline_window_ok'])

    def test_outliers_and_failures_excluded(self):
        c = TC + self.rng.normal(0, 0.05, 100)
        raw = make_traces(c, self.rng)
        raw[:4] = make_traces([TC + 3.0] * 4, self.rng)             # timing outliers (+3 ps)
        raw[4:7] = self.rng.normal(0, 0.5, (3, N))                    # pulse missing
        res = core.process_group(T, raw, self.fitter, base_cfg())
        tr, st = res['traces'], res['stats']
        self.assertTrue(np.all(~tr['valid'].to_numpy()[:7]))
        self.assertEqual(st['n_valid'], 93)
        off = self.fitter.fit_gaussian_to_trace(T, pulse(TC))['t0_ps'] - TC
        self.assertLess(abs(st['median_arrival_ps'] - off - np.median(c[7:])) * 1000, 3.0)

    def test_lobe_hopping_rejected(self):
        # Equal lobes: the strongest |lobe| flips with noise; flipped traces must be rejected.
        c = TC + self.rng.normal(0, 0.03, 100)
        raw = make_traces(c, self.rng, ratio=0.5)
        raw[:5] = -make_traces(c[:5], self.rng, ratio=0.5)           # inverted polarity
        st = core.process_group(T, raw, self.fitter, base_cfg())['stats']
        self.assertEqual(st['n_polarity_outliers'], 5)
        self.assertEqual(st['n_valid'], 95)

    def test_within_group_trend(self):
        e = np.arange(300) * 0.32                                     # 96 s
        c = TC + 1e-3 * (1.0 * e + self.rng.normal(0, 50, len(e)))   # 1 fs/s + 50 fs jitter
        st = core.process_group(T, make_traces(c, self.rng), self.fitter, base_cfg(), elapsed_s=e)['stats']
        self.assertAlmostEqual(st['slope_fs_per_s'], 1.0, delta=0.3)
        self.assertLess(st['slope_ci_low_fs_per_s'], 1.0); self.assertGreater(st['slope_ci_high_fs_per_s'], 1.0)
        self.assertAlmostEqual(st['sd_detrended_fs'], 50, delta=8)
        self.assertGreater(st['sd_raw_fs'], st['sd_detrended_fs'])

    def test_interval_check(self):
        e = np.arange(400) * 0.032
        e[180:] += 10; e[360:] += 10; e[5] = np.nan
        r = core.interval_check(e)
        self.assertEqual(r['irregular_after_traces'], [180, 360])
        self.assertEqual(r['n_missing_timestamps'], 1)


# ---------------------------------------------------------------------------
# End-to-end: write a synthetic acquisition folder and run main()
# ---------------------------------------------------------------------------

SEQ = [0, 100, 0, -100, 0, 200, 0, -200, 0, 3000, 0, -3000, 0]   # µm, as in the experiment
ZREF = 51.904


def write_dataset(root, z_um, drift_fs_per_s=0.0, jitter_fs=20.0, n=30, interval_s=3.2,
                  gap_s=10.0, sig_fn=None, seed=2):
    """True arrival = TC + z/ (c/2) + drift·t + jitter. Stage error = 0 by construction."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    man, log = [], []
    t_now = 0.0
    t0 = pd.Timestamp('2026-11-11T08:57:00Z')
    prev = ZREF
    for step, z in enumerate(z_um):
        zc = ZREF + z / 1000
        folder = f'C{min(step, 1):02d}_S{step:02d}_Z{zc:.4f}'
        (root / folder).mkdir()
        sig = sig_fn(z) if sig_fn else SIG
        for k in range(1, n + 1):
            el = t_now + (k - 1) * interval_s
            arr = TC + z * FS_PER_UM / 1000 + drift_fs_per_s * el / 1000 + rng.normal(0, jitter_fs) / 1000
            y = (pulse(arr, sig=sig) + 1.5 + rng.normal(0, 0.5, N)) / 0.0414
            fname = f'Trace_{k:04d}.tsv'
            with open(root / folder / fname, 'w') as f:
                f.write('Position (a.u.)\tSignal 0 (a.u.)\t\n')
                f.write(''.join(f'{a:.6e}\t{b:.6e}\t\n' for a, b in zip(T, y)))
            log.append(dict(cycle=min(step, 1), step=step, z_command=f'{zc:.4f}', trace_number=k,
                            acquired_utc=(t0 + pd.Timedelta(seconds=el)).isoformat(),
                            relative_file=f'{folder}/{fname}'))
        man.append(dict(cycle=min(step, 1), step=step, z_command=f'{zc:.4f}', previous_z_command=f'{prev:.4f}',
                        approach_direction='initial' if step == 0 else ('increasing_Z' if zc > prev else 'decreasing_Z'),
                        trace_count=n, folder=folder, status='completed'))
        prev = zc
        t_now += (n - 1) * interval_s + gap_s
    pd.DataFrame(man).to_csv(root / 'manifest.csv', index=False)
    pd.DataFrame(log).to_csv(root / 'trace_log.csv', index=False)
    (root / 'experiment_config.txt').write_text(f'z_reference={ZREF:.4f}\nincidence_angle_deg=0\n')


def run_pipeline(data_dir, out_dir, **analysis):
    cfg = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf8'))
    cfg.update(data_dir=str(data_dir), output_dir=str(out_dir), z_reference=ZREF, incidence_angle_deg=0.0)
    cfg['analysis'].update(max_traces_per_position=None, n_workers=1, **analysis)
    cfg['plot']['zoom_window_ps'] = None
    p = Path(out_dir) / 'cfg.yaml'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    out = agj.main(p)
    pos = pd.read_csv(out / 'position_results.csv')
    summ = yaml.safe_load((out / 'summary.yaml').read_text(encoding='utf8'))
    return pos, summ, out


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='zstage_test_'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_motion_without_drift(self):
        write_dataset(self.tmp / 'motion', SEQ)
        pos, summ, out = run_pipeline(self.tmp / 'motion', self.tmp / 'out_motion')
        err = pos['apparent_error_um'].to_numpy()
        self.assertLess(np.max(np.abs(err)), 1.5, err)               # µm, incl. ±3 mm
        self.assertLess(summ['reference_returns']['return_std_um'], 1.0)
        # sign: +Z → later arrival → positive displacement
        self.assertGreater(pos.loc[pos.z_command > ZREF + 1, 'relative_delay_fs'].iloc[0], 0)
        self.assertLess(abs(summ['drift_diagnostics']['between_reference_slope_fs_per_s']), 0.1)
        for f in ('trace_timing.csv', 'reference_returns.csv', 'timing_drift_analysis.csv', 'summary.yaml',
                  '03_timing_vs_time.png', '09_drift_corrected_ASSUMPTION.png'):
            self.assertTrue((out / f).is_file(), f)

    def test_drift_without_motion(self):
        write_dataset(self.tmp / 'drift', [0] * 7, drift_fs_per_s=0.75)
        pos, summ, _ = run_pipeline(self.tmp / 'drift', self.tmp / 'out_drift')
        d = summ['drift_diagnostics']
        self.assertAlmostEqual(d['between_reference_slope_fs_per_s'], 0.75, delta=0.05)
        self.assertAlmostEqual(d['median_within_group_slope_fs_per_s'], 0.75, delta=0.3)
        # Apparent "return error" equals drift: ~0.75 fs/s · elapsed / 6.67 fs/µm
        last = pos.iloc[-1]
        self.assertAlmostEqual(last['apparent_displacement_um'],
                               0.75 * (last['elapsed_mid_s'] - pos.iloc[0]['elapsed_mid_s']) / FS_PER_UM, delta=1.0)
        self.assertGreater(summ['reference_returns']['return_max_abs_um'], 50)
        self.assertTrue(any('NOT demonstrated' in w for w in summ['precision_warnings']))

    def test_motion_with_drift_assumption_correction(self):
        write_dataset(self.tmp / 'both', SEQ, drift_fs_per_s=0.75)
        pos, summ, _ = run_pipeline(self.tmp / 'both', self.tmp / 'out_both')
        moved = pos[~pos.is_reference]
        self.assertGreater(moved['apparent_error_um'].abs().max(), 50)           # raw contains drift
        self.assertLess(moved['drift_corrected_error_um'].abs().max(), 1.5)       # linear drift removed
        np.testing.assert_allclose(summ['drift_diagnostics']['drift_corrected_scale_large_moves'], 1.0, atol=1e-3)

    def test_wrong_sign_is_reported(self):
        write_dataset(self.tmp / 'sign', SEQ[:5])
        _, summ, _ = run_pipeline(self.tmp / 'sign', self.tmp / 'out_sign', delay_sign_for_positive_z=-1)
        self.assertTrue(any('sign disagrees' in w for w in summ['data_issues']))

    def test_shape_change_bias_detected(self):
        # Lobes broaden 20 % at ±3 mm (defocus). Gaussian t0 and zero crossing shift in
        # opposite directions (≈ -20 / +19 fs): timing becomes estimator-dependent and the
        # pipeline must expose it through gauss_minus_zero_crossing_um.
        write_dataset(self.tmp / 'shape', SEQ, sig_fn=lambda z: SIG * (1.2 if abs(z) >= 1000 else 1.0))
        pos, _, _ = run_pipeline(self.tmp / 'shape', self.tmp / 'out_shape')
        far = pos[pos.z_command.sub(ZREF).abs() > 1]
        near = pos[pos.z_command.sub(ZREF).abs() < 1]
        self.assertTrue((far['gauss_minus_zero_crossing_um'].abs() > 3).all())
        self.assertTrue((far['apparent_error_um'].abs() > 1.5).all())      # Gaussian estimate biased
        self.assertLess(near['gauss_minus_zero_crossing_um'].abs().max(), 1.5)  # same shape: agree

    def test_incomplete_group_skipped(self):
        write_dataset(self.tmp / 'inc', SEQ[:5])
        m = pd.read_csv(self.tmp / 'inc' / 'manifest.csv')
        m.loc[3, 'status'] = 'aborted'; m.to_csv(self.tmp / 'inc' / 'manifest.csv', index=False)
        shutil.rmtree(self.tmp / 'inc' / m.loc[4, 'folder'])
        pos, summ, _ = run_pipeline(self.tmp / 'inc', self.tmp / 'out_inc')
        self.assertEqual(len(pos), 3)
        self.assertTrue(any('aborted' in w for w in summ['data_issues']))
        self.assertTrue(any('folder missing' in w for w in summ['data_issues']))


if __name__ == '__main__':
    unittest.main()
