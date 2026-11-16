# Technical review: THz-TDS Z-stage repeatability analysis

Data: `/mnt/samples/Z_Repeatability_20261111_095729`. It contains 13 groups × 1000 traces, 25 000
samples per trace at dt = 0.033372 ps, scanLength 834.3, and 10 s buffer clears every 180 traces.
All numbers below come from running the corrected pipeline on all 13 000 traces. Statements are
marked **confirmed** (checked in code or data) or **hypothesis** (consistent with the data but
not tested independently).

---

## 1. Confirmed bugs and their consequences

| # | Where | Problem | Consequence | Fix |
|---|---|---|---|---|
| 1 | `analyze_gaussian_jitter.py` | The time-axis uniformity test compared per-step diffs with `rtol=1e-3`. The TSV export rounds time values, so steps alternate between 0.0333 and 0.0334 ps (0.3 %). | The script stopped on valid data and produced no results. | Test deviation from a linear fit instead (max 5×10⁻⁵ ps) |
| 2 | `analyze_gaussian_jitter.py` | Quality control: only exceptions counted as failures. Fits on a bound, wrong-lobe fits and timing or amplitude outliers went into the median and the average. | When the positive and negative lobes are similar in size, a fit can lock onto the other lobe. That trace then jumps by about 0.9 ps (≈ 140 µm). The synthetic test shows this. **In this dataset nothing was rejected** (lobe ratio 2852 / −1423). | Bound, polarity, amplitude and 6·MAD checks; counts reported per group |
| 3 | `analyze_gaussian_jitter.py` | Integer shifts padded vacated samples with 0. | With baseline correction off, the DC offset produced fake steps at the record edges of the average. This does not affect timing. | NaN padding with per-sample averaging |
| 4 | `analyze_gaussian_jitter.py` | `status` and `trace_count` in `manifest.csv` were ignored, and `trace_log.csv` was neither used nor checked. | Aborted or incomplete groups would have been analysed silently. No timing information over time was available. | Status filter, count and file-name checks, timestamp analysis |
| 5 | `analyze_gaussian_jitter.py` | The return statistics included the initial reference, which is 0 by definition. `return_error_um` was assigned after the CSV had already been written (dead code). | The SD of the returns was diluted by a fixed zero. | Returns reported without the initial reference, plus peak-to-peak and leave-one-out residuals |
| 6 | both scripts | Both wrote `position_results.csv` and `summary.yaml` to the same `output_dir`. | Running one script overwrote the other's results. | Separate subfolders |
| 7 | `thz_tds/dataset.py` | Files were sorted only when `max_traces` was set. | `trace_number` was not guaranteed to follow acquisition order, which matters for trend and drift plots. | Always sort; `loaded_files` attribute added |
| 8 | `thz_tds/dataset.py` `_estimate_sigma` | The FWHM was measured from the **first to the last** sample above half-maximum anywhere in the trace. | Any echo or ringing above half-maximum would inflate σ and switch to the wide fallback window. **Not triggered here**: the adaptive window was used for 100 % of traces, σ = 0.237 ± 0.006 ps. | Use only the contiguous lobe around the peak |
| 9 | `analyze_z_stage.py` | `max_delay_ps` and `local_window_ps` were missing from `config.yaml`. | KeyError at start-up | Keys added (fixed earlier) |

Not changed: `analyze_z_stage.py` (the cross-correlation method) still keeps all 13 000 traces in
memory (about 2.6 GB). It is a separate method and does not use `THZDataset` alignment.

## 2. Gaussian-median integer method: point-by-point verification

1. **Gaussian fit on a bipolar pulse.** The fit models only the strongest lobe: the positive lobe,
   FWHM ≈ 0.56 ps. The fit window (±0.9 ps) reaches into the rising edge of the negative lobe, 0.93 ps
   later. The fitted t0 therefore has a shape-dependent offset from the true pulse position. This
   offset cancels in differences only if the pulse shape does not change.
   - **Confirmed against a model-free estimator** (the zero crossing between the two lobes): the two
     agree within **≤ 0.15 µm** for all reference returns and the ±100/200 µm steps.
   - At ±3 mm, where σ changes from 0.237 to 0.251 / 0.227 ps and the amplitude drops to 89 %, they
     differ by **0.8 and 1.2 µm**.
   - The synthetic test with 20 % pulse broadening gives a discrepancy of about 6 µm.

   Conclusion: the method is adequate for the returns. At large excursions it has about 1 µm of
   shape bias.
2. **Fit window.** Adaptive (±5 σ_ini) for all 13 000 traces; width 1.80 ± 0.07 ps. Correct.
3. **Failures and outliers.** The data contains 0 failures, 0 bound hits and 0 outliers. QC has been
   added (bug #2).
4. **Integer shift.** `z[n] = y[n+lag]`, so lag > 0 (a late trace) moves the waveform earlier. The
   direction is correct and tested explicitly. Padding is now NaN.
5. **Group arrival time.** It is the median of the unshifted t0. The shifts only change the averaged
   waveform. Arrival times are preserved; tested.
6. **Residual-jitter calculation.** The formula (t0 − lag·dt − median) is correct mathematically. Its
   SD is about dt/√12 = 9.6 fs by construction, so it measures **quantisation, not physical jitter**.
   It has been renamed `residual_after_integer_shift_std_fs`.
7. **Corrected average uses valid traces only.** Yes. The "raw average" uses all traces.
8. **Is the timing reliable enough for 1 µm?**
   - The per-trace SD is 63–114 fs, i.e. 10–17 µm per trace.
   - Residuals are autocorrelated (lag-1 ρ ≈ 0.3–0.45, n_eff ≈ 400 of 1000). The statistical standard
     error of a group median is therefore **0.5–0.9 µm** (`se_displacement_um`).
   - So the estimator is statistically close to 1 µm, but systematic drift (section 4) is ten times
     larger.

## 3. Baseline subtraction

- **Is the baseline window pulse-free?** The first 30 samples cover 0–1 ps, about 500 ps before the
  pulse. Their RMS is no larger than that of the pulse-free region, so the check passes for all groups.
- **Is the raw baseline near zero?** No. The DC level is about 1.1 scaled units (median), against a
  peak of about 116, with a trace-to-trace SD of about 0.9.
- **Effect on Gaussian t0:** **2×10⁻⁹ fs** (numerically zero) for every trace. The fit has a free
  offset C, which absorbs any constant baseline. The synthetic DC-offset test confirms this.
- Baseline correction is now optional (`baseline_correction`). It is harmless and does not matter for
  timing, but it does make the waveform plots and the zero-crossing cross-check meaningful.

## 4. The apparent reference-return drift

Measured, uncorrected: 0, 38.0, 71.5, 101.7, 107.8, 123.3, 143.6 µm (returns at 0.7 → 22 min).
These match your numbers within 1–5 µm; the earlier values came from a different trace subset or
version.

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Python implementation error | Synthetic end-to-end tests recover motion without drift (< 1.5 µm), drift without motion (slope 0.75 fs/s exact) and both combined. An independent zero-crossing estimator reproduces the drift. | **Ruled out** |
| Gaussian-fit bias | Gaussian − zero-crossing ≤ 0.15 µm for the returns | **Ruled out** |
| Baseline processing | Effect 0 fs | **Ruled out** |
| Sign or units | +100 µm commanded → +756 fs (later), so the sign is correct. 25 000 × 0.033372 = 834.3 = scanLength, so the time axis is in ps. 1 µm = 6.671 fs at θ = 0. | **Consistent.** The stage and time-axis units are marked UNVERIFIED in `experiment_config.txt`; the incidence angle is NaN there. |
| Waveform distortion | Pulse shape is identical at all returns (σ 0.2373–0.2376 ps, amplitude ±1 %) | **Ruled out for returns** |
| Time-grid instability in the files | The nominal time axis is identical in every file (< 10⁻⁵ ps). This cannot detect a real ECOPS time-base drift, because the axis is computed, not measured. | Not detectable from files |
| **Continuous drift in time** (ECOPS time base and/or optical path: thermal expansion, stage or mount creep) | Fig. 03 shows one continuous, decelerating curve through all groups. Timing also drifts **while the stage is commanded static**: the median within-group slope is +0.81 fs/s (10/13 positive), against +0.67 fs/s between returns (CI 0.48–1.05). The returns rise monotonically regardless of approach direction. The early groups drift fastest (1.6–3.0 fs/s), as in a warm-up. | **Strongly supported** |
| Return (positioning or backlash) error | Under a linear-drift assumption, the leave-one-out return residuals have SD 6.4 µm (max 12.1 µm, return 6). Mean by approach direction: +4.0 µm (decreasing) vs −1.5 µm (increasing), n = 3 each. | **Hypothesis.** A few-µm component is possible, but it cannot be separated from non-linear drift. |
| Acquisition-related timing steps | Fig. 02/04: t0 jumps by tens of fs between the 180-trace blocks separated by the 10 s buffer clears | **Confirmed observation.** Probably instrument-side; see `examples/diagnostics/test_buffer_clear_jumps.py`. |

**A separate finding (hypothesis on its cause):** after removing drift by interpolation, the ±3 mm
moves measure **0.990 × commanded** in both directions: +2970.7 and −2969.6 µm. A symmetric 1 %
deficit indicates a scale error, not drift. Possible causes:
- ECOPS time-axis calibration;
- an incidence angle of about 8° instead of 0° (cos θ = 0.99);
- the stage lead or scale.

Gaussian shape bias (≤ 1.2 µm) cannot explain 30 µm.

The slope confidence intervals from Theil–Sen assume independent residuals. Because ρ ≈ 0.4,
the intervals are too narrow. Read "13/13 groups with CI excluding 0" as indicative only.

## 5. What this analysis can and cannot establish

**It can establish:**
- The single-trace timing noise: 63–114 fs SD (10–17 µm).
- That group medians are statistically precise to about 0.6 µm.
- That the measured optical path drifted by **+144 µm-equivalent over 21 min** (≈ 6 µm/min,
  slowing down over time), and that this happened inside stationary acquisitions as well.
- That after a linear-drift correction the ±100/200 µm steps reproduce to −5.7 … +5.9 µm, the returns
  scatter by 6.4 µm SD, and the 3 mm moves show a 1 % scale deficit.

**It cannot establish:**
- Whether the drift is in the ECOPS time base or in the physical mirror position (thermal or creep).
  The THz pulse measures only their sum.
- Return repeatability at the 1 µm level. With 11 µm of drift *within* each 95 s group, and a
  non-linear drift between groups 3.5 min apart, 1 µm repeatability is **neither demonstrated nor
  excluded**.
- The "drift-corrected" values in fig. 09 and the `drift_corrected_*` columns depend on the
  linear-drift assumption. They are **not** stage accuracy.

## 6. Acquisition script (C#) review

Checked against the files it produced (`manifest.csv`, `trace_log.csv`, `experiment_config.txt`).

| # | Finding | Effect on this dataset |
|---|---|---|
| A1 | `double zReference` is declared twice (`= 51.9040` and `= zPositions[0]`). In a C# method or script that is a compile error (CS0128). The safety block (`zMinSafe = NaN`, `safeLimitsConfirmed = false`) also throws before any motion. **The script as pasted cannot have produced this data**, so the version that ran differed. | Both declarations give 51.9040, so the value is right. Delete the first declaration. |
| A2 | **ECOPS is initialised about 15 s before the first saved trace** (`created_utc` 08:57:29.5 → first trace 08:57:44.3). | **Hypothesis:** the drift is fastest in the first groups (1.6–3.0 fs/s) and then slows, which fits a post-initialisation warm-up. A stationary run after a long warm-up would test this. |
| A3 | Each `discardBufferedData()` is followed by about 5 s before data arrives: the first trace is logged exactly 5.0 s after `acquisition_start` in all 13 groups. Every buffer clear runs discard → warm-up → discard, which costs about 10 s. | **54 % of each group (51 of 95 s) is dead time**, and drift keeps accumulating during it. The median timing jumps by about 51 fs (≈ 8 µm, SD) between 180-trace blocks. |
| A4 | The second discard after the warm-up traces restarts the reader again. The warm-up then no longer protects the next saved trace. | Checked: the first trace after each restart is not an outlier (−7 ± 8 fs vs block median). Redundant, but harmless here. |
| A5 | The timestamp is taken **after** `saveResult` and a file append. | A few ms offset per trace; negligible against 95 s trends. |
| A6 | `previousZ = zPositions[(step-1) % (n-1)]` is correct only because the sequence must end at the reference, which the script enforces. | Correct |
| A7 | `folderName` uses `{2:F4}` without `InvariantCulture`. On a German-locale PC the folder would be `Z51,9040`. | Not triggered (folders use "."). Pass `CultureInfo.InvariantCulture`. |
| A8 | Only `waitMs(1500)` after `moveAbs`. Neither motion completion nor the actual (encoder) position is recorded. | Commanded and actual Z cannot be compared. If the axis API exposes a position read-back, log it in the manifest. |
| A9 | `numPulse` is used both as the block-reader size and as the number of saved traces. | Cannot verify against the Toptica API; check that the two meanings are meant to be equal. |
| A10 | `experiment_config.txt` records the stage unit and THz time unit as UNVERIFIED and the incidence angle as NaN. | The analysis uses mm, ps and 0° from config.yaml. Fill these in after measuring, because the 1 % scale finding depends on the angle. |

## 7. Recommended experiments

1. **Stationary control:** same timing and duration (about 23 min), stage not moved. If about 140 µm
   of apparent drift appears again, the drift is not caused by the motion.
2. **Common-mode reference in the same waveform:** a reflection from a surface that does not move
   with the stage (e.g. a window or beam-splitter face) gives an ECOPS timing reference trace by trace.
   Analyse the difference in arrival times.
3. **Warm-up** of 1–2 h before measuring; log laser, room and stage temperatures.
4. **Shorter groups, denser references:** e.g. 100 traces (≈ 4 s, < 1 µm of drift at the current rate)
   with a reference between every excursion. This is cheaper than 1000-trace groups, which are
   limited by drift, not by noise.
5. **Scale:** measure the incidence angle, and check the 1 % scale against an independent standard
   (interferometer, gauge block).
6. **Acquisition script:** check whether `clearBufferEveryN = 180` is needed at all. If it is, use
   one discard per clear rather than two. Either change removes up to 50 s of dead time per group,
   and the buffer-clear timing steps along with it (see §6, A3).
7. For a calibrated 1 µm claim, use an independent displacement sensor (capacitive or interferometric).
