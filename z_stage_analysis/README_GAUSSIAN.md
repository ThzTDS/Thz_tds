# Gaussian-median integer Z-stage analysis

Evaluates Z-stage return repeatability from THz reflection off a metal mirror (Toptica ECOPS,
1000 single-shot traces per commanded position). Alignment uses only the
`gaussian_median_integer` method: no cross-correlation is used.

The technical review, the list of bugs fixed, and the scientific interpretation of the
`Z_Repeatability_20261111_095729` data are in [REVIEW.md](REVIEW.md).

## Running

```bash
cd /mnt/z_stage_analysis
python3 analyze_gaussian_jitter.py --config config.yaml      # full analysis
python3 -m unittest discover -s tests -v                     # synthetic validation (~30 s)
```

Set `data_dir` in `config.yaml` to the folder containing `manifest.csv`, `trace_log.csv`,
`experiment_config.txt` and the `C??_S??_Z…` trace folders. Results are written to
`<output_dir>/gaussian_median_integer/`. The cross-correlation script `analyze_z_stage.py`
writes to `<output_dir>/cross_correlation/`, so the two never overwrite each other.

For a quick test, set `analysis.max_traces_per_position: 40`. A full run of 13 × 1000 traces takes
about 2 min of CPU time. On a WSL `/mnt` drive it is limited by file reading (about 13 min); copying
the data to the Linux file system speeds this up.
`analysis.n_workers` processes groups in parallel and needs about 0.6 GB of RAM per worker. Only one
group per worker is held in memory.

## Processing steps (per group)

1. Load with `THZDataset(align=False)`. The time axis must be uniform (checked with a linear fit,
   because the exported values are rounded) and identical across groups.
2. Optional baseline subtraction: median of the first `baseline_samples` samples
   (`preprocessing.baseline_correction`). The pipeline checks that this window contains no pulse.
   With `baseline_sensitivity` on, every trace is also fitted with the opposite setting.
3. Gaussian fit of each trace with `THZDataset.fit_gaussian_to_trace` gives the arrival time t0.
4. Quality control. A trace is excluded from the median and the average if:
   - the fit failed;
   - the fit converged onto a bound;
   - the fitted lobe has the wrong polarity (lobe hopping);
   - its amplitude deviates by more than 50 %;
   - its timing deviates by more than 6·MAD.
5. Group arrival time = median t0 of the valid traces. Each valid trace is shifted by
   `round((t0 − median)/dt)` samples toward the median. Vacated samples are NaN, not 0. The shifted
   traces are averaged. This sharpens the averaged waveform but leaves the group arrival time unchanged.
6. Timestamps from `trace_log.csv`: robust Theil–Sen trend of t0 against acquisition time, the
   detrended scatter, lag-1 autocorrelation, effective N, the standard error of the median, and
   acquisition gaps.

Across groups, the first group defines zero delay, and Δz = c·Δt / (2 cos θ) (1 µm = 6.671 fs at
θ = 0). From this the pipeline derives the reference-return statistics, the drift rates, a
cross-check of the Gaussian timing against the zero crossing, and an assumption-based drift
correction.

## Outputs

| File | Content |
|---|---|
| `trace_timing.csv` | One row per trace: t0, σ, amplitude, QC flags, baseline level, timestamp, detrended residual, zero-crossing time, t0 with the other baseline setting, and the timing offset from the nominal commanded delay |
| `position_results.csv` | One row per group: median arrival, jitter SD/MAD, SE, apparent displacement and error, incremental error, zero-crossing cross-check, trend, baseline check, drift-corrected (assumption) columns |
| `reference_returns.csv` | Groups at `z_reference`: elapsed time, approach direction, apparent displacement, leave-one-out residual |
| `timing_drift_analysis.csv` | Within-group trend per group (slope in fs/s and fs/min with 95 % CI, trend in fs and µm, SDs, autocorrelation, n_eff, gaps) |
| `summary.yaml` | Conversion constants, baseline effect, jitter, return statistics, drift diagnostics, precision warnings, data issues |

## Plots

Wherever the x-axis is the measurement sequence, ticks are labelled with the commanded Z of each group in measurement order. Z repeats at the reference returns, so the order is kept and only the labels change.

| File | What it shows | How to read it |
|---|---|---|
| `01_waveform_NNN[_zoom].png` | Raw average vs jitter-corrected average for each group | The corrected average is taller and narrower because jitter smearing is removed. Its timing is unchanged. |
| `02_trace_arrival_times.png` | t0 − group median for every trace, by trace number; rejected traces marked ✕ | Shows outliers and the block structure from buffer clears every 180 traces |
| `03_timing_vs_time.png` | Every trace's arrival minus its **nominal** commanded delay, against elapsed time | **Key plot.** With a perfect stage and a stable clock this would be flat at 0. A continuous curve through all groups means drift that continues while the stage moves. Isolated offsets at single groups would point to positioning errors. The ±3 mm groups also show any scale error. |
| `04_within_group_trends.png` | t0 against time inside each group, with Theil–Sen line; title gives fs/s and µm per group | The stage is commanded static here, so any slope comes from the instrument or the optical path |
| `04b_trend_slopes.png` | Within-group slopes (95 % CI) compared with the slope between reference returns | Similar values mean one continuous process explains both |
| `05_reference_returns.png` | Apparent displacement of each return, labelled by the Z it returned from, and against elapsed time; marker = approach direction | Uncorrected. A monotonic trend in time points to drift; alternating ± by direction would point to backlash. |
| `06_displacement_vs_commanded.png` | Identity plot and apparent error per group | Uncorrected; stage and timing contributions are not separated |
| `07_jitter_std.png` | Per-trace timing SD (raw, detrended, MAD) per group, with a µm-equivalent axis | Single-trace timing noise; the median of 1000 traces is about 20× more precise |
| `08_baseline_sensitivity.png` | Group-median difference (baseline on − off) and per-trace baseline level against time | A difference of about 0 fs means baseline handling does not affect the timing |
| `09_drift_corrected_ASSUMPTION.png` | Uncorrected error compared with the error after subtracting a timing baseline linearly interpolated between neighbouring references | **This is an assumption, not a measurement of stage accuracy.** It is valid only if the drift is linear over about 3.5 min. |

## Configuration keys added

`preprocessing.baseline_correction / baseline_sensitivity / baseline_warn_ratio`,
`gaussian_fit.min_fit_points / reject_at_bounds / outlier_mad_threshold / max_amplitude_deviation`,
`timing.use_trace_log / interval_outlier_factor`, `reference.z_tolerance / scale_check_min_um`,
`analysis.require_status / precision_target_um / n_workers`, `plot.zoom_window_ps`.
