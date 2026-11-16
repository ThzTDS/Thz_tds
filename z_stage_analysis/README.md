# THz Z-stage displacement analysis

This package includes **your uploaded `THZDataset` class** (`thz_tds/dataset.py`) unchanged, plus a minimal `spectral.py` compatibility module because your original `.spectral` module was not uploaded. The analysis calls `THZDataset(..., align=False, fit_method=None)` and processes `all_y_raw` so that physical pulse shifts are **not erased** by jitter correction.

## Setup

1. Unzip into a working directory. Install dependencies: `python -m pip install -r requirements.txt`.
2. Open `config.yaml` and set `data_dir` to the **parent** folder that contains `manifest.csv` and the `C00_S00_Z...` folders.
3. Verify `time_unit` from a saved TSV (your dataset class assumes ps), `stage_unit` from the stage software, `incidence_angle_deg`, and `z_reference` (51.9040 in the example). If the acquisition was at the displayed 51.9044 instead, adjust accordingly.
4. Run: `python analyze_z_stage.py --config config.yaml`.

## Output

Results go to `<output_dir>/cross_correlation/` (this script) and `<output_dir>/gaussian_median_integer/` (`analyze_gaussian_jitter.py`, see README_GAUSSIAN.md and REVIEW.md).

- `00_raw_waveforms.png`: unaligned position averages, essential for inspecting ±3 mm traces.
- `01_displacement.png`, `02_identity.png`: commanded vs measured displacement.
- `03_position_error.png`: measured-minus-commanded displacement.
- `04_reference_returns.png`: apparent return error at reference.
- `05_trace_stability.png`: per-trace timing fluctuations for reference returns.
- `position_results.csv`, `trace_results.csv`, `summary.yaml`.

## Important methodological limits

- The dataset loader reads the first two columns of **all regular files** in a position folder as tab-separated time and signal, with a header row. Position folders should contain only waveform TSV files. It can interpolate inconsistent time grids, so check the original TSV time axes if a calibration issue is suspected.
- `correction_factor=0.0414` is copied from your dataset class; amplitude scaling cancels in delay estimation.
- **Do not enable `align=True`**: alignment removes the displacement and timing jitter being investigated.
- The method uses full-record FFT cross-correlation to find large delays, then a local correlation around the main pulse to refine them. The local refinement assumes the pulse shape remains similar. It may fail at ±3 mm if the beam is strongly defocused or the waveform is truncated.
- Correlation peak parabolic interpolation is an estimator; it does **not** guarantee 1 µm resolution. Validate it using synthetic known shifts and independent controls.
- Quality flags check correlation and relative amplitude. Inspect flagged positions before drawing conclusions.
- The delay-to-Z sign is inferred from nonzero stage excursions. Review this sign against the physical orientation of your stage.
- ECOPS timing drift and stage motion are not independently separable using only these traces. The output is *apparent optical displacement*. For calibrated stage positioning accuracy, compare against an external sensor or interferometer. Return scatter is not absolute position error.
- `trace_log.csv` can be used to study long-term drift; the present script plots within-position trace order, but does not claim a validated drift correction.
- The **first recorded baseline trace is not an absolute position standard**. It defines the zero of the optical delay measurement.

## Suggested validation

Run a stationary 1,000-trace measurement with **no stage motion** to quantify ECOPS timing variation. Repeat the same Z excursions in reversed order. Test algorithmic bias by applying artificial known shifts to one waveform and verifying recovery. If stage precision must be reported below 1 µm, use an independent displacement standard.
