---
layout: default
title: Tutorial
nav_order: 2
---

# thz_tds — Tutorial

A practical guide for processing THz-TDS measurements with this package.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Setting up your config file](#2-setting-up-your-config-file)
3. [Loading a single dataset](#3-loading-a-single-dataset)
4. [Alignment methods explained](#4-alignment-methods-explained)
5. [Power spectrum](#5-power-spectrum)
6. [Refractive index and absorption coefficient](#6-refractive-index-and-absorption-coefficient)
7. [Fabry-Perot echo removal](#7-fabry-perot-echo-removal)
8. [Multi-sample batch survey](#8-multi-sample-batch-survey)
9. [Saving figures](#9-saving-figures)
10. [Using in a Jupyter notebook](#10-using-in-a-jupyter-notebook)
11. [Common errors and fixes](#11-common-errors-and-fixes)

---

## 1. Installation

```bash
pip install -e /path/to/thz_tds
```

Verify it works:

```python
import thz_tds
print(thz_tds.__version__)   # → 0.1.0
```

---

## 2. Setting up your config file

Copy the template and edit it for your experiment:

```bash
cp /path/to/thz_tds/config/default_config.yaml  my_experiment.yaml
```

Open `my_experiment.yaml` and fill in the paths:

```yaml
paths:
  reference: "/mnt/Data/experiment_01/reference"   # folder with air scans
  base_dir:  "/mnt/Data/experiment_01"             # root folder — scanned for sample subfolders
  output:    "/mnt/Data/results/experiment_01"     # where figures and CSVs are saved

loading:
  correction_factor: 0.0414   # your instrument's amplitude scaling factor

alignment:
  enabled: true
  method: "correlation_subsample"   # recommended

refractive_index:
  thickness_m: 3.405e-3    # sample thickness in metres
  f_low_THz: 0.4
  f_high_THz: 1.1
```

Your folder structure should look like this:

```
/mnt/Data/experiment_01/
├── reference/          ← air (reference) scan files (.tsv)
├── SampleA/            ← sample A scan files (.tsv)
├── SampleB/            ← sample B scan files (.tsv)
└── SampleC/
```

Each subfolder contains one `.tsv` file per scan.  Each `.tsv` must have two
columns separated by a tab: **time (ps)** and **amplitude**.

---

## 3. Loading a single dataset

```python
import thz_tds

cfg = thz_tds.load_config("my_experiment.yaml")

# Load the reference (air) scan
ref = thz_tds.THZDataset(
    path=cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor,
    align=cfg.alignment.enabled,
    align_method=cfg.alignment.method,
)

print(ref.name)                  # folder name
print(ref.all_y.shape)           # (num_scans, num_time_points)
print(ref.x_ps[:5])              # first 5 time points in ps
print(ref.y_avg[:5])             # first 5 averaged amplitude values
```

Load a specific sample:

```python
sam = thz_tds.THZDataset(
    path="/mnt/Data/experiment_01/SampleA",
    correction_factor=cfg.loading.correction_factor,
    align=True,
    align_method="correlation_subsample",
)
```

Plot it:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
thz_tds.viz.plot_raw_and_average(ref, zoom_xlim=(90, 110), ax=ax)
plt.show()
```

This produces a plot with all raw traces in faint grey and the average in red.

---

## 4. Alignment methods explained

| Method | Best for | Notes |
|--------|----------|-------|
| `"correlation_subsample"` | Most cases | Sub-sample precision; **recommended** |
| `"correlation"` | Fast preview | Integer-sample shift only |
| `"minimum"` | Very fast, approximate | Aligns by position of minimum value |
| `"gaussian"` | Clean, high-SNR pulses | Fits a Gaussian to find the centre |

```python
# Compare the jitter shifts reported by a dataset
ref_aligned = thz_tds.THZDataset(
    path=cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor,
    align=True,
    align_method="correlation_subsample",
)
print("Jitter shifts (ps):", ref_aligned.jitter_shifts_ps)
print("Peak-to-peak jitter:", ref_aligned.jitter_shifts_ps.ptp(), "ps")
```

---

## 5. Power spectrum

```python
# Compute and plot power spectrum of the averaged trace
freq_THz, power_dB = thz_tds.power_spectrum_dB(
    ref.x_ps, ref.y_avg, pad_factor=4
)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 4))
plt.plot(freq_THz, power_dB, linewidth=2)
plt.xlim(0, 3)
plt.xlabel("Frequency [THz]")
plt.ylabel("Power [dB]")
plt.title("Reference power spectrum")
plt.grid(True)
plt.show()
```

Or use the built-in `compute_fft` method to store it on the dataset object:

```python
ref.compute_fft(pad_factor=4)    # stores freq_THz, spectrum_dB, spectrum_complex
sam.compute_fft(pad_factor=4)

fig, ax = plt.subplots()
ax.plot(ref.freq_THz, ref.spectrum_dB, label="Reference")
ax.plot(sam.freq_THz, sam.spectrum_dB, label="Sample A")
ax.set_xlim(0, 2)
ax.set_xlabel("Frequency [THz]")
ax.set_ylabel("Power [dB]")
ax.legend(); ax.grid(True)
plt.show()
```

---

## 6. Refractive index and absorption coefficient

### Single sample

```python
cfg = thz_tds.load_config("my_experiment.yaml")

ref = thz_tds.THZDataset(cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor, align=True)
sam = thz_tds.THZDataset("/mnt/Data/experiment_01/SampleA",
    correction_factor=cfg.loading.correction_factor, align=True)

f_THz, n_f, T, phi_full, alpha_cm = thz_tds.compute_refractive_index(
    ref_ds=ref,
    sam_ds=sam,
    thickness_m=cfg.refractive_index.thickness_m,
    f_low_THz=cfg.refractive_index.f_low_THz,
    f_high_THz=cfg.refractive_index.f_high_THz,
)

print(f"Average n in band: {n_f[(f_THz>0.5)&(f_THz<1.0)].mean():.4f}")
```

### Plot n(f) and α(f)

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

# Refractive index
ax1.plot(f_THz, n_f, linewidth=2)
ax1.set_xlim(0.4, 1.1)
ax1.set_ylim(1.0, 2.5)
ax1.set_xlabel("Frequency [THz]")
ax1.set_ylabel("n(f)")
ax1.set_title(f"Refractive index — {sam.name}")
ax1.grid(True)

# Absorption coefficient
ax2.plot(f_THz, alpha_cm, linewidth=2)
ax2.set_xlim(0.4, 1.1)
ax2.set_ylim(0, 20)
ax2.set_xlabel("Frequency [THz]")
ax2.set_ylabel("α [cm⁻¹]")
ax2.set_title("Absorption coefficient")
ax2.grid(True)

fig.tight_layout()
plt.show()
```

### Using raw arrays (no THZDataset)

```python
# If you already have arrays from another source (e.g. thzpy):
f_THz, n_f, T, phi, alpha_cm = thz_tds.compute_refractive_index_arrays(
    time_ps=my_time_array,
    y_ref=my_ref_signal,
    y_sam=my_sam_signal,
    thickness_m=3.405e-3,
    f_low_THz=0.4,
    f_high_THz=1.1,
)
```

---

## 7. Fabry-Perot echo removal

The Liu model fits the sample signal in the time domain to remove internal
reflections (echoes) before extracting optical constants.

```python
from thz_tds.fabry_perot import THzTrace

ref = thz_tds.THZDataset(cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor, align=True)
sam = thz_tds.THZDataset("/mnt/Data/experiment_01/SampleA",
    correction_factor=cfg.loading.correction_factor, align=True)

# Wrap in THzTrace containers
ref_trace = THzTrace(time_ps=ref.x_ps, y=ref.y_avg)
sam_trace = THzTrace(time_ps=sam.x_ps, y=sam.y_avg)

# Run FP removal (grid search takes a few minutes)
result = thz_tds.remove_fabry_perot(
    ref=ref_trace,
    sam=sam_trace,
    thickness_min_m=2.90e-3,
    thickness_max_m=3.01e-3,
    thickness_step_m=1e-6,       # 1 µm step — reduce to 5e-6 for a quick test
    echo_count=3,
    f_min_THz=0.2,
    f_max_THz=1.8,
)

fit   = result["fit"]            # LiuFitResult dataclass
spec  = result["spectrum"]       # dict: f_THz, n, alpha_1_per_cm, ...

print(f"Best thickness : {fit.thickness_m*1e3:.4f} mm")
print(f"n_av           : {fit.nav:.4f}")
print(f"R²             : {fit.r2:.4f}")
```

### Plot the time-domain fit

```python
fig, ax = plt.subplots(figsize=(11, 5))
thz_tds.viz.plot_fp_fit(ref_trace, sam_trace, fit, ax=ax)
plt.show()
```

### Access the cleaned optical constants

```python
f  = spec["f_THz"]
n  = spec["n"]
a  = spec["alpha_1_per_cm"]

plt.figure(figsize=(8, 4))
plt.plot(f, n, label="n(f) after FP removal")
plt.xlim(0.2, 1.8); plt.grid(True)
plt.xlabel("Frequency [THz]"); plt.ylabel("Refractive index")
plt.legend(); plt.show()
```

---

## 8. Multi-sample batch survey

The workflow functions load all samples automatically and return plain lists
of dicts — no side effects, no plots until you ask for them.

```python
cfg = thz_tds.load_config("my_experiment.yaml")

# Run the spectral survey (fast)
results = thz_tds.run_spectral_survey(cfg)

# Each result is a dict:
for r in results:
    tag = " [ref]" if r["is_reference"] else ""
    print(f"{r['name']}{tag}  — {r['all_y'].shape[0]} traces")
```

Generate the standard 4-panel figure:

```python
fig = thz_tds.viz.plot_survey_figure(results, cfg)
plt.show()
```

Save it:

```python
thz_tds.viz.save_figure(fig, cfg.paths.output, stem="survey",
                         dpi=300, formats=["png", "pdf"])
```

### Refractive index survey (all samples at once)

```python
ri_results = thz_tds.run_refractive_survey(cfg)

# Plot all samples on one figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
thz_tds.viz.plot_refractive_index(ri_results,
    f_min_THz=0.4, f_max_THz=1.1, ax=ax1)
thz_tds.viz.plot_absorption(ri_results,
    f_min_THz=0.4, f_max_THz=1.1, ax=ax2)
plt.tight_layout()
plt.show()
```

### FP-removal survey (all samples)

```python
fp_results = thz_tds.run_fp_survey(cfg)    # takes a while

for r in fp_results:
    fit = r["fit"]
    print(f"{r['name']}:  d={fit.thickness_m*1e3:.3f} mm  n={fit.nav:.4f}  R²={fit.r2:.4f}")
```

---

## 9. Saving figures

```python
# Save a single figure to PNG and PDF
thz_tds.viz.save_figure(fig, "/mnt/Data/results", stem="my_figure",
                         dpi=300, formats=["png", "pdf"])

# Or use the paths from the config
thz_tds.viz.save_figure(fig, cfg.paths.output, stem="survey",
                         dpi=cfg.plot.dpi, formats=cfg.plot.formats)
```

Save data to a TSV file:

```python
import numpy as np
from thz_tds.io import save_tsv

save_tsv("/mnt/Data/results/SampleA_n.tsv", {
    "freq_THz": f_THz,
    "n_f": n_f,
    "alpha_cm": alpha_cm,
})
```

---

## 10. Using in a Jupyter notebook

```python
import thz_tds
import matplotlib.pyplot as plt
%matplotlib inline

cfg = thz_tds.load_config("my_experiment.yaml")

# Quick check: load reference, plot raw + average
ref = thz_tds.THZDataset(
    cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor,
    align=True,
    align_method="correlation_subsample",
)

fig, ax = plt.subplots(figsize=(12, 4))
thz_tds.viz.plot_raw_and_average(ref, zoom_xlim=cfg.plot.zoom_xlim_ps, ax=ax)
plt.tight_layout()
```

Tip — pass `ax=ax` to every viz function so you control the figure layout:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

thz_tds.viz.plot_refractive_index(ri_results,
    f_min_THz=0.4, f_max_THz=1.1, ax=axes[0])

thz_tds.viz.plot_absorption(ri_results,
    f_min_THz=0.4, f_max_THz=1.1, ylim=(0, 15), ax=axes[1])

fig.suptitle("Polymer samples — optical constants", fontsize=13)
plt.tight_layout()
```

---

## 11. Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: Folder does not exist` | Wrong path in config | Check `paths.reference` and `paths.base_dir` in your YAML |
| `RuntimeError: No readable THz files found` | Files not TSV format | Check that files are tab-separated with at least 2 columns |
| `ConfigError: Missing required field 'reference'` | Config missing paths section | Copy `default_config.yaml` and fill in all `paths.*` fields |
| `RuntimeError: No frequencies in chosen good band` | `f_low_THz` too high or `f_high_THz` too low | Widen the band in your config (typical: 0.3–1.2 THz) |
| `No sample folders found` | `base_dir` folder has no subfolders | Check your data directory structure (each sample must be in its own subfolder) |
| FP grid search is very slow | `thickness_step_m` too small | Use `5e-6` (5 µm) for a quick test, `1e-6` (1 µm) for publication quality |
| Noisy n(f) at low / high frequencies | Frequency outside reliable band | Narrow the `f_low_THz` / `f_high_THz` window to where SNR is good |

---

## Quick reference

```python
import thz_tds

# Config
cfg  = thz_tds.load_config("experiment.yaml")

# Load
ref  = thz_tds.THZDataset(cfg.paths.reference, ...)
sam  = thz_tds.THZDataset("/path/to/sample", ...)
sams = thz_tds.scan_samples(cfg.paths.base_dir, ...)

# Spectral
f, P = thz_tds.power_spectrum_dB(ref.x_ps, ref.y_avg)

# Optical constants
f, n, T, phi, a = thz_tds.compute_refractive_index(ref, sam, thickness_m=3.4e-3)

# Fabry-Perot
result = thz_tds.remove_fabry_perot(ref_trace, sam_trace, ...)

# Batch
results = thz_tds.run_spectral_survey(cfg)
results = thz_tds.run_refractive_survey(cfg)
results = thz_tds.run_fp_survey(cfg)

# Viz
thz_tds.viz.plot_raw_and_average(ref, ax=ax)
thz_tds.viz.plot_time_domain([ref, sam], ax=ax)
thz_tds.viz.plot_power_spectra(results, ax=ax)
thz_tds.viz.plot_refractive_index(results, ax=ax)
thz_tds.viz.plot_absorption(results, ax=ax)
thz_tds.viz.plot_fp_fit(ref_trace, sam_trace, fit, ax=ax)
thz_tds.viz.plot_survey_figure(results, cfg)
thz_tds.viz.save_figure(fig, output_dir, stem="name")
```
