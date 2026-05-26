"""
Alignment method comparison
Loads air4 (raw), runs all 4 alignment methods, plots results.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import thz_tds
from thz_tds.dataset import THZDataset

# ── Config ───────────────────────────────────────────────────────────────────
default_config = "/mnt/Code/thz_tds/config/default_config.yaml"
config_path = sys.argv[1] if len(sys.argv) >= 2 else default_config
cfg = thz_tds.load_config(config_path)

# ── Load raw (no alignment) ───────────────────────────────────────────────────
print("Loading reference (no alignment)…")
ds_raw = THZDataset(
    path=cfg.paths.reference,
    correction_factor=cfg.loading.correction_factor,
    align=False,
)
t        = ds_raw.x_ps
all_raw  = ds_raw.all_y        # shape (M, N)
avg_raw  = ds_raw.y_avg
n_traces = all_raw.shape[0]
print(f"  {ds_raw.name}: {n_traces} traces loaded")

# ── Run all alignment methods ─────────────────────────────────────────────────
METHODS = {
    "No alignment":            (all_raw, np.zeros(n_traces), avg_raw),
    "Minimum":                 ds_raw.align_by_minimum_to_first(t, all_raw),
    "Correlation":             ds_raw.align_by_correlation_to_first(t, all_raw, subsample=False),
    "Correlation (subsample)": ds_raw.align_by_correlation_to_first(t, all_raw, subsample=True),
    "Gaussian":                ds_raw.align_by_gaussian_to_first(t, all_raw),
}
# Normalise to same dict shape
results = {}
for name, val in METHODS.items():
    aligned, shifts, _ = val if isinstance(val, tuple) and len(val) == 3 else val
    results[name] = {"aligned": aligned, "shifts": shifts, "avg": np.mean(aligned, axis=0)}

# Fix "No alignment" entry (already a tuple of 3)
results["No alignment"] = {"aligned": all_raw, "shifts": np.zeros(n_traces), "avg": avg_raw}

# ── Zoom ──────────────────────────────────────────────────────────────────────
xlim = cfg.plot.zoom_xlim_ps
ALPHA = max(0.04, min(0.3, 5.0 / n_traces))

os.makedirs(cfg.paths.output, exist_ok=True)

# ── One figure per method ─────────────────────────────────────────────────────
colors = ["gray", "steelblue", "darkorange", "green", "purple"]
for (name, res), col in zip(results.items(), colors):
    fig, ax = plt.subplots(figsize=(12, 5))
    for y in res["aligned"]:
        ax.plot(t, y, color=col, alpha=ALPHA, linewidth=0.5)
    ax.plot(t, res["avg"], color=col, linewidth=2, label=f"{name} average")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (nA)")
    ax.set_title(f"{name}  —  {ds_raw.name}  ({n_traces} traces)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = f"align_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
    #fig.savefig(os.path.join(cfg.paths.output, f"{stem}.png"), dpi=cfg.plot.dpi)
print(f"Figures saved to {cfg.paths.output}")

plt.show()
