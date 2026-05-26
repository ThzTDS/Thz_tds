"""
plot_single_pair_td.py
======================
Plots a single reference/sample pair in the time domain.

Reference : /mnt/samples/PA6_B_3/air4
Sample    : /mnt/samples/PA6_B_3/point4

Usage
-----
    python examples/plot_single_pair_td.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import thz_tds

# ── Paths ────────────────────────────────────────────────────────────────────
REF_PATH = Path("/mnt/samples/PA6_B_3/air4")
SAM_PATH = Path("/mnt/samples/PA6_B_3/point4")

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading reference : {REF_PATH}")
ref_ds = thz_tds.THZDataset(path=REF_PATH, correction_factor=0.0414, align=True)

print(f"Loading sample    : {SAM_PATH}")
sam_ds = thz_tds.THZDataset(path=SAM_PATH, correction_factor=0.0414, align=True)

t_ref = ref_ds.x_ps
y_ref = ref_ds.y_avg
t_sam = sam_ds.x_ps
y_sam = sam_ds.y_avg

print(f"Reference traces  : {ref_ds.all_y.shape[0]}")
print(f"Sample traces     : {sam_ds.all_y.shape[0]}")

# ── Poster-scale settings (A1: ~594 × 841 mm) ───────────────────────────────
plt.rcParams.update({
    "font.size":         36,
    "axes.titlesize":    40,
    "axes.labelsize":    38,
    "xtick.labelsize":   34,
    "ytick.labelsize":   34,
    "axes.linewidth":    2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size":  12,
    "ytick.major.size":  12,
})

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(26, 12))

# averages
ax.plot(t_ref, y_ref, color="tab:blue",   lw=5.0)
ax.plot(t_sam, y_sam, color="tab:orange", lw=5.0)

# inline labels at the end of each line
ax.text(t_ref[-1], y_ref[-1], "  Reference (PA6_B)",
        color="tab:blue",   fontsize=36, va="center", ha="left", clip_on=False)
ax.text(t_sam[-1], y_sam[-1], "  Sample (PA6_B)",
        color="tab:orange", fontsize=36, va="center", ha="left", clip_on=False)

ax.set_title(f"Time domain — PA6_B  |  {REF_PATH.name} vs {SAM_PATH.name}", pad=22)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Amplitude (a.u.)")
ax.grid(True, alpha=0.3, linewidth=1.5)
fig.tight_layout()

# save poster-ready PNG (300 dpi)
out_path = Path(__file__).parent.parent.parent / "output" / "PA6_B_td_poster.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()
