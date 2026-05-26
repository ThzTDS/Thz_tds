from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = Path("/mnt/samples/PVC_B_6/air10")
correction_factor = 0.0414

files = sorted([p for p in folder.iterdir() if p.is_file()])

rows = []
x_ref = None

for p in files:
    df = pd.read_csv(p, sep="\t", header=0, comment="#")

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float) * correction_factor

    if x_ref is None:
        x_ref = x
    elif len(x) != len(x_ref) or not np.allclose(x, x_ref, atol=1e-9):
        y = np.interp(x_ref, x, y, left=0.0, right=0.0)
        x = x_ref

    # Use minimum if your main THz pulse is negative
    idx_min = int(np.argmin(y))
    t_min = float(x[idx_min])

    # Also check strongest absolute peak
    baseline = np.median(y)
    idx_abs = int(np.argmax(np.abs(y - baseline)))
    t_abs = float(x[idx_abs])

    rows.append({
        "file": p.name,
        "t_min_ps": t_min,
        "t_abs_peak_ps": t_abs,
    })

df_peaks = pd.DataFrame(rows)

print(df_peaks.describe())
print("\nEarliest traces:")
print(df_peaks.sort_values("t_min_ps").head(20))

print("\nLatest traces:")
print(df_peaks.sort_values("t_min_ps").tail(20))

plt.figure(figsize=(8, 4))
plt.hist(df_peaks["t_min_ps"], bins=80)
plt.xlabel("Time of negative minimum / ps")
plt.ylabel("Number of traces")
plt.title(f"Peak-time histogram: {folder.name}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
