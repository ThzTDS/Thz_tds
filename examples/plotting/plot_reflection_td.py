"""
plot_reflection_td.py
=====================
Time-domain plots for every sample listed in all_samples_Reflection.yaml.

Each sample ``dir`` is expected to contain a ``Reflector/`` and a ``Sample/``
folder of TSV traces.  A folder without these subfolders (the overlay
measurements, where the sample lies on the metal reflector) is treated as the
sample itself and has no separate reference.

Figures produced:

  1. reflection_td_<name>_sample.png — averaged sample trace, one figure per
                                       sample (overlays included)
  2. reflection_td_both.png          — reference and sample together, one
                                       panel per sample that has a reference

Usage
-----
    python examples/plotting/plot_reflection_td.py
    python examples/plotting/plot_reflection_td.py config/all_samples_Reflection.yaml
"""

import sys
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

import thz_tds


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "all_samples_Reflection.yaml"
OUT_DIR_DEFAULT = Path(__file__).parent.parent.parent / "output"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_folders(entry: dict) -> tuple[Path | None, Path]:
    """Return (reference folder, sample folder) for one YAML sample entry."""
    sample_dir = Path(entry["dir"])
    ref_dir = sample_dir / "Reflector"
    sam_dir = sample_dir / "Sample"

    if not sam_dir.is_dir():
        sam_dir = sample_dir
    if not ref_dir.is_dir():
        ref_dir = None
    return ref_dir, sam_dir


def dataset_kwargs(cfg: dict) -> dict:
    """Build THZDataset keyword arguments from the shared YAML settings."""
    load_cfg = cfg["loading"]
    align_cfg = cfg["alignment"]
    return dict(
        correction_factor=load_cfg["correction_factor"],
        max_traces=load_cfg.get("max_traces"),
        align=align_cfg["enabled"],
        align_method=align_cfg["method"],
        corr_window_ps=align_cfg["corr_window_ps"],
        adaptive_fit_window=align_cfg.get("adaptive_fit_window", True),
        fit_window_sigma_factor=align_cfg.get("fit_window_sigma_factor", 5.0),
        fallback_fit_window_ps=align_cfg.get("fallback_fit_window_ps", 15.0),
        min_sigma_ps=align_cfg.get("min_sigma_ps", 0.05),
        max_sigma_ps=align_cfg.get("max_sigma_ps", 10.0),
        min_fit_points=align_cfg.get("min_fit_points", 10),
        max_center_shift_ps=align_cfg.get("max_center_shift_ps", 5.0),
    )


def style_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, alpha=0.3)


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)
    ds_kwargs = dataset_kwargs(cfg)

    # Load every reference/sample folder once (shared references are cached).
    cache: dict[Path, thz_tds.THZDataset] = {}

    def load(folder: Path) -> thz_tds.THZDataset:
        if folder not in cache:
            print(f"Loading {folder}")
            cache[folder] = thz_tds.THZDataset(path=folder, **ds_kwargs)
            print(f"  traces: {cache[folder].all_y.shape[0]}")
        return cache[folder]

    data = []
    for entry in cfg["samples"]:
        ref_dir, sam_dir = resolve_folders(entry)
        ref_ds = load(ref_dir) if ref_dir is not None else None
        data.append((entry["name"], ref_ds, load(sam_dir)))

    out_dir = Path(cfg.get("output") or OUT_DIR_DEFAULT) / "reflection_td"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # 1. each sample on its own, overlays included
    for name, _, sam_ds in data:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.plot(sam_ds.x_ps, sam_ds.y_avg, color="tab:orange", lw=1.2)
        style_axes(ax, f"Sample — {name}")
        figures.append((fig, f"{name}_sample"))

    # 2. reference + sample, one panel per sample with a reference
    pairs = [d for d in data if d[1] is not None]
    fig_both, axes = plt.subplots(len(pairs), 1, figsize=(10, 3.5 * len(pairs)),
                                  sharex=True, squeeze=False, constrained_layout=True)
    for ax, (name, ref_ds, sam_ds) in zip(axes[:, 0], pairs):
        ax.plot(ref_ds.x_ps, ref_ds.y_avg, color="tab:blue", lw=1.2, label="Reference")
        ax.plot(sam_ds.x_ps, sam_ds.y_avg, color="tab:orange", lw=1.2, label="Sample")
        style_axes(ax, name)
        ax.legend()
    figures.append((fig_both, "both"))

    for fig, stem in figures:
        out_path = out_dir / f"reflection_td_{stem}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
