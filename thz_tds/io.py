"""
thz_tds.io
==========
Low-level file I/O for THz-TDS data stored as tab-separated value (TSV) files.

All filesystem operations are isolated here so no other module needs to call
``pd.read_csv`` or ``os.listdir`` directly.  Each function raises a clear,
specific exception on failure rather than silently swallowing errors.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def list_tsv_files(folder: str | Path) -> List[Path]:
    """
    Return a sorted list of all regular files inside *folder*.

    Parameters
    ----------
    folder : str or Path
        Directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of :class:`pathlib.Path` objects for every file found.
        Subdirectories are not included.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist or is not a directory.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {folder}")

    return sorted(p for p in folder.iterdir() if p.is_file())


def load_trace(
    path: str | Path,
    correction_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one THz time-domain file and return the time axis and amplitude.

    The file is expected to be a tab-separated file with at least two columns.
    The first column is the time axis (in picoseconds) and the second is the
    raw electric-field amplitude.

    Parameters
    ----------
    path : str or Path
        Path to the TSV file.
    correction_factor : float
        Multiplicative scaling applied to the raw amplitude column.

    Returns
    -------
    time_ps : np.ndarray
        Time axis in picoseconds.
    amplitude : np.ndarray
        Scaled electric-field amplitude.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file has fewer than two columns or cannot be parsed as floats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    try:
        df = pd.read_csv(path, sep="\t", header=0, comment="#")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc

    if df.shape[1] < 2:
        raise ValueError(
            f"{path} has only {df.shape[1]} column(s); at least 2 required."
        )

    try:
        time_ps = df.iloc[:, 0].to_numpy(dtype=float)
        amplitude = df.iloc[:, 1].to_numpy(dtype=float) * correction_factor
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric data in {path}: {exc}") from exc

    return time_ps, amplitude


def load_folder(
    folder: str | Path,
    correction_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all THz traces in *folder* and return a common time axis and a
    matrix of amplitudes (one row per file).

    Traces that cannot be parsed are skipped with a warning; the common time
    axis is taken from the first successfully loaded file.  Any trace with a
    different time axis is re-interpolated onto the common axis.

    Parameters
    ----------
    folder : str or Path
        Directory containing the TSV trace files.
    correction_factor : float
        Multiplicative scaling applied to every amplitude column.

    Returns
    -------
    time_ps : np.ndarray, shape (N,)
        Common time axis in picoseconds.
    y_matrix : np.ndarray, shape (M, N)
        Matrix of scaled amplitudes; M is the number of successfully loaded files.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    RuntimeError
        If no file in *folder* can be loaded successfully.
    """
    files = list_tsv_files(folder)
    if not files:
        raise RuntimeError(f"No files found in {folder}")

    rows: List[np.ndarray] = []
    time_ref: np.ndarray | None = None

    for fpath in files:
        try:
            t, y = load_trace(fpath, correction_factor=correction_factor)
        except (ValueError, OSError) as exc:
            log.warning("Skipping %s: %s", fpath.name, exc)
            continue

        if time_ref is None:
            time_ref = t
        elif len(t) != len(time_ref) or not np.allclose(t, time_ref, atol=1e-9):
            y = np.interp(time_ref, t, y, left=0.0, right=0.0)

        rows.append(y)

    if not rows:
        raise RuntimeError(f"No readable THz files found in {folder}")

    return time_ref, np.vstack(rows)


def save_tsv(path: str | Path, data: Dict[str, np.ndarray]) -> None:
    """
    Save a dictionary of named columns to a tab-separated file.

    Parameters
    ----------
    path : str or Path
        Output file path.  Parent directories are created if needed.
    data : dict[str, np.ndarray]
        Mapping from column name to 1-D array.  All arrays must have the
        same length.

    Raises
    ------
    ValueError
        If *data* is empty or arrays have inconsistent lengths.
    """
    if not data:
        raise ValueError("data dict is empty; nothing to save.")

    lengths = {k: len(v) for k, v in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All arrays must have the same length. Got: {lengths}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(path, sep="\t", index=False)
    log.info("Saved %d rows to %s", len(df), path)
