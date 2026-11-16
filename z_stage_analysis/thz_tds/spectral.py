import numpy as np

def fft_field(t_ps, field, pad_factor=4):
    t = np.asarray(t_ps, dtype=float)
    y = np.asarray(field, dtype=float)
    dt = float(np.median(np.diff(t)))
    n = max(len(y), int(pad_factor) * len(y))
    return np.fft.rfftfreq(n, d=dt), np.fft.rfft(y, n=n)

def power_spectrum_dB(t_ps, field, pad_factor=4):
    f, e = fft_field(t_ps, field, pad_factor)
    return f, 10 * np.log10(np.maximum(np.abs(e)**2, 1e-300))
