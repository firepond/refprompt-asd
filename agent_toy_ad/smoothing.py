from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_waveform_savgol(samples: np.ndarray) -> np.ndarray:
    length = int(samples.size)
    if length < 9:
        return samples.astype(np.float64)

    target = max(_nine_or_more_odd(min(length - 1 if length % 2 == 0 else length, max(101, length // 200))), 9)
    window_length = min(target, length - 1 if length % 2 == 0 else length)
    if window_length < 9:
        window_length = 9 if length >= 9 else length | 1
    if window_length % 2 == 0:
        window_length -= 1
    polyorder = min(3, window_length - 2)
    return savgol_filter(samples.astype(np.float64), window_length=window_length, polyorder=polyorder, mode="interp")


def _nine_or_more_odd(value: int) -> int:
    candidate = max(value, 9)
    if candidate % 2 == 0:
        candidate += 1
    return candidate
