from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    waveform = np.asarray(data)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = max(abs(info.min), abs(info.max))
        waveform = waveform.astype(np.float64) / float(scale)
    else:
        waveform = waveform.astype(np.float64)

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return int(sample_rate), waveform
