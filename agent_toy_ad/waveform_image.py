from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from .smoothing import smooth_waveform_savgol


def render_raw_waveform_png(
    samples: np.ndarray,
    *,
    width: int = 512,
    height: int = 256,
) -> bytes:
    image = build_raw_waveform_image(samples, width=width, height=height)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_raw_waveform_image(
    samples: np.ndarray,
    *,
    width: int = 512,
    height: int = 256,
) -> Image.Image:
    return _build_waveform_image(samples, width=width, height=height)


def render_smoothed_waveform_png(
    samples: np.ndarray,
    *,
    width: int = 512,
    height: int = 256,
) -> bytes:
    image = build_smoothed_waveform_image(samples, width=width, height=height)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_smoothed_waveform_image(
    samples: np.ndarray,
    *,
    width: int = 512,
    height: int = 256,
) -> Image.Image:
    smoothed = smooth_waveform_savgol(samples)
    return _build_waveform_image(smoothed, width=width, height=height)


def _build_waveform_image(
    samples: np.ndarray,
    *,
    width: int,
    height: int,
) -> Image.Image:
    if samples.size == 0:
        raise ValueError("Audio clip is empty.")

    reduced = _resample_1d(samples, width)

    scale = float(np.percentile(np.abs(reduced), 99.0))
    if scale <= 1e-8:
        scale = 1.0
    normalized = np.clip(reduced / scale, -1.0, 1.0)

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    mid_y = (height - 1) / 2.0
    amplitude = (height - 1) * 0.42
    points = []
    for index, value in enumerate(normalized):
        y = mid_y - float(value) * amplitude
        points.append((index, y))

    draw.line([(0, mid_y), (width - 1, mid_y)], fill=220, width=1)
    draw.line(points, fill=0, width=2)
    return image


def _resample_1d(values: np.ndarray, width: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=width, endpoint=True)
    return np.interp(x_new, x_old, values)
