from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import signal

from .config import EPSILON


def render_log_mel_spectrogram_png(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = 64,
    width: int = 512,
    height: int = 256,
) -> bytes:
    image = build_log_mel_spectrogram_image(
        samples,
        sample_rate,
        n_mels=n_mels,
        width=width,
        height=height,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_log_mel_spectrogram_image(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = 64,
    width: int = 512,
    height: int = 256,
) -> Image.Image:
    if samples.size == 0:
        raise ValueError("Audio clip is empty.")

    nperseg = min(2048, max(256, _largest_power_of_two(min(samples.size, 2048))))
    noverlap = max(0, nperseg // 2)
    _, _, stft = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft)
    power = magnitude**2 + EPSILON

    mel_power = _mel_filterbank(power.shape[0], sample_rate, n_mels=n_mels) @ power
    log_mel = np.log10(np.maximum(mel_power, EPSILON))
    normalized = _normalize_image(log_mel)
    flipped = np.flipud(normalized)
    pixels = np.clip(np.round(flipped * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _largest_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _normalize_image(values: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(values, 5.0))
    hi = float(np.percentile(values, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(values, lo, hi)
    return (clipped - lo) / (hi - lo)


def _mel_filterbank(n_fft_bins: int, sample_rate: int, *, n_mels: int) -> np.ndarray:
    freq_bins = np.linspace(0.0, sample_rate / 2.0, n_fft_bins)
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, len(freq_bins)), dtype=np.float64)
    for mel_index in range(n_mels):
        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]
        if center <= left or right <= center:
            continue

        left_mask = (freq_bins >= left) & (freq_bins <= center)
        right_mask = (freq_bins >= center) & (freq_bins <= right)
        filters[mel_index, left_mask] = (freq_bins[left_mask] - left) / (center - left + EPSILON)
        filters[mel_index, right_mask] = (right - freq_bins[right_mask]) / (right - center + EPSILON)
    return filters


def _hz_to_mel(freq_hz: float) -> float:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)
