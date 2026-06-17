from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .config import EPSILON
from .models import FeatureSet


def extract_features(samples: np.ndarray, sample_rate: int) -> FeatureSet:
    if samples.size == 0:
        raise ValueError("Audio clip is empty.")

    nperseg = min(2048, max(256, _largest_power_of_two(min(samples.size, 2048))))
    noverlap = max(0, nperseg // 2)
    freqs, _, stft = signal.stft(
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
    mean_spectrum = power.mean(axis=1) if power.ndim == 2 else power

    spectral_centroid = _spectral_centroid(freqs, mean_spectrum)
    spectral_bandwidth = _spectral_bandwidth(freqs, mean_spectrum, spectral_centroid)
    spectral_rolloff = _spectral_rolloff(freqs, mean_spectrum, 0.85)
    dominant_frequency, dominant_index = _dominant_frequency(freqs, mean_spectrum)
    harmonic_strength = _harmonic_strength(mean_spectrum, dominant_index)
    high_frequency_ratio = _high_frequency_ratio(freqs, mean_spectrum, sample_rate)
    burst_count = _burst_count(samples)
    log_mel_mean, log_mel_std = _log_mel_stats(power, sample_rate)

    values = {
        "duration_seconds": float(samples.size) / float(sample_rate),
        "rms_energy": float(np.sqrt(np.mean(samples**2))),
        "zero_crossing_rate": _zero_crossing_rate(samples),
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "dominant_frequency": dominant_frequency,
        "harmonic_strength": harmonic_strength,
        "high_frequency_ratio": high_frequency_ratio,
        "burst_count": float(burst_count),
        "log_mel_mean": log_mel_mean,
        "log_mel_std": log_mel_std,
    }
    return FeatureSet(sample_rate=sample_rate, values=values)


def _largest_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _zero_crossing_rate(samples: np.ndarray) -> float:
    signs = np.signbit(samples)
    return float(np.mean(signs[1:] != signs[:-1]))


def _spectral_centroid(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    return float(np.sum(freqs * spectrum) / np.sum(spectrum))


def _spectral_bandwidth(freqs: np.ndarray, spectrum: np.ndarray, centroid: float) -> float:
    variance = np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)
    return float(math.sqrt(max(variance, 0.0)))


def _spectral_rolloff(freqs: np.ndarray, spectrum: np.ndarray, threshold: float) -> float:
    cumulative = np.cumsum(spectrum)
    cutoff = threshold * cumulative[-1]
    index = int(np.searchsorted(cumulative, cutoff))
    index = min(index, len(freqs) - 1)
    return float(freqs[index])


def _dominant_frequency(freqs: np.ndarray, spectrum: np.ndarray) -> tuple[float, int]:
    if len(spectrum) <= 1:
        return 0.0, 0
    start_index = 1 if len(spectrum) > 1 else 0
    relative_index = int(np.argmax(spectrum[start_index:]))
    absolute_index = start_index + relative_index
    return float(freqs[absolute_index]), absolute_index


def _harmonic_strength(spectrum: np.ndarray, peak_index: int) -> float:
    if spectrum.size == 0:
        return 0.0
    lo = max(0, peak_index - 2)
    hi = min(spectrum.size, peak_index + 3)
    peak_energy = float(np.sum(spectrum[lo:hi]))
    total_energy = float(np.sum(spectrum))
    return peak_energy / total_energy if total_energy > EPSILON else 0.0


def _high_frequency_ratio(freqs: np.ndarray, spectrum: np.ndarray, sample_rate: int) -> float:
    cutoff = 0.35 * (sample_rate / 2.0)
    high_energy = float(np.sum(spectrum[freqs >= cutoff]))
    total_energy = float(np.sum(spectrum))
    return high_energy / total_energy if total_energy > EPSILON else 0.0


def _burst_count(samples: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> int:
    if samples.size < frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        frames = padded[None, :]
    else:
        starts = range(0, samples.size - frame_length + 1, hop_length)
        frames = np.stack([samples[start : start + frame_length] for start in starts], axis=0)

    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    median = float(np.median(frame_rms))
    mad = float(np.median(np.abs(frame_rms - median))) + EPSILON
    threshold = median + 3.0 * mad

    count = 0
    for index, value in enumerate(frame_rms):
        prev_value = frame_rms[index - 1] if index > 0 else -np.inf
        next_value = frame_rms[index + 1] if index < len(frame_rms) - 1 else -np.inf
        if value > threshold and value >= prev_value and value >= next_value:
            count += 1
    return int(count)


def _log_mel_stats(power: np.ndarray, sample_rate: int, n_mels: int = 20) -> tuple[float, float]:
    if power.ndim != 2 or power.shape[0] < 2:
        return 0.0, 0.0

    n_fft_bins = power.shape[0]
    freq_bins = np.linspace(0.0, sample_rate / 2.0, n_fft_bins)
    mel_filters = _mel_filterbank(freq_bins, sample_rate, n_mels=n_mels)
    mel_power = np.maximum(mel_filters @ power, EPSILON)
    log_mel = np.log10(mel_power)
    return float(np.mean(log_mel)), float(np.std(log_mel))


def _mel_filterbank(freq_bins: np.ndarray, sample_rate: int, n_mels: int) -> np.ndarray:
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, len(freq_bins)))
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
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)
