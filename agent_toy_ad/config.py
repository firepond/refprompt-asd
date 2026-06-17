from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/ToyConveyor")
DEFAULT_BASELINE_LIMIT = 12
DEFAULT_CNT_WINDOW_SECONDS = 10.0
DEFAULT_CNT_HOP_SECONDS = 10.0
DEFAULT_CNT_BASELINE_WINDOWS_PER_CLIP = 5
DEFAULT_CNT_TOP_K_WINDOWS = 3
EPSILON = 1e-8

FEATURE_ORDER = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "dominant_frequency",
    "harmonic_strength",
    "high_frequency_ratio",
    "burst_count",
    "log_mel_mean",
    "log_mel_std",
]

FEATURE_LABELS = {
    "rms_energy": "RMS energy",
    "zero_crossing_rate": "Zero-crossing rate",
    "spectral_centroid": "Spectral centroid",
    "spectral_bandwidth": "Spectral bandwidth",
    "spectral_rolloff": "Spectral rolloff",
    "dominant_frequency": "Dominant frequency",
    "harmonic_strength": "Harmonic strength",
    "high_frequency_ratio": "High-frequency energy ratio",
    "burst_count": "Burst count",
    "log_mel_mean": "Log-mel mean",
    "log_mel_std": "Log-mel std",
}

FEATURE_WEIGHTS = {
    "rms_energy": 0.9,
    "zero_crossing_rate": 0.6,
    "spectral_centroid": 1.2,
    "spectral_bandwidth": 1.0,
    "spectral_rolloff": 0.8,
    "dominant_frequency": 0.7,
    "harmonic_strength": 1.1,
    "high_frequency_ratio": 1.3,
    "burst_count": 1.3,
    "log_mel_mean": 0.4,
    "log_mel_std": 0.8,
}


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None
    model: str
    base_url: str
    timeout_seconds: float

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def get_openai_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
    )


def get_ollama_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:35b-a3b-q4_K_M"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
        timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
    )


def get_gemini_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")),
    )
