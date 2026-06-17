from __future__ import annotations

from .models import FeatureComparison, Observation


def build_observations(comparisons: dict[str, FeatureComparison], max_items: int = 5) -> list[Observation]:
    observations: list[Observation] = []

    noise_combo = (
        comparisons["high_frequency_ratio"].salience >= 1.0
        and comparisons["high_frequency_ratio"].direction == "higher"
        and comparisons["harmonic_strength"].salience >= 0.75
        and comparisons["harmonic_strength"].direction == "lower"
    )
    if noise_combo:
        salience = max(
            comparisons["high_frequency_ratio"].salience,
            comparisons["harmonic_strength"].salience,
        )
        observations.append(
            Observation(
                feature="noise_tonality_combo",
                text="The sound appears noisier and less tonal than the normal baseline.",
                severity=_severity_from_salience(salience),
                salience=salience,
            )
        )

    for feature, comparison in sorted(
        comparisons.items(),
        key=lambda item: item[1].salience,
        reverse=True,
    ):
        if comparison.severity == "stable":
            continue

        text = _describe_feature(comparison)
        if not text:
            continue

        observations.append(
            Observation(
                feature=feature,
                text=text,
                severity=comparison.severity,
                salience=comparison.salience,
            )
        )

        if len(observations) >= max_items:
            break

    if len(observations) < max_items:
        dominant = comparisons["dominant_frequency"]
        observations.append(
            Observation(
                feature="dominant_frequency_context",
                text=f"The dominant frequency is around {dominant.value:.0f} Hz relative to the same-case baseline.",
                severity="stable",
                salience=dominant.salience,
            )
        )

    if not observations:
        observations.append(
            Observation(
                feature="overall",
                text="Most measured acoustic features remain close to the normal baseline.",
                severity="stable",
                salience=0.0,
            )
        )

    return observations[:max_items]


def _describe_feature(comparison: FeatureComparison) -> str | None:
    severity = _severity_adverb(comparison.severity)
    degree = _comparative_degree(comparison.severity)
    adjective = _severity_adjective(comparison.severity)

    if comparison.feature == "rms_energy":
        return f"Overall energy is {severity} {comparison.direction} than the normal baseline."
    if comparison.feature == "zero_crossing_rate":
        return f"Zero-crossing activity is {severity} {comparison.direction}, suggesting a rougher waveform shape."
    if comparison.feature == "spectral_centroid":
        shifted = "upward" if comparison.direction == "higher" else "downward"
        return f"Spectral centroid is shifted {shifted} with {adjective} strength."
    if comparison.feature == "spectral_bandwidth":
        width = "wider" if comparison.direction == "higher" else "narrower"
        return f"Spectral bandwidth is {severity} {width} than baseline."
    if comparison.feature == "spectral_rolloff":
        return f"Spectral rolloff is {severity} {comparison.direction} than baseline."
    if comparison.feature == "dominant_frequency":
        shifted = "upward" if comparison.direction == "higher" else "downward"
        return f"Dominant frequency is shifted {shifted} to about {comparison.value:.0f} Hz."
    if comparison.feature == "harmonic_strength":
        tonality = "stronger" if comparison.direction == "higher" else "weaker"
        return f"The tonal component is {degree} {tonality} than normal."
    if comparison.feature == "high_frequency_ratio":
        return f"High-frequency energy is {severity} {comparison.direction} than baseline."
    if comparison.feature == "burst_count":
        burst_word = "more" if comparison.direction == "higher" else "fewer"
        return f"{severity.capitalize()} evidence of {burst_word} impulsive bursts is present."
    if comparison.feature == "log_mel_mean":
        return f"Average log-mel energy is {severity} {comparison.direction} than normal."
    if comparison.feature == "log_mel_std":
        variation = "higher" if comparison.direction == "higher" else "lower"
        return f"Time-frequency variation is {severity} {variation} than baseline."
    return None


def _severity_from_salience(salience: float) -> str:
    if salience >= 2.5:
        return "strong"
    if salience >= 1.5:
        return "moderate"
    if salience >= 0.75:
        return "mild"
    return "stable"


def _severity_adverb(severity: str) -> str:
    if severity == "strong":
        return "strongly"
    if severity == "moderate":
        return "moderately"
    if severity == "mild":
        return "mildly"
    return "lightly"


def _severity_adjective(severity: str) -> str:
    if severity == "strong":
        return "strong"
    if severity == "moderate":
        return "moderate"
    if severity == "mild":
        return "mild"
    return "light"


def _comparative_degree(severity: str) -> str:
    if severity == "strong":
        return "much"
    if severity == "moderate":
        return "noticeably"
    if severity == "mild":
        return "slightly"
    return "lightly"
