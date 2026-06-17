from __future__ import annotations

import numpy as np

from .config import EPSILON, FEATURE_ORDER
from .models import BaselineProfile, FeatureComparison, FeatureSet


def build_baseline_profile(feature_sets: list[FeatureSet], clip_paths: list[str]) -> BaselineProfile:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    if not feature_sets:
        for feature in FEATURE_ORDER:
            means[feature] = 0.0
            stds[feature] = 0.0
        return BaselineProfile(clip_paths=clip_paths, means=means, stds=stds)

    for feature in FEATURE_ORDER:
        values = np.array([feature_set.values[feature] for feature_set in feature_sets], dtype=np.float64)
        means[feature] = float(np.mean(values))
        stds[feature] = float(np.std(values))

    return BaselineProfile(clip_paths=clip_paths, means=means, stds=stds)


def compare_to_baseline(target: FeatureSet, baseline: BaselineProfile) -> dict[str, FeatureComparison]:
    comparisons: dict[str, FeatureComparison] = {}

    for feature in FEATURE_ORDER:
        value = target.values[feature]
        baseline_mean = baseline.means[feature]
        baseline_std = baseline.stds[feature]
        delta = value - baseline_mean
        relative_delta = delta / (abs(baseline_mean) + EPSILON)

        if baseline_std > EPSILON:
            z_score = delta / baseline_std
        else:
            z_score = relative_delta

        salience = abs(z_score)
        severity = _severity_label(salience)
        direction = "stable"
        if salience >= 0.5:
            direction = "higher" if delta > 0 else "lower"

        comparisons[feature] = FeatureComparison(
            feature=feature,
            value=float(value),
            baseline_mean=float(baseline_mean),
            baseline_std=float(baseline_std),
            delta=float(delta),
            relative_delta=float(relative_delta),
            z_score=float(z_score),
            salience=float(salience),
            severity=severity,
            direction=direction,
        )

    return comparisons


def build_absolute_feature_comparisons(target: FeatureSet) -> dict[str, FeatureComparison]:
    comparisons: dict[str, FeatureComparison] = {}

    for feature in FEATURE_ORDER:
        value = float(target.values[feature])
        comparisons[feature] = FeatureComparison(
            feature=feature,
            value=value,
            baseline_mean=0.0,
            baseline_std=0.0,
            delta=value,
            relative_delta=value,
            z_score=0.0,
            salience=0.0,
            severity="stable",
            direction="stable",
        )

    return comparisons


def _severity_label(salience: float) -> str:
    if salience >= 2.5:
        return "strong"
    if salience >= 1.5:
        return "moderate"
    if salience >= 0.75:
        return "mild"
    return "stable"
