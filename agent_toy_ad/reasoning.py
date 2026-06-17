from __future__ import annotations

import json
import os
import re
from base64 import b64encode
from urllib import error, request

import numpy as np

from .config import FEATURE_WEIGHTS, OpenAIConfig, get_gemini_config, get_ollama_config, get_openai_config
from .models import ClipMetadata, FeatureComparison, FeatureSet, Observation, ReasoningResult

CAUSE_CHOICES = {
    "tension_pulley_excessive_tension",
    "tail_pulley_excessive_tension",
    "tail_pulley_removed",
    "belt_attached_metallic_object",
    "over_voltage",
    "under_voltage",
    "none",
}

EXCLUDED_LOG_MEL_FEATURES = {"log_mel_mean", "log_mel_std"}
EXCLUDED_LOG_MEL_STD_FEATURES = {"log_mel_std"}
FEATURE_CONFIDENCE_FEATURES = [
    "log_mel_std",
    "harmonic_strength",
    "burst_count",
    "zero_crossing_rate",
    "spectral_centroid",
    "high_frequency_ratio",
]


def reason_about_clip(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    *,
    mode: str = "auto",
    exclude_features: set[str] | None = None,
    baseline_feature_sets: list[FeatureSet] | None = None,
    image_png: bytes | None = None,
    secondary_image_png: bytes | None = None,
) -> ReasoningResult:
    rule_result = _rule_based_reasoning(metadata, comparisons, observations)

    if mode == "rule":
        return rule_result

    if mode == "ollama":
        return _openai_reasoning(
            metadata,
            comparisons,
            observations,
            rule_result,
            config=get_ollama_config(),
            exclude_features=exclude_features,
        )
    if mode == "ollama_conservative":
        return _ollama_conservative_reasoning(
            metadata,
            comparisons,
            observations,
            rule_result,
            config=get_ollama_config(),
            exclude_features=exclude_features,
        )
    if mode == "gemini":
        return _gemini_reasoning(
            metadata,
            comparisons,
            observations,
            rule_result,
            config=get_gemini_config(),
            exclude_features=exclude_features,
        )

    config = get_openai_config()
    if mode in {
        "openai",
        "openai_no_logmel",
        "openai_no_logmel_std",
        "openai_analysis",
        "openai_cause",
        "openai_cause_prior",
        "openai_image",
        "openai_image_describe_label",
        "openai_raw_waveform_image",
        "openai_waveform_image",
        "openai_dual_image",
        "openai_dsp_image",
        "openai_distribution",
        "openai_chebyshev",
        "openai_feature_confidence",
        "openai_baseline_free_dsp",
        "openai_zero_shot_dsp_conservative",
        "openai_zero_shot_dsp_normal_default",
        "openai_zero_shot_dsp_fault_plausibility",
        "openai_zero_shot_cause_scores_b",
        "openai_zero_shot_cause_scores_b_reordered",
        "openai_zero_shot_multiclass",
    } and not config.enabled:
        raise RuntimeError("OpenAI reasoning was requested, but OPENAI_API_KEY is not configured.")
    gemini_config = get_gemini_config()
    if mode == "gemini" and not gemini_config.enabled:
        raise RuntimeError("Gemini reasoning was requested, but GEMINI_API_KEY is not configured.")

    if mode == "auto" and not config.enabled:
        return rule_result

    try:
        if mode == "openai_analysis":
            return _openai_analysis_reasoning(metadata, comparisons, observations, rule_result, config=config)
        if mode == "openai_no_logmel":
            return _openai_no_logmel_reasoning(metadata, comparisons, observations, rule_result, config=config)
        if mode == "openai_no_logmel_std":
            return _openai_no_logmel_std_reasoning(metadata, comparisons, observations, rule_result, config=config)
        if mode == "openai_cause":
            return _openai_cause_reasoning(metadata, comparisons, observations, rule_result, config=config)
        if mode == "openai_cause_prior":
            return _openai_cause_prior_reasoning(metadata, comparisons, observations, rule_result, config=config)
        if mode == "openai_image":
            if image_png is None:
                raise ValueError("openai_image mode requires a target spectrogram image.")
            return _openai_image_reasoning(metadata, image_png, rule_result, config=config)
        if mode == "openai_image_describe_label":
            if image_png is None:
                raise ValueError("openai_image_describe_label mode requires a target spectrogram image.")
            return _openai_image_describe_label_reasoning(metadata, image_png, rule_result, config=config)
        if mode == "openai_raw_waveform_image":
            if image_png is None:
                raise ValueError("openai_raw_waveform_image mode requires a target waveform image.")
            return _openai_raw_waveform_image_reasoning(metadata, image_png, rule_result, config=config)
        if mode == "openai_waveform_image":
            if image_png is None:
                raise ValueError("openai_waveform_image mode requires a target waveform image.")
            return _openai_waveform_image_reasoning(metadata, image_png, rule_result, config=config)
        if mode == "openai_dual_image":
            if image_png is None or secondary_image_png is None:
                raise ValueError("openai_dual_image mode requires both waveform and spectrogram images.")
            return _openai_dual_image_reasoning(
                metadata,
                waveform_image_png=image_png,
                spectrogram_image_png=secondary_image_png,
                fallback=rule_result,
                config=config,
            )
        if mode == "openai_dsp_image":
            if image_png is None:
                raise ValueError("openai_dsp_image mode requires a target spectrogram image.")
            return _openai_dsp_image_reasoning(
                metadata,
                comparisons,
                observations,
                image_png,
                rule_result,
                config=config,
            )
        if mode == "openai_distribution":
            return _openai_distribution_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                baseline_feature_sets=baseline_feature_sets or [],
                config=config,
            )
        if mode == "openai_feature_confidence":
            return _openai_feature_confidence_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
                exclude_features=exclude_features,
            )
        if mode == "openai_chebyshev":
            return _openai_chebyshev_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                baseline_feature_sets=baseline_feature_sets or [],
                config=config,
            )
        if mode == "openai_baseline_free_dsp":
            return _openai_baseline_free_dsp_reasoning(
                metadata,
                comparisons,
                observations,
                rule_result,
                config=config,
            )
        if mode == "openai_zero_shot_dsp_conservative":
            return _openai_zero_shot_dsp_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
                variant="conservative",
            )
        if mode == "openai_zero_shot_dsp_normal_default":
            return _openai_zero_shot_dsp_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
                variant="normal_default",
            )
        if mode == "openai_zero_shot_dsp_fault_plausibility":
            return _openai_zero_shot_dsp_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
                variant="fault_plausibility",
            )
        if mode == "openai_zero_shot_cause_scores_b":
            return _openai_zero_shot_cause_scores_b_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
            )
        if mode == "openai_zero_shot_cause_scores_b_reordered":
            return _openai_zero_shot_cause_scores_b_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
                reordered=True,
            )
        if mode == "openai_zero_shot_multiclass":
            return _openai_zero_shot_multiclass_reasoning(
                metadata,
                comparisons,
                observations,
                fallback=rule_result,
                config=config,
            )
        return _openai_reasoning(
            metadata,
            comparisons,
            observations,
            rule_result,
            config=config,
            exclude_features=exclude_features,
        )
    except Exception:
        if mode in {
            "openai",
            "openai_no_logmel",
            "openai_no_logmel_std",
            "openai_analysis",
            "openai_cause",
            "openai_cause_prior",
            "openai_image",
            "openai_image_describe_label",
            "openai_raw_waveform_image",
            "openai_waveform_image",
            "openai_dual_image",
            "openai_dsp_image",
            "openai_distribution",
            "openai_chebyshev",
            "openai_feature_confidence",
            "openai_baseline_free_dsp",
            "openai_zero_shot_dsp_conservative",
            "openai_zero_shot_dsp_normal_default",
            "openai_zero_shot_dsp_fault_plausibility",
            "openai_zero_shot_cause_scores_b",
            "openai_zero_shot_cause_scores_b_reordered",
            "openai_zero_shot_multiclass",
        }:
            raise
        rule_result.raw_response = "OpenAI reasoning unavailable; used rule-based fallback."
        return rule_result


def _rule_based_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
) -> ReasoningResult:
    metrics = compute_rule_anomaly_metrics(comparisons)
    anomaly_score = metrics["anomaly_score"]
    very_strong_count = metrics["very_strong_count"]

    if anomaly_score >= 2.75 or very_strong_count >= 2:
        prediction = "abnormal"
    else:
        prediction = "normal"

    if anomaly_score >= 3.5:
        confidence = "high"
    elif anomaly_score >= 2.25:
        confidence = "medium"
    else:
        confidence = "low"

    evidence = [observation.text for observation in observations[:3]]
    possible_causes = _possible_causes(comparisons, prediction)
    suggested_checks = _suggested_checks(possible_causes, prediction)

    if prediction == "abnormal":
        explanation = (
            "The clip looks abnormal relative to the same-case normal baseline because several "
            "acoustic descriptors move together in a suspicious direction. "
            + " ".join(evidence[:2])
        ).strip()
    else:
        explanation = (
            "The clip looks closer to normal because the extracted features stay reasonably close "
            "to the same-case baseline, with no strong cluster of abnormal shifts. "
            + " ".join(evidence[:2])
        ).strip()

    return ReasoningResult(
        source="rule-based",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        evidence=evidence,
        possible_causes=possible_causes,
        suggested_checks=suggested_checks,
    )


def compute_rule_anomaly_metrics(comparisons: dict[str, FeatureComparison]) -> dict[str, float]:
    weighted_saliences = []
    for feature, comparison in comparisons.items():
        weight = FEATURE_WEIGHTS.get(feature, 0.0)
        weighted_saliences.append(min(comparison.salience, 4.0) * weight)

    top_weighted_saliences = sorted(weighted_saliences, reverse=True)[:4]
    anomaly_score = (
        sum(top_weighted_saliences) / len(top_weighted_saliences) if top_weighted_saliences else 0.0
    )
    very_strong_count = sum(1 for comparison in comparisons.values() if comparison.salience >= 2.5)
    return {
        "anomaly_score": anomaly_score,
        "very_strong_count": float(very_strong_count),
    }


def _possible_causes(
    comparisons: dict[str, FeatureComparison],
    prediction: str,
) -> list[str]:
    if prediction != "abnormal":
        return []

    causes: list[str] = []

    if (
        comparisons["high_frequency_ratio"].direction == "higher"
        and comparisons["harmonic_strength"].direction == "lower"
    ):
        causes.append("possible broadband friction or rough-contact noise increase")

    if comparisons["burst_count"].direction == "higher" and comparisons["burst_count"].salience >= 1.0:
        causes.append("possible impact-like or intermittent-contact abnormality")

    if (
        comparisons["dominant_frequency"].salience >= 1.0
        or comparisons["spectral_centroid"].direction == "higher"
    ):
        causes.append("possible unstable rotation or speed-related acoustic change")

    if (
        comparisons["harmonic_strength"].direction == "higher"
        and comparisons["spectral_bandwidth"].direction == "lower"
    ):
        causes.append("possible resonance-like tonal concentration or a sharper rotating component")

    if (
        comparisons["rms_energy"].direction == "higher"
        and comparisons["spectral_bandwidth"].direction == "higher"
    ):
        causes.append("possible load increase or rougher mechanical interaction")

    return causes[:3]


def _suggested_checks(possible_causes: list[str], prediction: str) -> list[str]:
    if prediction != "abnormal":
        return ["Continue routine monitoring and compare against more clips if the sound changes."]

    checks: list[str] = []
    for cause in possible_causes:
        if "friction" in cause:
            checks.append("Inspect rollers, bearings, lubrication, and contact surfaces for rough running.")
        elif "impact-like" in cause:
            checks.append("Check for loose parts, intermittent contact, or debris along the conveyor path.")
        elif "rotation" in cause:
            checks.append("Check drive speed stability, belt tension, and alignment.")
        elif "resonance-like" in cause:
            checks.append("Inspect rotating elements, guards, and mounts for resonance or concentrated tonal vibration.")
        elif "load increase" in cause:
            checks.append("Inspect load path, mounting rigidity, and mechanical tension.")

    if not checks:
        checks.append("Inspect the machine around the channel location and compare with additional normal clips.")
    return checks[:3]


def _openai_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    exclude_features: set[str] | None = None,
) -> ReasoningResult:
    system_prompt, user_payload = build_label_only_prompt_payload(
        metadata,
        comparisons,
        observations,
        exclude_features=exclude_features,
    )

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=_source_from_base_url(config.base_url),
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _gemini_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    exclude_features: set[str] | None = None,
) -> ReasoningResult:
    system_prompt, user_payload = build_label_only_prompt_payload(
        metadata,
        comparisons,
        observations,
        exclude_features=exclude_features,
    )

    content = _gemini_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source="gemini",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _ollama_conservative_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    exclude_features: set[str] | None = None,
) -> ReasoningResult:
    system_prompt, user_payload = build_label_only_prompt_payload(
        metadata,
        comparisons,
        observations,
        exclude_features=exclude_features,
        conservative=True,
    )

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=_source_from_base_url(config.base_url),
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_no_logmel_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt, user_payload = build_label_only_prompt_payload(
        metadata,
        comparisons,
        observations,
        exclude_features=EXCLUDED_LOG_MEL_FEATURES,
    )

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=_source_from_base_url(config.base_url),
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_no_logmel_std_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt, user_payload = build_label_only_prompt_payload(
        metadata,
        comparisons,
        observations,
        exclude_features=EXCLUDED_LOG_MEL_STD_FEATURES,
    )

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=_source_from_base_url(config.base_url),
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def build_label_only_prompt_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    *,
    exclude_features: set[str] | None = None,
    conservative: bool = False,
) -> tuple[str, dict]:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "Possible abnormal causes in this dataset include: "
        "excessive tension at the tension pulley; "
        "excessive tension at the tail pulley; "
        "tail pulley removed; "
        "a metallic object attached to the belt; "
        "over voltage; "
        "under voltage. "
        "Use those causes only as domain context, not as a requirement that every abnormal clip match one cause exactly. "
        "Use only the supplied acoustic measurements and baseline statistics. "
        "Judge the evidence yourself from the numbers. "
    )
    if conservative:
        system_prompt += (
            "Normal operating variation can be substantial even within the same machine case and channel. "
            "Do not treat several closely related spectral features as independent evidence if they describe the same underlying shift. "
            "Call a clip abnormal only when there is clear and consistent support from multiple distinct kinds of evidence. "
            "If the evidence is mixed, modest, or uncertain, choose normal. "
        )
    system_prompt += (
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    del observations
    return system_prompt, _build_raw_feature_payload(
        metadata,
        comparisons,
        exclude_features=exclude_features,
    )


def _build_raw_feature_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    *,
    exclude_features: set[str] | None = None,
) -> dict:
    excluded = exclude_features or set()
    return {
        "clip": {
            "case": metadata.case,
            "mode": metadata.mode,
            "channel": metadata.channel,
        },
        "feature_statistics": {
            feature: {
                "value": round(comparison.value, 6),
                "baseline_mean": round(comparison.baseline_mean, 6),
                "baseline_std": round(comparison.baseline_std, 6),
                "delta": round(comparison.delta, 6),
                "z_score": round(comparison.z_score, 6),
            }
            for feature, comparison in comparisons.items()
            if feature not in excluded
        },
    }


def _build_selected_feature_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    *,
    selected_features: list[str],
) -> dict:
    return {
        "clip": {
            "case": metadata.case,
            "mode": metadata.mode,
            "channel": metadata.channel,
        },
        "selected_feature_statistics": {
            feature: {
                "value": round(comparisons[feature].value, 6),
                "baseline_mean": round(comparisons[feature].baseline_mean, 6),
                "baseline_std": round(comparisons[feature].baseline_std, 6),
                "delta": round(comparisons[feature].delta, 6),
                "z_score": round(comparisons[feature].z_score, 6),
            }
            for feature in selected_features
        },
    }


def _build_distribution_feature_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    baseline_feature_sets: list[FeatureSet],
) -> dict:
    baseline_count = len(baseline_feature_sets)
    feature_statistics: dict[str, dict] = {}
    abs_z_values = []

    for feature, comparison in comparisons.items():
        baseline_values = np.array(
            [float(feature_set.values[feature]) for feature_set in baseline_feature_sets],
            dtype=np.float64,
        )
        if baseline_values.size == 0:
            raise ValueError("Distribution-aware prompting requires non-empty baseline_feature_sets.")

        min_value = float(np.min(baseline_values))
        q25 = float(np.quantile(baseline_values, 0.25))
        median = float(np.median(baseline_values))
        q75 = float(np.quantile(baseline_values, 0.75))
        max_value = float(np.max(baseline_values))
        mad = float(np.median(np.abs(baseline_values - median)))
        robust_scale = 1.4826 * mad
        if robust_scale > 1e-8:
            robust_z_score = float((comparison.value - median) / robust_scale)
        else:
            robust_z_score = 0.0

        percentile = float(np.mean(baseline_values <= comparison.value))
        iqr = q75 - q25
        feature_statistics[feature] = {
            "value": round(comparison.value, 6),
            "baseline_count": baseline_count,
            "baseline_min": round(min_value, 6),
            "baseline_q25": round(q25, 6),
            "baseline_median": round(median, 6),
            "baseline_q75": round(q75, 6),
            "baseline_max": round(max_value, 6),
            "baseline_mean": round(comparison.baseline_mean, 6),
            "baseline_std": round(comparison.baseline_std, 6),
            "baseline_mad": round(mad, 6),
            "baseline_iqr": round(iqr, 6),
            "delta": round(comparison.delta, 6),
            "z_score": round(comparison.z_score, 6),
            "robust_z_score": round(robust_z_score, 6),
            "query_percentile_in_baseline": round(percentile, 6),
        }
        abs_z_values.append(abs(comparison.z_score))

    abs_z_values = np.array(abs_z_values, dtype=np.float64)
    top_abs_z = sorted(
        ((feature, abs(comparison.z_score)) for feature, comparison in comparisons.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    return {
        "clip": {
            "case": metadata.case,
            "mode": metadata.mode,
            "channel": metadata.channel,
        },
        "aggregate_statistics": {
            "baseline_count": baseline_count,
            "mean_abs_z_score": round(float(np.mean(abs_z_values)), 6),
            "max_abs_z_score": round(float(np.max(abs_z_values)), 6),
            "count_abs_z_ge_1": int(np.sum(abs_z_values >= 1.0)),
            "count_abs_z_ge_2": int(np.sum(abs_z_values >= 2.0)),
            "count_abs_z_ge_3": int(np.sum(abs_z_values >= 3.0)),
            "top_abs_z_features": [
                {"feature": feature, "abs_z_score": round(float(score), 6)}
                for feature, score in top_abs_z
            ],
        },
        "feature_statistics": feature_statistics,
    }


def _build_chebyshev_feature_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    baseline_feature_sets: list[FeatureSet],
) -> dict:
    payload = _build_raw_feature_payload(metadata, comparisons)
    payload["chebyshev_distance_summary"] = _compute_chebyshev_distance_summary(
        comparisons,
        baseline_feature_sets,
    )
    return payload


def _compute_chebyshev_distance_summary(
    comparisons: dict[str, FeatureComparison],
    baseline_feature_sets: list[FeatureSet],
) -> dict:
    if not baseline_feature_sets:
        raise ValueError("Chebyshev prompting requires non-empty baseline_feature_sets.")

    feature_names = list(comparisons.keys())
    query_vector = np.array(
        [float(comparisons[feature].value) for feature in feature_names],
        dtype=np.float64,
    )
    baseline_vectors = np.array(
        [
            [float(feature_set.values[feature]) for feature in feature_names]
            for feature_set in baseline_feature_sets
        ],
        dtype=np.float64,
    )

    baseline_center = baseline_vectors.mean(axis=0)
    query_abs_diffs = np.abs(query_vector - baseline_center)
    query_distance = float(np.max(query_abs_diffs))
    max_feature_index = int(np.argmax(query_abs_diffs))
    max_feature = feature_names[max_feature_index]

    internal_distances = []
    if len(baseline_vectors) == 1:
        internal_distances = [0.0]
    else:
        total = baseline_vectors.sum(axis=0)
        count = len(baseline_vectors)
        for index in range(count):
            loo_center = (total - baseline_vectors[index]) / float(count - 1)
            distance = float(np.max(np.abs(baseline_vectors[index] - loo_center)))
            internal_distances.append(distance)

    internal = np.array(internal_distances, dtype=np.float64)
    internal_mean = float(np.mean(internal))
    internal_std = float(np.std(internal))
    internal_median = float(np.median(internal))
    internal_q75 = float(np.quantile(internal, 0.75))
    internal_min = float(np.min(internal))
    internal_max = float(np.max(internal))
    percentile = float(np.mean(internal <= query_distance))
    ratio_to_mean = float(query_distance / (internal_mean + 1e-8))
    ratio_to_median = float(query_distance / (internal_median + 1e-8))

    return {
        "baseline_count": len(baseline_feature_sets),
        "metric": "chebyshev",
        "query_distance": round(query_distance, 6),
        "internal_min": round(internal_min, 6),
        "internal_mean": round(internal_mean, 6),
        "internal_std": round(internal_std, 6),
        "internal_median": round(internal_median, 6),
        "internal_q75": round(internal_q75, 6),
        "internal_max": round(internal_max, 6),
        "query_percentile_in_internal_distances": round(percentile, 6),
        "query_to_internal_mean_ratio": round(ratio_to_mean, 6),
        "query_to_internal_median_ratio": round(ratio_to_median, 6),
        "max_deviation_feature": max_feature,
        "max_deviation_abs_delta": round(float(query_abs_diffs[max_feature_index]), 6),
    }


def _build_absolute_feature_payload(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
) -> dict:
    return {
        "clip": {
            "case": metadata.case,
            "mode": metadata.mode,
            "channel": metadata.channel,
        },
        "feature_values": {
            feature: round(comparison.value, 6)
            for feature, comparison in comparisons.items()
        },
    }


def _openai_distribution_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    baseline_feature_sets: list[FeatureSet],
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "Possible abnormal causes in this dataset include: "
        "excessive tension at the tension pulley; "
        "excessive tension at the tail pulley; "
        "tail pulley removed; "
        "a metallic object attached to the belt; "
        "over voltage; "
        "under voltage. "
        "Use those causes only as domain context, not as a requirement that every abnormal clip match one cause exactly. "
        "You are given numeric feature values for the query clip plus a summarized statistical distribution of same-case same-channel normal references. "
        "Use the distribution information carefully: quantiles, percentiles, standard deviation, and robust deviation indicate how unusual the query is relative to the baseline distribution. "
        "Do not overreact to small shifts in a single feature. "
        "A clip should be abnormal only when the overall statistical evidence is clearly inconsistent with the normal reference distribution across multiple features. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    del observations
    user_payload = _build_distribution_feature_payload(metadata, comparisons, baseline_feature_sets)

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-distribution",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_feature_confidence_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    exclude_features: set[str] | None = None,
) -> ReasoningResult:
    del observations
    excluded = exclude_features or set()
    selected_features = [
        feature for feature in FEATURE_CONFIDENCE_FEATURES
        if feature in comparisons and feature not in excluded
    ]
    if not selected_features:
        return _openai_reasoning(
            metadata,
            comparisons,
            observations=[],
            fallback=fallback,
            config=config,
            exclude_features=exclude_features,
        )

    selected_feature_list = ", ".join(selected_features)
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "Use only the supplied acoustic measurements and baseline statistics. "
        "You are given a selected subset of features that have been identified as potentially important for this decision. "
        "For each selected feature, assign an anomaly-evidence confidence of low, medium, or high based only on how strongly that feature supports abnormality relative to the provided baseline. "
        "Then make a final normal/abnormal decision from the combined evidence. "
        "Do not overcall abnormal when evidence is mixed, redundant, or only mildly shifted across a few related features. "
        "If the evidence is limited or ambiguous, choose normal. "
        f"The selected features are: {selected_feature_list}. "
        "Return valid JSON with exactly these keys: prediction, confidence, feature_confidence. "
        "prediction must be either normal or abnormal. "
        "confidence must be low, medium, or high. "
        "feature_confidence must be an object whose keys are exactly the selected feature names and whose values are each low, medium, or high. "
        "Do not return markdown fences or extra text."
    )
    user_payload = _build_selected_feature_payload(
        metadata,
        comparisons,
        selected_features=selected_features,
    )

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])
    confidence = _normalize_confidence(parsed.get("confidence", fallback.confidence))
    feature_confidence = parsed.get("feature_confidence", {})
    evidence = []
    if isinstance(feature_confidence, dict):
        for feature in selected_features:
            if feature in feature_confidence:
                level = _normalize_cause_level(feature_confidence[feature])
                evidence.append(f"{feature}:{level}")

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-feature-confidence",
        prediction=prediction,
        confidence=confidence,
        explanation="",
        evidence=evidence,
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_chebyshev_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    baseline_feature_sets: list[FeatureSet],
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "Possible abnormal causes in this dataset include: "
        "excessive tension at the tension pulley; "
        "excessive tension at the tail pulley; "
        "tail pulley removed; "
        "a metallic object attached to the belt; "
        "over voltage; "
        "under voltage. "
        "Use those causes only as domain context, not as a requirement that every abnormal clip match one cause exactly. "
        "You are given per-feature query-versus-baseline statistics plus a Chebyshev distance summary. "
        "The Chebyshev summary compares the query's overall distance from the normal baseline center against the internal leave-one-out distance distribution of the normal references. "
        "Use that summary as a compact indicator of whether the query lies outside normal baseline variation. "
        "Do not overreact to a weak signal from one feature alone. "
        "A clip should be abnormal only when the overall evidence is clearly inconsistent with the normal reference distribution. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    del observations
    user_payload = _build_chebyshev_feature_payload(metadata, comparisons, baseline_feature_sets)

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-chebyshev",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_analysis_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are an assistant for machine sound anomaly interpretation. "
        "The target machine is a belt conveyor. "
        "Possible abnormal causes in this dataset include: "
        "excessive tension at the tension pulley; "
        "excessive tension at the tail pulley; "
        "tail pulley removed; "
        "a metallic object attached to the belt; "
        "over voltage; "
        "under voltage. "
        "Use those causes only as domain context, not as a requirement that every abnormal clip match one cause exactly. "
        "Use only the supplied acoustic measurements and baseline statistics. "
        "Do not assume access to filenames or ground-truth labels. "
        "Return valid JSON with keys prediction, confidence, analysis, evidence, cautions. "
        "prediction must be either normal or abnormal. "
        "analysis must be a short paragraph explaining the decision from the numeric deviations. "
        "evidence and cautions must be arrays of short strings. "
        "Do not wrap the JSON in markdown fences."
    )
    del observations
    user_payload = _build_raw_feature_payload(metadata, comparisons)

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])
    confidence = _normalize_confidence(parsed.get("confidence", fallback.confidence))

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-analysis",
        prediction=prediction,
        confidence=confidence,
        explanation=str(parsed.get("analysis", "")).strip(),
        evidence=[str(item) for item in parsed.get("evidence", [])],
        possible_causes=[],
        suggested_checks=[str(item) for item in parsed.get("cautions", [])],
        raw_response=content,
    )


def _openai_cause_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier for a belt conveyor. "
        "Possible abnormal causes in this dataset are limited to exactly these six choices: "
        "tension_pulley_excessive_tension, "
        "tail_pulley_excessive_tension, "
        "tail_pulley_removed, "
        "belt_attached_metallic_object, "
        "over_voltage, "
        "under_voltage. "
        "Use only the supplied acoustic measurements and baseline statistics. "
        "Judge the evidence yourself from the numbers. "
        "Return valid JSON with exactly these keys: prediction, confidence, anomaly_cause. "
        "prediction must be either normal or abnormal. "
        "confidence must be low, medium, or high. "
        "If prediction is abnormal, anomaly_cause must be exactly one of the six cause choices. "
        "If prediction is normal, anomaly_cause must be none. "
        "Do not return markdown fences or extra text."
    )
    del observations
    user_payload = _build_raw_feature_payload(metadata, comparisons)

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])
    confidence = _normalize_confidence(parsed.get("confidence", fallback.confidence))
    predicted_cause = _normalize_cause(parsed.get("anomaly_cause"), prediction)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-cause",
        prediction=prediction,
        confidence=confidence,
        explanation="",
        predicted_cause=predicted_cause,
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_cause_prior_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier for a belt conveyor. "
        "Possible abnormal causes in this dataset are limited to exactly these six choices: "
        "tension_pulley_excessive_tension, "
        "tail_pulley_excessive_tension, "
        "tail_pulley_removed, "
        "belt_attached_metallic_object, "
        "over_voltage, "
        "under_voltage. "
        "You may use the following soft prior knowledge about common acoustic tendencies, but you must still decide from the supplied numeric evidence rather than forcing a match. "
        "tension_pulley_excessive_tension often looks like sustained load-related change with stronger energy, stronger tonal concentration, and moderate narrowing of the spectrum. "
        "tail_pulley_excessive_tension often looks similar but may show clearer rotation-related shifts, stronger harmonics, and a stable but strained tonal pattern. "
        "tail_pulley_removed often looks like a large structural change with strong redistribution toward lower-frequency tonal energy, markedly stronger harmonics, and clearly reduced centroid, bandwidth, or rolloff. "
        "belt_attached_metallic_object often looks more irregular or impact-like, with stronger bursts, higher high-frequency content, wider bandwidth, or sharper local deviations. "
        "over_voltage often looks like a faster or harsher operating state, with higher dominant frequency, centroid, rolloff, or energy. "
        "under_voltage often looks like a slower or weaker operating state, with lower dominant frequency, centroid, rolloff, or energy. "
        "If the numeric evidence does not support any specific cause strongly, choose the closest cause only after deciding the clip is abnormal. "
        "Use only the supplied acoustic measurements and baseline statistics. "
        "Return valid JSON with exactly these keys: prediction, confidence, anomaly_cause. "
        "prediction must be either normal or abnormal. "
        "confidence must be low, medium, or high. "
        "If prediction is abnormal, anomaly_cause must be exactly one of the six cause choices. "
        "If prediction is normal, anomaly_cause must be none. "
        "Do not return markdown fences or extra text."
    )
    del observations
    user_payload = _build_raw_feature_payload(metadata, comparisons)

    content = _chat_completion_content(system_prompt, user_payload, config=config)
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])
    confidence = _normalize_confidence(parsed.get("confidence", fallback.confidence))
    predicted_cause = _normalize_cause(parsed.get("anomaly_cause"), prediction)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-cause-prior",
        prediction=prediction,
        confidence=confidence,
        explanation="",
        predicted_cause=predicted_cause,
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_image_reasoning(
    metadata: ClipMetadata,
    image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given a single grayscale log-mel spectrogram image generated from one IND audio clip. "
        "In this dataset, IND clips can include machine startup and shutdown, so time-varying changes near the beginning or end can be normal parts of the recording. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Judge from the time-frequency structure in the image whether the machine sound is more likely normal or abnormal. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    content = _chat_completion_with_image_content(
        system_prompt,
        image_png,
        metadata,
        image_description="single target log-mel spectrogram",
        config=config,
    )
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-image",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_image_describe_label_reasoning(
    metadata: ClipMetadata,
    image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given a single grayscale log-mel spectrogram image generated from one IND audio clip. "
        "In this dataset, IND clips can include machine startup and shutdown, so time-varying changes near the beginning or end can be normal parts of the recording. "
        "First describe the visible spectrogram characteristics in a short phrase, focusing on acoustic image traits such as low-frequency band strength, harmonic banding, broadband texture, periodicity, or localized changes. "
        "Then classify the clip as normal or abnormal. "
        "Return valid JSON with exactly these keys: image_features, prediction. "
        "image_features must be a short phrase. "
        "prediction must be either normal or abnormal. "
        "Do not return markdown fences or extra text."
    )
    content = _chat_completion_with_image_content(
        system_prompt,
        image_png,
        metadata,
        image_description="single target log-mel spectrogram",
        config=config,
    )
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-image-describe-label",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation=str(parsed.get("image_features", "")).strip(),
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_waveform_image_reasoning(
    metadata: ClipMetadata,
    image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given a single grayscale time-series waveform image generated from one IND audio clip after Savitzky-Golay smoothing. "
        "In this dataset, IND clips can include machine startup and shutdown, so envelope changes near the beginning or end can be normal parts of the recording. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Judge from the overall temporal shape and fluctuation pattern in the image whether the machine sound is more likely normal or abnormal. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    content = _chat_completion_with_image_content(
        system_prompt,
        image_png,
        metadata,
        image_description="single target smoothed waveform image",
        config=config,
    )
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-waveform-image",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_raw_waveform_image_reasoning(
    metadata: ClipMetadata,
    image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given a single grayscale raw time-series waveform image generated from one IND audio clip. "
        "In this dataset, IND clips can include machine startup and shutdown, so envelope changes near the beginning or end can be normal parts of the recording. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Judge from the overall raw temporal shape and fluctuation pattern in the image whether the machine sound is more likely normal or abnormal. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    content = _chat_completion_with_image_content(
        system_prompt,
        image_png,
        metadata,
        image_description="single target raw waveform image",
        config=config,
    )
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-raw-waveform-image",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_dual_image_reasoning(
    metadata: ClipMetadata,
    waveform_image_png: bytes,
    spectrogram_image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given two grayscale images generated from the same IND audio clip after Savitzky-Golay smoothing: "
        "a smoothed waveform image and a smoothed log-mel spectrogram image. "
        "In this dataset, IND clips can include machine startup and shutdown, so boundary changes near the beginning or end can be normal parts of the recording. "
        "Use both images together. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Judge from the temporal shape and the time-frequency structure whether the machine sound is more likely normal or abnormal. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    content = _chat_completion_with_images_content(
        system_prompt,
        {
            "clip": {
                "case": metadata.case,
                "mode": metadata.mode,
                "channel": metadata.channel,
            },
            "image_descriptions": [
                "smoothed waveform image",
                "smoothed target log-mel spectrogram",
            ],
        },
        images=[
            ("smoothed waveform image", waveform_image_png),
            ("smoothed target log-mel spectrogram", spectrogram_image_png),
        ],
        config=config,
    )
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-dual-image",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_dsp_image_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    image_png: bytes,
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    del observations
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "You are given structured DSP feature statistics relative to a normal baseline, together with a single grayscale log-mel spectrogram image generated from the same IND audio clip after Savitzky-Golay smoothing. "
        "In this dataset, IND clips can include machine startup and shutdown, so image changes near the beginning or end can be normal parts of the recording. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Use both the numeric deviations and the image. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    user_payload = _build_raw_feature_payload(metadata, comparisons)
    content = _chat_completion_with_payload_and_images_content(
        system_prompt,
        user_payload,
        images=[("smoothed target log-mel spectrogram", image_png)],
        config=config,
    )
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-dsp-image",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_baseline_free_dsp_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    del observations
    system_prompt = (
        "You are a machine sound anomaly classifier. "
        "The target machine is a belt conveyor. "
        "Possible abnormal causes in this dataset include: "
        "excessive tension at the tension pulley; "
        "excessive tension at the tail pulley; "
        "tail pulley removed; "
        "a metallic object attached to the belt; "
        "over voltage; "
        "under voltage. "
        "You are given only the absolute DSP feature values extracted from one audio clip. "
        "No normal reference or baseline statistics are provided. "
        "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
        "Judge whether this single clip is more likely normal or abnormal for a belt conveyor. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    user_payload = _build_absolute_feature_payload(metadata, comparisons)
    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-baseline-free-dsp",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_zero_shot_dsp_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    variant: str,
) -> ReasoningResult:
    del observations
    if variant == "conservative":
        system_prompt = (
            "You are a machine sound anomaly classifier. "
            "The target machine is a belt conveyor. "
            "Possible abnormal causes in this dataset include: "
            "excessive tension at the tension pulley; "
            "excessive tension at the tail pulley; "
            "tail pulley removed; "
            "a metallic object attached to the belt; "
            "over voltage; "
            "under voltage. "
            "You are given only the absolute DSP feature values extracted from one audio clip. "
            "No normal reference or baseline statistics are provided. "
            "In this zero-shot setting, machine-to-machine variation can be large because different conveyor instances can have different detailed structures and recording conditions. "
            "Do not treat unusual absolute values alone as sufficient evidence of abnormality. "
            "Classify as abnormal only when the feature pattern provides strong and internally consistent evidence of a real abnormal operating state. "
            "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
            "Return exactly one word: normal or abnormal. "
            "Do not return any explanation, punctuation, markdown, or extra text."
        )
        source_suffix = "zero-shot-dsp-conservative"
    elif variant == "normal_default":
        system_prompt = (
            "You are a machine sound anomaly classifier. "
            "The target machine is a belt conveyor. "
            "You are given only the absolute DSP feature values extracted from one audio clip. "
            "No normal reference or baseline statistics are provided. "
            "In this zero-shot setting, uncertainty should default to normal. "
            "Classify as abnormal only if multiple independent feature families jointly support a clear anomaly. "
            "Do not call a clip abnormal from energy level, dominant frequency, harmonicity, or spectral spread alone. "
            "If the evidence is mixed, incomplete, or plausibly explained by normal machine-to-machine variation, output normal. "
            "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
            "Return exactly one word: normal or abnormal. "
            "Do not return any explanation, punctuation, markdown, or extra text."
        )
        source_suffix = "zero-shot-dsp-normal-default"
    elif variant == "fault_plausibility":
        system_prompt = (
            "You are a machine sound anomaly classifier. "
            "The target machine is a belt conveyor. "
            "Possible abnormal causes in this dataset include: "
            "excessive tension at the tension pulley; "
            "excessive tension at the tail pulley; "
            "tail pulley removed; "
            "a metallic object attached to the belt; "
            "over voltage; "
            "under voltage. "
            "You are given only the absolute DSP feature values extracted from one audio clip. "
            "No normal reference or baseline statistics are provided. "
            "Judge abnormality only when the observed feature pattern is plausibly consistent with a coherent conveyor fault state, not merely because some values look uncommon. "
            "If the values do not form a believable fault pattern, output normal. "
            "Analyze the pattern and determine the most appropriate classification label based on the observed fluctuations. "
            "Return exactly one word: normal or abnormal. "
            "Do not return any explanation, punctuation, markdown, or extra text."
        )
        source_suffix = "zero-shot-dsp-fault-plausibility"
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported zero-shot DSP variant: {variant}")

    user_payload = _build_absolute_feature_payload(metadata, comparisons)
    content = _chat_completion_content(system_prompt, user_payload, config=config)
    prediction = _parse_label_response(content)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-{source_suffix}",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_zero_shot_cause_scores_b_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
    reordered: bool = False,
) -> ReasoningResult:
    del observations
    if reordered:
        cause_order = [
            "belt_attached_metallic_object",
            "over_voltage",
            "under_voltage",
            "tail_pulley_removed",
            "tail_pulley_excessive_tension",
            "tension_pulley_excessive_tension",
        ]
        source_suffix = "zero-shot-cause-scores-b-reordered"
    else:
        cause_order = [
            "tension_pulley_excessive_tension",
            "tail_pulley_excessive_tension",
            "tail_pulley_removed",
            "belt_attached_metallic_object",
            "over_voltage",
            "under_voltage",
        ]
        source_suffix = "zero-shot-cause-scores-b"

    system_prompt = (
        "You are evaluating possible anomaly causes for a belt conveyor sound. "
        "You are given only absolute DSP feature values extracted from a single audio clip. "
        "No normal reference clip, baseline statistics, or same-machine comparison is provided. "
        "In this zero-shot setting, different conveyor instances can vary in detailed structure, operating condition, and recording setup. "
        "Do not treat unusual absolute values alone as sufficient evidence of abnormality. "
        "Instead, assess how plausibly the observed feature pattern matches each possible fault type below. "
        "For each cause, assign exactly one level: "
        "low means weak, insufficient, or contradictory evidence; "
        "medium means some plausible supporting evidence but not strong enough to be confident; "
        "high means strong and coherent evidence that this cause is a good explanation. "
        f"Possible causes are exactly: {', '.join(cause_order)}. "
        "Return valid JSON with exactly these six keys and no additional text. "
        "Each value must be one of: low, medium, high."
    )
    user_payload = _build_absolute_feature_payload(metadata, comparisons)
    content = _chat_completion_content(system_prompt, user_payload, config=config)
    parsed = _parse_json_object(content)
    cause_levels = _parse_cause_levels(parsed)
    prediction = _rule_b_prediction_from_cause_levels(cause_levels)
    predicted_cause = _top_cause_from_levels(cause_levels, prediction, cause_order=cause_order)
    confidence = _confidence_from_cause_levels(cause_levels, prediction, fallback.confidence)

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-{source_suffix}",
        prediction=prediction,
        confidence=confidence,
        explanation="",
        predicted_cause=predicted_cause,
        evidence=[],
        possible_causes=[cause for cause, level in cause_levels.items() if level in {"medium", "high"}],
        suggested_checks=[],
        raw_response=content,
    )


def _openai_zero_shot_multiclass_reasoning(
    metadata: ClipMetadata,
    comparisons: dict[str, FeatureComparison],
    observations: list[Observation],
    fallback: ReasoningResult,
    *,
    config: OpenAIConfig,
) -> ReasoningResult:
    del observations
    labels = [
        "normal",
        "over_voltage",
        "under_voltage",
        "tail_pulley_removed",
        "belt_attached_metallic_object",
        "tail_pulley_excessive_tension",
        "tension_pulley_excessive_tension",
    ]
    system_prompt = (
        "You are a zero-shot machine sound classifier for a belt conveyor. "
        "You are given only absolute DSP feature values extracted from a single audio clip. "
        "No normal reference clip, baseline statistics, or same-machine comparison is provided. "
        "In this setting, different conveyor instances can vary in detailed structure, operating condition, and recording setup. "
        "Do not treat unusual absolute values alone as sufficient evidence of abnormality. "
        f"Choose exactly one label from the following list: {', '.join(labels)}. "
        "Use a fault label only if the observed feature pattern is more plausibly explained by that specific fault than by normal machine-to-machine variation. "
        "If the evidence is weak, mixed, generic, or does not clearly support one specific fault type, choose normal. "
        "Return exactly one label from the list above. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )
    user_payload = _build_absolute_feature_payload(metadata, comparisons)
    content = _chat_completion_content(system_prompt, user_payload, config=config)
    multiclass_label = _parse_multiclass_label_response(content, labels)
    prediction = "normal" if multiclass_label == "normal" else "abnormal"

    return ReasoningResult(
        source=f"{_source_from_base_url(config.base_url)}-zero-shot-multiclass",
        prediction=prediction,
        confidence=fallback.confidence,
        explanation="",
        predicted_cause="none" if multiclass_label == "normal" else multiclass_label,
        evidence=[],
        possible_causes=[],
        suggested_checks=[],
        raw_response=content,
    )


def _chat_completion_content(system_prompt: str, user_payload: dict, *, config: OpenAIConfig) -> str:
    if _source_from_base_url(config.base_url) == "ollama":
        return _ollama_chat_content(system_prompt, user_payload, config=config)

    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
    if reasoning_effort and config.model.startswith("gpt-5"):
        return _openai_responses_content(
            system_prompt,
            user_payload,
            config=config,
            reasoning_effort=reasoning_effort,
        )

    body = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
    }
    req = request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    return payload["choices"][0]["message"]["content"]


def _openai_responses_content(
    system_prompt: str,
    user_payload: dict,
    *,
    config: OpenAIConfig,
    reasoning_effort: str,
) -> str:
    body = {
        "model": config.model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        "reasoning": {
            "effort": reasoning_effort,
        },
        "text": {
            "verbosity": "low",
        },
    }
    req = request.Request(
        f"{config.base_url}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI responses request failed: {exc}") from exc

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = str(content.get("text", "")).strip()
                if text:
                    return text

    raise RuntimeError(f"OpenAI responses returned no text content: {payload!r}")


def _ollama_chat_content(system_prompt: str, user_payload: dict, *, config: OpenAIConfig) -> str:
    base_url = config.base_url
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    body = {
        "model": config.model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        "options": {
            "temperature": 0,
        },
    }
    req = request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    message = payload.get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError(f"Ollama returned empty content: {payload!r}")
    return content


def _gemini_completion_content(system_prompt: str, user_payload: dict, *, config: OpenAIConfig) -> str:
    generation_config: dict[str, object] = {
        "temperature": 0,
    }
    if config.model.startswith("gemini-3"):
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
        }

    body = {
        "systemInstruction": {
            "parts": [
                {
                    "text": system_prompt,
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(user_payload, indent=2),
                    }
                ],
            }
        ],
        "generationConfig": generation_config,
    }
    req = request.Request(
        f"{config.base_url}/models/{config.model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-goog-api-key": str(config.api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {payload!r}")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    content = "\n".join(text for text in texts if text).strip()
    if not content:
        raise RuntimeError(f"Gemini returned empty content: {payload!r}")
    return content


def _chat_completion_with_image_content(
    system_prompt: str,
    image_png: bytes,
    metadata: ClipMetadata,
    *,
    image_description: str,
    config: OpenAIConfig,
) -> str:
    return _chat_completion_with_images_content(
        system_prompt,
        {
            "clip": {
                "case": metadata.case,
                "mode": metadata.mode,
                "channel": metadata.channel,
            },
            "image_descriptions": [image_description],
        },
        images=[(image_description, image_png)],
        config=config,
    )


def _chat_completion_with_payload_and_images_content(
    system_prompt: str,
    user_payload: dict,
    *,
    images: list[tuple[str, bytes]],
    config: OpenAIConfig,
) -> str:
    payload = dict(user_payload)
    payload["image_descriptions"] = [description for description, _ in images]
    return _chat_completion_with_images_content(
        system_prompt,
        payload,
        images=images,
        config=config,
    )


def _chat_completion_with_images_content(
    system_prompt: str,
    user_payload: dict,
    *,
    images: list[tuple[str, bytes]],
    config: OpenAIConfig,
) -> str:
    content = [
        {
            "type": "text",
            "text": json.dumps(user_payload, indent=2),
        }
    ]
    for _, image_png in images:
        data_url = f"data:image/png;base64,{b64encode(image_png).decode('ascii')}"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "low",
                },
            }
        )
    body = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ],
    }
    req = request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI image request failed: {exc}") from exc

    return payload["choices"][0]["message"]["content"]


def _parse_label_response(content: str) -> str:
    cleaned = re.sub(r"`+", " ", content).strip().lower()
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]

    for token in tokens:
        normalized = _normalize_prediction(token)
        if normalized in {"normal", "abnormal"}:
            return normalized

    normalized = _normalize_prediction(cleaned)
    if normalized in {"normal", "abnormal"}:
        return normalized

    raise ValueError(f"Could not parse prediction label from model response: {content!r}")


def _parse_multiclass_label_response(content: str, labels: list[str]) -> str:
    cleaned = re.sub(r"`+", " ", content).strip().lower()
    cleaned = re.sub(r"[^a-z_\s]", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]

    joined = "_".join(tokens)
    for label in labels:
        if cleaned == label or joined == label:
            return label

    for token in tokens:
        if token in labels:
            return token

    for label in labels:
        if label in cleaned or label in joined:
            return label

    raise ValueError(f"Could not parse multiclass label from model response: {content!r}")


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])

    raise ValueError(f"Could not parse JSON object from model response: {content!r}")


def _source_from_base_url(base_url: str) -> str:
    if ":11434" in base_url:
        return "ollama"
    if "generativelanguage.googleapis.com" in base_url:
        return "gemini"
    return "openai"


def _normalize_prediction(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"abnormal", "anomalous", "anomaly"}:
        return "abnormal"
    if normalized in {"normal", "nominal"}:
        return "normal"
    return normalized


def _normalize_cause_level(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"low", "medium", "high"}:
        raise ValueError(f"Unsupported cause level: {value!r}")
    return normalized


def _parse_cause_levels(parsed: dict) -> dict[str, str]:
    return {
        "tension_pulley_excessive_tension": _normalize_cause_level(parsed["tension_pulley_excessive_tension"]),
        "tail_pulley_excessive_tension": _normalize_cause_level(parsed["tail_pulley_excessive_tension"]),
        "tail_pulley_removed": _normalize_cause_level(parsed["tail_pulley_removed"]),
        "belt_attached_metallic_object": _normalize_cause_level(parsed["belt_attached_metallic_object"]),
        "over_voltage": _normalize_cause_level(parsed["over_voltage"]),
        "under_voltage": _normalize_cause_level(parsed["under_voltage"]),
    }


def _rule_b_prediction_from_cause_levels(cause_levels: dict[str, str]) -> str:
    high_count = sum(1 for level in cause_levels.values() if level == "high")
    medium_count = sum(1 for level in cause_levels.values() if level == "medium")
    return "abnormal" if high_count >= 1 or medium_count >= 2 else "normal"


def _top_cause_from_levels(cause_levels: dict[str, str], prediction: str, *, cause_order: list[str] | None = None) -> str:
    if prediction != "abnormal":
        return "none"
    order = {"low": 0, "medium": 1, "high": 2}
    priority = {cause: index for index, cause in enumerate(cause_order or list(cause_levels.keys()))}
    ranked = sorted(
        cause_levels.items(),
        key=lambda item: (order[item[1]], -priority.get(item[0], 10_000)),
        reverse=True,
    )
    return ranked[0][0]


def _confidence_from_cause_levels(cause_levels: dict[str, str], prediction: str, fallback_confidence: str) -> str:
    high_count = sum(1 for level in cause_levels.values() if level == "high")
    medium_count = sum(1 for level in cause_levels.values() if level == "medium")
    if prediction == "abnormal":
        if high_count >= 2:
            return "high"
        if high_count >= 1 or medium_count >= 3:
            return "medium"
        return "low"
    if medium_count == 0 and high_count == 0:
        return "high"
    if medium_count <= 1:
        return "medium"
    return fallback_confidence


def _normalize_confidence(value: object) -> str:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 0.8:
            return "high"
        if numeric >= 0.5:
            return "medium"
        return "low"

    normalized = str(value).strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    if "high" in normalized:
        return "high"
    if "med" in normalized:
        return "medium"
    if "low" in normalized:
        return "low"
    return normalized


def _normalize_cause(value: object, prediction: str) -> str | None:
    if value is None:
        return "none" if prediction == "normal" else None

    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    aliases = {
        "none": "none",
        "no_cause": "none",
        "normal": "none",
        "tensionpulley_excessive_tension": "tension_pulley_excessive_tension",
        "tailpulley_excessive_tension": "tail_pulley_excessive_tension",
        "tailpulley_removed": "tail_pulley_removed",
        "belt_attached_metal_object": "belt_attached_metallic_object",
        "belt_attached_metallic_object": "belt_attached_metallic_object",
        "metallic_object_attached_to_belt": "belt_attached_metallic_object",
        "metal_object_attached_to_belt": "belt_attached_metallic_object",
        "overvoltage": "over_voltage",
        "over_voltage": "over_voltage",
        "undervoltage": "under_voltage",
        "under_voltage": "under_voltage",
    }
    normalized = aliases.get(normalized, normalized)

    if prediction == "normal":
        return "none"
    if normalized in CAUSE_CHOICES and normalized != "none":
        return normalized
    raise ValueError(f"Could not parse anomaly cause from model response: {value!r}")
