from __future__ import annotations

from pathlib import Path

from .audio import load_audio
from .baseline import build_absolute_feature_comparisons, build_baseline_profile, compare_to_baseline
from .config import DEFAULT_BASELINE_LIMIT, DEFAULT_DATA_ROOT
from .data import parse_clip_metadata, select_baseline_paths
from .features import extract_features
from .models import AnalysisReport
from .observations import build_observations
from .reasoning import reason_about_clip
from .smoothing import smooth_waveform_savgol
from .spectrogram_image import render_log_mel_spectrogram_png
from .waveform_image import render_raw_waveform_png, render_smoothed_waveform_png

BASELINE_FREE_REASONER_MODES = {
    "openai_baseline_free_dsp",
    "openai_zero_shot_dsp_conservative",
    "openai_zero_shot_dsp_normal_default",
    "openai_zero_shot_dsp_fault_plausibility",
    "openai_zero_shot_cause_scores_b",
    "openai_zero_shot_cause_scores_b_reordered",
    "openai_zero_shot_multiclass",
}


def analyze_clip(
    clip_path: Path,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    baseline_limit: int = DEFAULT_BASELINE_LIMIT,
    reasoner_mode: str = "auto",
    baseline_channel_strategy: str = "same_channel",
    exclude_features: set[str] | None = None,
) -> AnalysisReport:
    metadata = parse_clip_metadata(clip_path)

    if metadata.mode == "IND":
        return _analyze_ind_clip(
            clip_path=clip_path,
            data_root=data_root,
            baseline_limit=baseline_limit,
            reasoner_mode=reasoner_mode,
            baseline_channel_strategy=baseline_channel_strategy,
            exclude_features=exclude_features,
        )
    if metadata.mode == "CNT":
        return _analyze_cnt_clip_placeholder(metadata)

    raise ValueError(f"Unsupported ToyADMOS mode: {metadata.mode}")


def _analyze_ind_clip(
    *,
    clip_path: Path,
    data_root: Path,
    baseline_limit: int,
    reasoner_mode: str,
    baseline_channel_strategy: str,
    exclude_features: set[str] | None,
) -> AnalysisReport:
    metadata = parse_clip_metadata(clip_path)
    sample_rate, samples = load_audio(clip_path)
    target_features = extract_features(samples, sample_rate)

    if reasoner_mode in BASELINE_FREE_REASONER_MODES:
        baseline_paths: list[Path] = []
        baseline_profile = build_baseline_profile([], clip_paths=[])
        comparisons = build_absolute_feature_comparisons(target_features)
        observations = []
        baseline_feature_sets = []
    else:
        baseline_paths = select_baseline_paths(
            metadata,
            data_root=data_root,
            limit=baseline_limit,
            channel_strategy=baseline_channel_strategy,
        )
        if not baseline_paths:
            raise ValueError("A baseline-aware analysis path requires at least one baseline clip.")

        baseline_feature_sets = []
        for baseline_path in baseline_paths:
            baseline_sample_rate, baseline_samples = load_audio(baseline_path)
            baseline_feature_sets.append(extract_features(baseline_samples, baseline_sample_rate))

        baseline_profile = build_baseline_profile(
            baseline_feature_sets,
            clip_paths=[str(path) for path in baseline_paths],
        )
        comparisons = compare_to_baseline(target_features, baseline_profile)
        observations = build_observations(comparisons)
    image_png = None
    secondary_image_png = None
    if reasoner_mode in {"openai_image", "openai_image_describe_label"}:
        image_png = render_log_mel_spectrogram_png(samples, sample_rate)
    elif reasoner_mode == "openai_dsp_image":
        smoothed_samples = smooth_waveform_savgol(samples)
        image_png = render_log_mel_spectrogram_png(smoothed_samples, sample_rate)
    elif reasoner_mode == "openai_raw_waveform_image":
        image_png = render_raw_waveform_png(samples)
    elif reasoner_mode == "openai_waveform_image":
        image_png = render_smoothed_waveform_png(samples)
    elif reasoner_mode == "openai_dual_image":
        smoothed_samples = smooth_waveform_savgol(samples)
        image_png = render_smoothed_waveform_png(smoothed_samples)
        secondary_image_png = render_log_mel_spectrogram_png(smoothed_samples, sample_rate)
    reasoning = reason_about_clip(
        metadata,
        comparisons,
        observations,
        mode=reasoner_mode,
        exclude_features=exclude_features,
        baseline_feature_sets=baseline_feature_sets,
        image_png=image_png,
        secondary_image_png=secondary_image_png,
    )

    notes = []
    if reasoner_mode == "auto" and reasoning.source == "rule-based":
        notes.append("LLM reasoning was not used; rule-based reasoning handled the final decision.")

    return AnalysisReport(
        metadata=metadata,
        baseline=baseline_profile,
        target_features=target_features,
        comparisons=comparisons,
        observations=observations,
        reasoning=reasoning,
        mode_handler="ind_reference_baseline",
        mode_details={
            "mode": "IND",
            "baseline_limit": baseline_limit,
            "baseline_channel_strategy": baseline_channel_strategy,
            "exclude_features": sorted(exclude_features or []),
        },
        notes=notes,
    )


def _analyze_cnt_clip_placeholder(metadata) -> AnalysisReport:
    raise NotImplementedError(
        "CNT mode is reserved as a future extension point. "
        "The current acoustic anomaly detector only supports IND clips."
    )
