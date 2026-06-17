from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .audio import load_audio
from .config import DEFAULT_DATA_ROOT, get_openai_config
from .data import discover_clips, parse_clip_metadata
from .evaluation import (
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_COST_PER_200_CALLS,
    DEFAULT_EVALUATION_MANIFEST,
)
from .features import extract_features
from .reasoning import _chat_completion_content, _normalize_prediction, _parse_json_object


DEFAULT_REFERENCE_EXAMPLE_OUTPUT_DIR = Path("reports/reference_examples")


def run_reference_example_benchmark(
    *,
    manifest_path: Path = DEFAULT_EVALUATION_MANIFEST,
    data_root: Path = DEFAULT_DATA_ROOT,
    reference_count: int = 4,
    channel_strategy: str = "same_channel",
    output_dir: Path = DEFAULT_REFERENCE_EXAMPLE_OUTPUT_DIR,
    max_workers: int = DEFAULT_BENCHMARK_WORKERS,
    cost_per_200_calls: float = DEFAULT_COST_PER_200_CALLS,
) -> dict:
    if reference_count != 4:
        raise ValueError("This benchmark currently expects reference_count=4.")
    if channel_strategy not in {"same_channel", "all_channels"}:
        raise ValueError("channel_strategy must be same_channel or all_channels.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_cache: dict[str, dict] = {}

    indexed_results: list[tuple[int, dict]] = []
    worker_count = max(1, min(max_workers, len(manifest)))
    if worker_count == 1:
        for index, item in enumerate(manifest):
            indexed_results.append(
                (
                    index,
                    _evaluate_query(
                        item,
                        data_root=data_root,
                        feature_cache=feature_cache,
                        reference_count=reference_count,
                        channel_strategy=channel_strategy,
                    ),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _evaluate_query,
                    item,
                    data_root=data_root,
                    feature_cache=feature_cache,
                    reference_count=reference_count,
                    channel_strategy=channel_strategy,
                ): index
                for index, item in enumerate(manifest)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                indexed_results.append((index, future.result()))

    results = [payload for _, payload in sorted(indexed_results, key=lambda item: item[0])]
    summary = _build_summary(
        manifest_path=manifest_path,
        results=results,
        reference_count=reference_count,
        channel_strategy=channel_strategy,
        worker_count=worker_count,
        cost_per_200_calls=cost_per_200_calls,
    )
    artifact = {"summary": summary, "results": results}

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"reference_examples_{channel_strategy}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(artifact), encoding="utf-8")
    return {
        "summary": summary,
        "json_path": str(json_path),
        "report_path": str(md_path),
    }


def _evaluate_query(
    item: dict,
    *,
    data_root: Path,
    feature_cache: dict[str, dict],
    reference_count: int,
    channel_strategy: str,
) -> dict:
    query_path = Path(item["path"])
    metadata = parse_clip_metadata(query_path)
    reference_paths = _select_reference_paths(
        metadata,
        query_path=query_path,
        data_root=data_root,
        reference_count=reference_count,
        channel_strategy=channel_strategy,
    )
    system_prompt = _build_system_prompt()
    user_payload = {
        "setup": {
            "case": metadata.case,
            "query_channel": metadata.channel,
            "mode": metadata.mode,
            "reference_count": reference_count,
            "reference_channel_strategy": channel_strategy,
            "note": "References are all normal examples from the same machine case.",
        },
        "normal_references": [
            {
                "channel": _get_feature_entry(path, feature_cache)["metadata"].channel,
                "feature_values": _get_feature_entry(path, feature_cache)["feature_values"],
            }
            for path in reference_paths
        ],
        "query_example": {
            "channel": metadata.channel,
            "feature_values": _get_feature_entry(query_path, feature_cache)["feature_values"],
        },
    }

    started = time.perf_counter()
    content = _chat_completion_content(system_prompt, user_payload, config=get_openai_config())
    latency_seconds = time.perf_counter() - started
    parsed = _parse_json_object(content)
    prediction = _normalize_prediction(parsed["prediction"])
    abnormal_score = _normalize_abnormal_score(parsed.get("abnormal_score", 50))
    return {
        **item,
        "prediction": prediction,
        "abnormal_score": abnormal_score,
        "latency_seconds": latency_seconds,
        "correct": prediction == item["expected_label"],
        "reference_channel_strategy": channel_strategy,
        "reference_paths": [str(path) for path in reference_paths],
        "raw_response": content,
    }


def _select_reference_paths(
    metadata,
    *,
    query_path: Path,
    data_root: Path,
    reference_count: int,
    channel_strategy: str,
) -> list[Path]:
    normal_paths = discover_clips(
        data_root=data_root,
        category="normal",
        case=metadata.case,
        mode=metadata.mode,
    )
    normal_paths = [path for path in normal_paths if path != query_path]
    if channel_strategy == "same_channel":
        same_channel_paths = [path for path in normal_paths if parse_clip_metadata(path).channel == metadata.channel]
        if len(same_channel_paths) < reference_count:
            raise ValueError(f"Not enough same-channel references for {query_path}")
        return _take_evenly_spaced(same_channel_paths, reference_count)

    selected = []
    for channel in (1, 2, 3, 4):
        channel_paths = [path for path in normal_paths if parse_clip_metadata(path).channel == channel]
        if not channel_paths:
            raise ValueError(f"Missing normal references for {metadata.case} ch{channel}")
        selected.append(channel_paths[0])
    if len(selected) != reference_count:
        raise ValueError("all_channels strategy currently expects one reference per channel.")
    return selected


def _get_feature_entry(path: Path, cache: dict[str, dict]) -> dict:
    key = str(path)
    if key not in cache:
        sample_rate, samples = load_audio(path)
        feature_values = extract_features(samples, sample_rate).values
        metadata = parse_clip_metadata(path)
        cache[key] = {
            "metadata": metadata,
            "feature_values": {feature: round(value, 6) for feature, value in feature_values.items()},
        }
    return cache[key]


def _build_system_prompt() -> str:
    return (
        "You are a machine sound anomaly classifier for a belt conveyor. "
        "You are given a small set of normal reference examples from the same machine case, plus one query example. "
        "Each reference includes its channel, which is a recording direction or acquisition viewpoint. "
        "If references come from multiple channels, treat channel differences as normal viewpoint variation, not as anomalies by themselves. "
        "Compare the query against the normal references using the overall DSP pattern across multiple features. "
        "Do not rely on a single feature alone. "
        "Return valid JSON with exactly these keys: prediction, abnormal_score. "
        "prediction must be normal or abnormal. "
        "abnormal_score must be a number from 0 to 100, where 0 means definitely normal and 100 means definitely abnormal. "
        "If the evidence is mixed or weak, lean toward normal. "
        "Do not return markdown or extra text."
    )


def _normalize_abnormal_score(value: object) -> float:
    numeric = float(value)
    return min(max(numeric, 0.0), 100.0)


def _build_summary(
    *,
    manifest_path: Path,
    results: list[dict],
    reference_count: int,
    channel_strategy: str,
    worker_count: int,
    cost_per_200_calls: float,
) -> dict:
    tp = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "abnormal")
    tn = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "normal")
    fp = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "abnormal")
    fn = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "normal")
    evaluated = len(results)
    y_true = np.array([1 if item["expected_label"] == "abnormal" else 0 for item in results], dtype=np.int64)
    y_score = np.array([item["abnormal_score"] for item in results], dtype=np.float64)
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latencies = sorted(item["latency_seconds"] for item in results)
    p95_latency = _percentile(latencies, 0.95) if latencies else 0.0
    return {
        "manifest_path": str(manifest_path),
        "reference_count": reference_count,
        "channel_strategy": channel_strategy,
        "worker_count": worker_count,
        "evaluated_clips": evaluated,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": p95_latency,
        "estimated_api_cost_usd": (evaluated / 200.0) * cost_per_200_calls,
    }


def _render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Reference Example Benchmark",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Reference count: {summary['reference_count']}",
        f"- Channel strategy: `{summary['channel_strategy']}`",
        f"- Workers: {summary['worker_count']}",
        f"- Evaluated clips: {summary['evaluated_clips']}",
        f"- Accuracy: {summary['accuracy']:.3f}",
        f"- ROC-AUC: {summary['roc_auc']:.3f}",
        f"- Precision (abnormal): {summary['precision_abnormal']:.3f}",
        f"- Recall (abnormal): {summary['recall_abnormal']:.3f}",
        f"- F1 (abnormal): {summary['f1_abnormal']:.3f}",
        f"- TP / TN / FP / FN: {summary['tp']} / {summary['tn']} / {summary['fp']} / {summary['fn']}",
        f"- Mean latency: {summary['mean_latency_seconds']:.2f}s",
        f"- P95 latency: {summary['p95_latency_seconds']:.2f}s",
        f"- Estimated API cost: ${summary['estimated_api_cost_usd']:.2f}",
        "",
    ]
    return "\n".join(lines)


def _take_evenly_spaced(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    step = (len(paths) - 1) / float(limit - 1)
    indices = sorted({round(i * step) for i in range(limit)})
    return [paths[index] for index in indices]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
