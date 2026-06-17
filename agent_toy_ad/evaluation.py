from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import DEFAULT_BASELINE_LIMIT, DEFAULT_DATA_ROOT
from .data import discover_clips, parse_clip_metadata
from .pipeline import analyze_clip
from .reasoning import build_label_only_prompt_payload, compute_rule_anomaly_metrics

DEFAULT_EVALUATION_OUTPUT_DIR = Path("reports")
DEFAULT_EVALUATION_MANIFEST = DEFAULT_EVALUATION_OUTPUT_DIR / "ind_evaluation_manifest.json"
DEFAULT_FALSE_POSITIVE_TRACE = DEFAULT_EVALUATION_OUTPUT_DIR / "ind_false_positive_traces.md"
DEFAULT_COST_PER_200_CALLS = 0.6
DEFAULT_BENCHMARK_WORKERS = 4


def build_fixed_ind_evaluation_manifest(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_path: Path = DEFAULT_EVALUATION_MANIFEST,
    normal_per_stratum: int = 5,
    abnormal_per_stratum: int = 5,
    sampling: str = "even",
    seed: int = 0,
) -> dict:
    grouped = _group_ind_clips(data_root)
    manifest = []
    rng = random.Random(seed)

    for case in ("case1", "case2", "case3"):
        for channel in (1, 2, 3, 4):
            normal_paths = grouped.get((case, channel, "normal"), [])
            abnormal_paths = grouped.get((case, channel, "anomalous"), [])
            if len(normal_paths) < normal_per_stratum:
                raise ValueError(f"Not enough normal IND clips for {case} ch{channel}")
            if len(abnormal_paths) < abnormal_per_stratum:
                raise ValueError(f"Not enough anomalous IND clips for {case} ch{channel}")

            normal_selected = _sample_paths(
                normal_paths,
                normal_per_stratum,
                sampling=sampling,
                rng=rng,
            )
            abnormal_selected = _sample_paths(
                abnormal_paths,
                abnormal_per_stratum,
                sampling=sampling,
                rng=rng,
            )

            for path in normal_selected:
                manifest.append(_manifest_row(path, case, channel, "normal"))
            for path in abnormal_selected:
                manifest.append(_manifest_row(path, case, channel, "anomalous"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(output_path),
        "clip_count": len(manifest),
        "normal_per_stratum": normal_per_stratum,
        "abnormal_per_stratum": abnormal_per_stratum,
        "sampling": sampling,
        "seed": seed,
    }


def run_fixed_ind_benchmark(
    *,
    manifest_path: Path = DEFAULT_EVALUATION_MANIFEST,
    reasoner_mode: str = "openai",
    data_root: Path = DEFAULT_DATA_ROOT,
    baseline_limit: int = DEFAULT_BASELINE_LIMIT,
    cost_per_200_calls: float = DEFAULT_COST_PER_200_CALLS,
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT_DIR,
    max_workers: int = DEFAULT_BENCHMARK_WORKERS,
    baseline_channel_strategy: str = "same_channel",
    exclude_features: list[str] | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed_results: list[tuple[int, dict]] = []
    indexed_errors: list[tuple[int, dict]] = []

    worker_count = max(1, min(max_workers, len(manifest)))
    if worker_count == 1:
        for index, item in enumerate(manifest):
            kind, payload = _evaluate_manifest_item(
                item,
                data_root=data_root,
                baseline_limit=baseline_limit,
                reasoner_mode=reasoner_mode,
                baseline_channel_strategy=baseline_channel_strategy,
                exclude_features=exclude_features,
            )
            if kind == "result":
                indexed_results.append((index, payload))
            else:
                indexed_errors.append((index, payload))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _evaluate_manifest_item,
                    item,
                    data_root=data_root,
                    baseline_limit=baseline_limit,
                    reasoner_mode=reasoner_mode,
                    baseline_channel_strategy=baseline_channel_strategy,
                    exclude_features=exclude_features,
                ): index
                for index, item in enumerate(manifest)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                kind, payload = future.result()
                if kind == "result":
                    indexed_results.append((index, payload))
                else:
                    indexed_errors.append((index, payload))

    results = [payload for _, payload in sorted(indexed_results, key=lambda item: item[0])]
    errors = [payload for _, payload in sorted(indexed_errors, key=lambda item: item[0])]

    summary = _build_benchmark_summary(
        manifest=manifest,
        manifest_path=manifest_path,
        results=results,
        errors=errors,
        reasoner_mode=reasoner_mode,
        cost_per_200_calls=cost_per_200_calls,
        worker_count=worker_count,
        baseline_limit=baseline_limit,
        baseline_channel_strategy=baseline_channel_strategy,
        exclude_features=exclude_features or [],
    )
    artifact = {"summary": summary, "results": results, "errors": errors}

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"ind_benchmark_{reasoner_mode}_{timestamp}.json"
    md_path = output_dir / f"ind_benchmark_{reasoner_mode}_{timestamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(render_benchmark_markdown(artifact), encoding="utf-8")

    return {
        "summary": summary,
        "json_path": str(json_path),
        "report_path": str(md_path),
    }


def write_false_positive_trace_report(
    *,
    benchmark_json_path: Path,
    output_path: Path = DEFAULT_FALSE_POSITIVE_TRACE,
    data_root: Path = DEFAULT_DATA_ROOT,
    baseline_limit: int = DEFAULT_BASELINE_LIMIT,
) -> dict:
    benchmark = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
    false_positives = [
        item
        for item in benchmark["results"]
        if item["expected_label"] == "normal" and item["prediction"] == "abnormal"
    ]

    traces = []
    feature_counter: Counter[str] = Counter()
    feature_z_sums: dict[str, float] = defaultdict(float)

    for item in false_positives:
        clip_path = Path(item["path"])
        report = analyze_clip(
            clip_path=clip_path,
            data_root=data_root,
            baseline_limit=baseline_limit,
            reasoner_mode="openai",
        )
        system_prompt, user_payload = build_label_only_prompt_payload(
            report.metadata,
            report.comparisons,
            report.observations,
        )
        top_features = sorted(
            report.comparisons.values(),
            key=lambda comparison: comparison.salience,
            reverse=True,
        )[:5]
        for feature in top_features:
            feature_counter[feature.feature] += 1
            feature_z_sums[feature.feature] += abs(feature.z_score)

        traces.append(
            {
                "path": item["path"],
                "case": item["case"],
                "channel": item["channel"],
                "prediction": item["prediction"],
                "confidence": item["confidence"],
                "latency_seconds": item["latency_seconds"],
                "anomaly_score": item["anomaly_score"],
                "baseline_paths": report.baseline.clip_paths,
                "observations": [observation.text for observation in report.observations],
                "top_features": [
                    {
                        "feature": feature.feature,
                        "value": round(feature.value, 6),
                        "baseline_mean": round(feature.baseline_mean, 6),
                        "z_score": round(feature.z_score, 6),
                        "direction": feature.direction,
                        "severity": feature.severity,
                    }
                    for feature in top_features
                ],
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "raw_response": report.reasoning.raw_response,
            }
        )

    aggregate_features = [
        {
            "feature": feature,
            "count": count,
            "mean_abs_z_score": round(feature_z_sums[feature] / count, 3),
        }
        for feature, count in feature_counter.most_common()
    ]
    by_case = Counter(trace["case"] for trace in traces)
    by_channel = Counter(trace["channel"] for trace in traces)

    payload = {
        "benchmark_json_path": str(benchmark_json_path),
        "false_positive_count": len(false_positives),
        "aggregate_features": aggregate_features,
        "by_case": dict(by_case),
        "by_channel": dict(sorted(by_channel.items())),
        "traces": traces,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_false_positive_markdown(payload), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "trace_report_path": str(output_path),
        "trace_json_path": str(json_path),
        "false_positive_count": len(false_positives),
    }


def render_benchmark_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# IND-Only Benchmark",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Reasoner: `{summary['reasoner_mode']}`",
        f"- Baseline limit: {summary['baseline_limit']}",
        f"- Baseline channel strategy: `{summary['baseline_channel_strategy']}`",
        f"- Exclude features: `{', '.join(summary['exclude_features']) if summary['exclude_features'] else '(none)'}`",
        f"- Workers: {summary['worker_count']}",
        f"- Evaluated clips: {summary['evaluated_clips']}",
        f"- Errors: {summary['error_count']}",
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
        "## Cause Distribution",
        "",
    ]
    cause_counts = summary.get("predicted_cause_counts", {})
    if not cause_counts:
        lines.append("- Not available for this reasoner")
    else:
        for cause, count in cause_counts.items():
            lines.append(f"- `{cause}`: {count}")

    lines.extend(
        [
            "",
        "## False Positives",
        "",
        ]
    )
    false_positives = [
        item for item in report["results"] if item["expected_label"] == "normal" and item["prediction"] == "abnormal"
    ]
    if not false_positives:
        lines.append("- None")
    else:
        for item in false_positives:
            lines.append(
                f"- `{Path(item['path']).name}` | {item['case']} | ch{item['channel']} | "
                f"confidence={item['confidence']} | anomaly_score={item['anomaly_score']:.3f}"
            )
    return "\n".join(lines)


def render_false_positive_markdown(report: dict) -> str:
    lines = [
        "# IND False Positive Trace Report",
        "",
        f"- Benchmark source: `{report['benchmark_json_path']}`",
        f"- False positive count: {report['false_positive_count']}",
        "",
        "## Aggregate Feature Pattern",
        "",
    ]
    if not report["aggregate_features"]:
        lines.append("- No false positives were found.")
        return "\n".join(lines)

    for item in report["aggregate_features"]:
        lines.append(
            f"- `{item['feature']}` appeared in {item['count']} false positives "
            f"with mean |z|={item['mean_abs_z_score']:.3f}"
        )

    lines.extend(["", "## Distribution", ""])
    lines.append("By case:")
    for case, count in report["by_case"].items():
        lines.append(f"- `{case}`: {count}")
    lines.append("")
    lines.append("By channel:")
    for channel, count in report["by_channel"].items():
        lines.append(f"- `ch{channel}`: {count}")

    lines.extend(["", "## Per-Clip Traces", ""])
    for trace in report["traces"]:
        lines.extend(
            [
                f"### `{Path(trace['path']).name}`",
                "",
                f"- Case / channel: `{trace['case']}` / `ch{trace['channel']}`",
                f"- Prediction: `{trace['prediction']}`",
                f"- Confidence: `{trace['confidence']}`",
                f"- Latency: `{trace['latency_seconds']:.2f}s`",
                f"- Rule anomaly score: `{trace['anomaly_score']:.3f}`",
                "",
                "Top feature deviations:",
            ]
        )
        for feature in trace["top_features"]:
            lines.append(
                f"- `{feature['feature']}`: z={feature['z_score']:.3f}, "
                f"{feature['direction']}, severity={feature['severity']}"
            )
        lines.extend(["", "Observations:"])
        for text in trace["observations"]:
            lines.append(f"- {text}")
        lines.extend(
            [
                "",
                "LLM raw response:",
                "",
                f"```text\n{trace['raw_response']}\n```",
                "",
            ]
        )
    return "\n".join(lines)


def _build_benchmark_summary(
    *,
    manifest: list[dict],
    manifest_path: Path,
    results: list[dict],
    errors: list[dict],
    reasoner_mode: str,
    cost_per_200_calls: float,
    worker_count: int,
    baseline_limit: int,
    baseline_channel_strategy: str,
    exclude_features: list[str],
) -> dict:
    tp = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "abnormal")
    tn = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "normal")
    fp = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "abnormal")
    fn = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "normal")
    cause_counts = Counter(item["predicted_cause"] for item in results if item.get("predicted_cause"))

    evaluated = len(results)
    roc_auc = 0.0
    if results:
        y_true = np.array([1 if item["expected_label"] == "abnormal" else 0 for item in results], dtype=np.int64)
        y_score = np.array([item["anomaly_score"] for item in results], dtype=np.float64)
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_score))
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latencies = sorted(item["latency_seconds"] for item in results)
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 0.95) if latencies else 0.0

    return {
        "manifest_path": str(manifest_path),
        "reasoner_mode": reasoner_mode,
        "baseline_limit": baseline_limit,
        "baseline_channel_strategy": baseline_channel_strategy,
        "exclude_features": exclude_features,
        "worker_count": worker_count,
        "evaluated_clips": evaluated,
        "error_count": len(errors),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_seconds": mean_latency,
        "p95_latency_seconds": p95_latency,
        "estimated_api_cost_usd": estimate_api_cost(len(manifest), cost_per_200_calls),
        "predicted_cause_counts": dict(cause_counts),
    }


def estimate_api_cost(call_count: int, cost_per_200_calls: float = DEFAULT_COST_PER_200_CALLS) -> float:
    return (call_count / 200.0) * cost_per_200_calls


def _group_ind_clips(data_root: Path) -> dict[tuple[str, int, str], list[Path]]:
    grouped: dict[tuple[str, int, str], list[Path]] = defaultdict(list)
    for path in discover_clips(data_root=data_root, mode="IND"):
        metadata = parse_clip_metadata(path)
        grouped[(metadata.case, metadata.channel, metadata.category)].append(path)
    return {key: sorted(value) for key, value in grouped.items()}


def _manifest_row(path: Path, case: str, channel: int, category: str) -> dict:
    return {
        "path": str(path),
        "case": case,
        "mode": "IND",
        "channel": channel,
        "category": category,
        "expected_label": "abnormal" if category == "anomalous" else "normal",
    }


def _normalize_prediction(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"abnormal", "anomalous"}:
        return "abnormal"
    return "normal" if normalized == "normal" else normalized


def _evaluate_manifest_item(
    item: dict,
    *,
    data_root: Path,
    baseline_limit: int,
    reasoner_mode: str,
    baseline_channel_strategy: str,
    exclude_features: list[str] | None,
) -> tuple[str, dict]:
    clip_path = Path(item["path"])
    started = time.perf_counter()
    try:
        report = analyze_clip(
            clip_path=clip_path,
            data_root=data_root,
            baseline_limit=baseline_limit,
            reasoner_mode=reasoner_mode,
            baseline_channel_strategy=baseline_channel_strategy,
            exclude_features=set(exclude_features or []),
        )
        latency_seconds = time.perf_counter() - started
        prediction = _normalize_prediction(report.reasoning.prediction)
        anomaly_metrics = compute_rule_anomaly_metrics(report.comparisons)
        return (
            "result",
            {
                **item,
                "prediction": prediction,
                "confidence": report.reasoning.confidence,
                "predicted_cause": report.reasoning.predicted_cause,
                "reasoning_source": report.reasoning.source,
                "latency_seconds": latency_seconds,
                "correct": prediction == item["expected_label"],
                "anomaly_score": anomaly_metrics["anomaly_score"],
                "very_strong_count": int(anomaly_metrics["very_strong_count"]),
            },
        )
    except Exception as exc:  # pragma: no cover - runtime path
        latency_seconds = time.perf_counter() - started
        return (
            "error",
            {**item, "latency_seconds": latency_seconds, "error": str(exc)},
        )


def _take_evenly_spaced(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    step = (len(paths) - 1) / float(limit - 1)
    indices = sorted({round(index * step) for index in range(limit)})
    return [paths[index] for index in indices]


def _sample_paths(
    paths: list[Path],
    limit: int,
    *,
    sampling: str,
    rng: random.Random,
) -> list[Path]:
    if sampling == "even":
        return _take_evenly_spaced(paths, limit)
    if sampling == "random":
        if limit <= 0 or len(paths) <= limit:
            return paths
        selected = rng.sample(paths, limit)
        return sorted(selected)
    raise ValueError(f"Unsupported sampling mode: {sampling}")


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
