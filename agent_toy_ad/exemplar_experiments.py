from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .audio import load_audio
from .config import DEFAULT_DATA_ROOT, get_openai_config
from .data import discover_clips, parse_clip_metadata
from .evaluation import DEFAULT_BENCHMARK_WORKERS, DEFAULT_COST_PER_200_CALLS
from .features import extract_features
from .reasoning import _chat_completion_content, _parse_label_response


DEFAULT_EXEMPLAR_OUTPUT_DIR = Path("reports/exemplar_experiments")


def run_ab02_single_anomaly_exemplar_benchmark(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    abnormal_support_count: int,
    output_dir: Path = DEFAULT_EXEMPLAR_OUTPUT_DIR,
    max_workers: int = DEFAULT_BENCHMARK_WORKERS,
    cost_per_200_calls: float = DEFAULT_COST_PER_200_CALLS,
) -> dict:
    if abnormal_support_count not in {0, 1, 2}:
        raise ValueError("abnormal_support_count must be one of 0, 1, or 2.")

    manifest = _build_ab02_single_anomaly_manifest(data_root=data_root)
    feature_cache = _build_feature_cache(manifest)
    system_prompt = _build_ab02_exemplar_system_prompt()

    indexed_results: list[tuple[int, dict]] = []
    worker_count = max(1, min(max_workers, len(manifest["queries"])))
    if worker_count == 1:
        for index, item in enumerate(manifest["queries"]):
            indexed_results.append(
                (
                    index,
                    _evaluate_ab02_query(
                        item,
                        support_manifest=manifest["support"],
                        feature_cache=feature_cache,
                        abnormal_support_count=abnormal_support_count,
                        system_prompt=system_prompt,
                    ),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _evaluate_ab02_query,
                    item,
                    support_manifest=manifest["support"],
                    feature_cache=feature_cache,
                    abnormal_support_count=abnormal_support_count,
                    system_prompt=system_prompt,
                ): index
                for index, item in enumerate(manifest["queries"])
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                indexed_results.append((index, future.result()))

    results = [payload for _, payload in sorted(indexed_results, key=lambda item: item[0])]
    summary = _summarize_results(
        results,
        abnormal_support_count=abnormal_support_count,
        worker_count=worker_count,
        cost_per_200_calls=cost_per_200_calls,
    )
    artifact = {
        "manifest": manifest,
        "summary": summary,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"ab02_exemplar_{abnormal_support_count}ab_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(artifact), encoding="utf-8")
    return {
        "summary": summary,
        "json_path": str(json_path),
        "report_path": str(md_path),
    }


def _build_ab02_single_anomaly_manifest(*, data_root: Path) -> dict:
    support = {}
    queries = []
    for case in ("case1", "case2", "case3"):
        for channel in (1, 2, 3, 4):
            normal_paths = discover_clips(
                data_root=data_root,
                category="normal",
                case=case,
                mode="IND",
                channel=channel,
            )
            abnormal_paths = [
                path
                for path in discover_clips(
                    data_root=data_root,
                    category="anomalous",
                    case=case,
                    mode="IND",
                    channel=channel,
                )
                if parse_clip_metadata(path).anomaly_code == "ab02"
            ]
            if len(normal_paths) < 9:
                raise ValueError(f"Need at least 9 normal IND clips for {case} ch{channel}")
            if len(abnormal_paths) < 5:
                raise ValueError(f"Need at least 5 ab02 IND clips for {case} ch{channel}")

            support_normals = _take_evenly_spaced(normal_paths, 4)
            remaining_normals = [path for path in normal_paths if path not in set(support_normals)]
            query_normals = _take_evenly_spaced(remaining_normals, 5)
            abnormal_support_pool = abnormal_paths[:2]
            query_abnormals = abnormal_paths[2:]

            stratum_key = f"{case}_ch{channel}"
            support[stratum_key] = {
                "case": case,
                "channel": channel,
                "normal_support_paths": [str(path) for path in support_normals],
                "ab02_support_paths": [str(path) for path in abnormal_support_pool],
            }

            for path in query_normals:
                queries.append(
                    {
                        "path": str(path),
                        "case": case,
                        "channel": channel,
                        "expected_label": "normal",
                        "stratum_key": stratum_key,
                    }
                )
            for path in query_abnormals:
                queries.append(
                    {
                        "path": str(path),
                        "case": case,
                        "channel": channel,
                        "expected_label": "abnormal",
                        "stratum_key": stratum_key,
                    }
                )

    return {
        "experiment": "ab02_single_anomaly_exemplars",
        "support": support,
        "queries": queries,
    }


def _build_feature_cache(manifest: dict) -> dict[str, dict]:
    all_paths = set()
    for support in manifest["support"].values():
        all_paths.update(support["normal_support_paths"])
        all_paths.update(support["ab02_support_paths"])
    for item in manifest["queries"]:
        all_paths.add(item["path"])

    cache = {}
    for path_str in sorted(all_paths):
        path = Path(path_str)
        sample_rate, samples = load_audio(path)
        features = extract_features(samples, sample_rate)
        metadata = parse_clip_metadata(path)
        cache[path_str] = {
            "metadata": metadata,
            "feature_values": {feature: round(value, 6) for feature, value in features.values.items()},
        }
    return cache


def _evaluate_ab02_query(
    item: dict,
    *,
    support_manifest: dict[str, dict],
    feature_cache: dict[str, dict],
    abnormal_support_count: int,
    system_prompt: str,
) -> dict:
    support = item["stratum_key"]
    normal_support_paths = support_manifest[support]["normal_support_paths"]
    ab_support_paths = support_manifest[support]["ab02_support_paths"][:abnormal_support_count]

    user_payload = {
        "setup": {
            "case": item["case"],
            "channel": item["channel"],
            "mode": "IND",
            "task": "binary classification with in-context support examples",
            "single_abnormal_pattern": "ab02",
        },
        "support_examples": {
            "normal": [
                {
                    "feature_values": feature_cache[path]["feature_values"],
                }
                for path in normal_support_paths
            ],
            "abnormal": [
                {
                    "feature_values": feature_cache[path]["feature_values"],
                }
                for path in ab_support_paths
            ],
        },
        "query_example": {
            "feature_values": feature_cache[item["path"]]["feature_values"],
        },
    }

    started = time.perf_counter()
    content = _chat_completion_content(system_prompt, user_payload, config=get_openai_config())
    prediction = _parse_label_response(content)
    latency_seconds = time.perf_counter() - started
    return {
        **item,
        "prediction": prediction,
        "correct": prediction == item["expected_label"],
        "latency_seconds": latency_seconds,
        "abnormal_support_count": abnormal_support_count,
        "raw_response": content,
    }


def _build_ab02_exemplar_system_prompt() -> str:
    return (
        "You are a machine sound anomaly classifier for a belt conveyor. "
        "All support examples and the query come from the same machine setting, meaning the same case and channel. "
        "The normal support examples are examples of normal operation. "
        "The abnormal support examples, if provided, are examples of one single abnormal pattern labeled ab02. "
        "Treat this as an in-context classification task: compare the query against the provided normal and abnormal examples. "
        "Use the DSP feature patterns across multiple features rather than any single feature alone. "
        "Do not assume any universal absolute threshold for anomaly, because machine sounds can vary. "
        "If the query is more consistent with the normal support examples, output normal. "
        "If the query is more consistent with the abnormal support examples, output abnormal. "
        "If the evidence is mixed or insufficient, prefer normal. "
        "Return exactly one word: normal or abnormal. "
        "Do not return any explanation, punctuation, markdown, or extra text."
    )


def _summarize_results(
    results: list[dict],
    *,
    abnormal_support_count: int,
    worker_count: int,
    cost_per_200_calls: float,
) -> dict:
    tp = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "abnormal")
    tn = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "normal")
    fp = sum(1 for item in results if item["expected_label"] == "normal" and item["prediction"] == "abnormal")
    fn = sum(1 for item in results if item["expected_label"] == "abnormal" and item["prediction"] == "normal")
    evaluated = len(results)
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latencies = sorted(item["latency_seconds"] for item in results)
    p95_index = min(max(int(len(latencies) * 0.95) - 1, 0), max(len(latencies) - 1, 0))
    p95 = latencies[p95_index] if latencies else 0.0
    return {
        "abnormal_support_count": abnormal_support_count,
        "evaluated_clips": evaluated,
        "worker_count": worker_count,
        "accuracy": accuracy,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_seconds": p95,
        "estimated_api_cost_usd": (evaluated / 200.0) * cost_per_200_calls,
    }


def _render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# ab02 Single-Anomaly Exemplar Benchmark",
        "",
        f"- Abnormal support count: {summary['abnormal_support_count']}",
        f"- Evaluated clips: {summary['evaluated_clips']}",
        f"- Workers: {summary['worker_count']}",
        f"- Accuracy: {summary['accuracy']:.3f}",
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
