from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .audio import load_audio
from .config import DEFAULT_BASELINE_LIMIT, DEFAULT_DATA_ROOT, EPSILON, FEATURE_ORDER
from .data import parse_clip_metadata, select_baseline_paths
from .evaluation import DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_OUTPUT_DIR
from .features import extract_features

CLASSICAL_METHODS = (
    "diag_mahalanobis",
    "full_mahalanobis",
    "knn_distance",
    "pca_residual",
    "one_class_svm",
    "isolation_forest",
)


def run_classical_zero_shot_benchmark(
    *,
    manifest_path: Path = DEFAULT_EVALUATION_MANIFEST,
    data_root: Path = DEFAULT_DATA_ROOT,
    baseline_limit: int = DEFAULT_BASELINE_LIMIT,
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT_DIR,
    quantile: float = 0.95,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_cache: dict[str, np.ndarray] = {}
    results_by_method = {method: [] for method in CLASSICAL_METHODS}

    for item in manifest:
        clip_path = Path(item["path"])
        metadata = parse_clip_metadata(clip_path)
        target_vector = _load_feature_vector(clip_path, feature_cache)
        baseline_paths = select_baseline_paths(metadata, data_root=data_root, limit=baseline_limit)
        baseline_matrix = np.stack(
            [_load_feature_vector(path, feature_cache) for path in baseline_paths],
            axis=0,
        )

        for method in CLASSICAL_METHODS:
            started = time.perf_counter()
            score, threshold = _score_with_threshold(method, target_vector, baseline_matrix, quantile=quantile)
            latency_seconds = time.perf_counter() - started
            prediction = "abnormal" if score > threshold else "normal"
            results_by_method[method].append(
                {
                    **item,
                    "method": method,
                    "prediction": prediction,
                    "score": float(score),
                    "threshold": float(threshold),
                    "latency_seconds": latency_seconds,
                    "correct": prediction == item["expected_label"],
                }
            )

    summaries = {
        method: _build_method_summary(
            manifest=manifest,
            manifest_path=manifest_path,
            method=method,
            results=results_by_method[method],
            quantile=quantile,
            baseline_limit=baseline_limit,
        )
        for method in CLASSICAL_METHODS
    }
    artifact = {
        "manifest_path": str(manifest_path),
        "baseline_limit": baseline_limit,
        "threshold_quantile": quantile,
        "methods": summaries,
        "results": results_by_method,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"ind_classical_benchmark_{timestamp}.json"
    md_path = output_dir / f"ind_classical_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(render_classical_benchmark_markdown(artifact), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "report_path": str(md_path),
        "methods": summaries,
    }


def render_classical_benchmark_markdown(report: dict) -> str:
    lines = [
        "# Classical Zero-Shot Benchmark",
        "",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Baseline limit: {report['baseline_limit']}",
        f"- Threshold quantile: {report['threshold_quantile']:.2f}",
        "",
    ]
    ranked = sorted(report["methods"].items(), key=lambda item: item[1]["f1_abnormal"], reverse=True)
    for method, summary in ranked:
        lines.extend(
            [
                f"## {method}",
                "",
                f"- Accuracy: {summary['accuracy']:.3f}",
                f"- Precision (abnormal): {summary['precision_abnormal']:.3f}",
                f"- Recall (abnormal): {summary['recall_abnormal']:.3f}",
                f"- F1 (abnormal): {summary['f1_abnormal']:.3f}",
                f"- TP / TN / FP / FN: {summary['tp']} / {summary['tn']} / {summary['fp']} / {summary['fn']}",
                f"- Mean latency: {summary['mean_latency_seconds'] * 1000.0:.2f} ms",
                f"- P95 latency: {summary['p95_latency_seconds'] * 1000.0:.2f} ms",
                "",
            ]
        )
    return "\n".join(lines)


def _build_method_summary(
    *,
    manifest: list[dict],
    manifest_path: Path,
    method: str,
    results: list[dict],
    quantile: float,
    baseline_limit: int,
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
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 0.95) if latencies else 0.0
    by_case = Counter(
        item["case"] for item in results if item["expected_label"] == "normal" and item["prediction"] == "abnormal"
    )
    return {
        "manifest_path": str(manifest_path),
        "method": method,
        "evaluated_clips": len(manifest),
        "baseline_limit": baseline_limit,
        "threshold_quantile": quantile,
        "accuracy": accuracy,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_seconds": mean_latency,
        "p95_latency_seconds": p95_latency,
        "false_positive_cases": dict(by_case),
    }


def _score_with_threshold(
    method: str,
    target_vector: np.ndarray,
    baseline_matrix: np.ndarray,
    *,
    quantile: float,
) -> tuple[float, float]:
    target_score = _score_method(method, target_vector, baseline_matrix)
    loo_scores = []
    for index in range(len(baseline_matrix)):
        ref = np.delete(baseline_matrix, index, axis=0)
        loo_scores.append(_score_method(method, baseline_matrix[index], ref))
    threshold = float(np.quantile(np.array(loo_scores, dtype=np.float64), quantile))
    return float(target_score), threshold


def _score_method(method: str, vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    if method == "diag_mahalanobis":
        return _diag_mahalanobis_score(vector, reference_matrix)
    if method == "full_mahalanobis":
        return _full_mahalanobis_score(vector, reference_matrix)
    if method == "knn_distance":
        return _knn_distance_score(vector, reference_matrix)
    if method == "pca_residual":
        return _pca_residual_score(vector, reference_matrix)
    if method == "one_class_svm":
        return _one_class_svm_score(vector, reference_matrix)
    if method == "isolation_forest":
        return _isolation_forest_score(vector, reference_matrix)
    raise ValueError(f"Unsupported classical method: {method}")


def _diag_mahalanobis_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    mean = reference_matrix.mean(axis=0)
    std = np.maximum(reference_matrix.std(axis=0), EPSILON)
    z = (vector - mean) / std
    return float(np.sqrt(np.mean(z**2)))


def _full_mahalanobis_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    mean = reference_matrix.mean(axis=0)
    centered = reference_matrix - mean
    covariance = np.cov(centered, rowvar=False, bias=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]], dtype=np.float64)
    reg = max(float(np.trace(covariance)) / max(covariance.shape[0], 1), 1e-6) * 0.05
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * reg
    precision = np.linalg.pinv(covariance)
    diff = vector - mean
    return float(np.sqrt(max(diff @ precision @ diff, 0.0)))


def _knn_distance_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    mean = reference_matrix.mean(axis=0)
    std = np.maximum(reference_matrix.std(axis=0), EPSILON)
    normalized_reference = (reference_matrix - mean) / std
    normalized_vector = (vector - mean) / std
    distances = np.sqrt(np.sum((normalized_reference - normalized_vector) ** 2, axis=1))
    return float(np.min(distances))


def _pca_residual_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    normalized_reference, normalized_vector = _normalize_reference_and_vector(vector, reference_matrix)
    n_components = max(1, min(normalized_reference.shape[0] - 1, normalized_reference.shape[1] - 1, 5))
    if n_components <= 0:
        return float(np.linalg.norm(normalized_vector - normalized_reference.mean(axis=0)))
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(normalized_reference)
    reconstructed = model.inverse_transform(model.transform(normalized_vector[None, :]))[0]
    residual = normalized_vector - reconstructed
    return float(np.linalg.norm(residual))


def _one_class_svm_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    normalized_reference, normalized_vector = _normalize_reference_and_vector(vector, reference_matrix)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    model.fit(normalized_reference)
    return float(-model.decision_function(normalized_vector[None, :])[0])


def _isolation_forest_score(vector: np.ndarray, reference_matrix: np.ndarray) -> float:
    normalized_reference, normalized_vector = _normalize_reference_and_vector(vector, reference_matrix)
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=0,
    )
    model.fit(normalized_reference)
    return float(-model.score_samples(normalized_vector[None, :])[0])


def _load_feature_vector(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    if key in cache:
        return cache[key]
    sample_rate, samples = load_audio(path)
    feature_set = extract_features(samples, sample_rate)
    vector = np.array([feature_set.values[name] for name in FEATURE_ORDER], dtype=np.float64)
    cache[key] = vector
    return vector


def _normalize_reference_and_vector(
    vector: np.ndarray,
    reference_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = reference_matrix.mean(axis=0)
    std = np.maximum(reference_matrix.std(axis=0), EPSILON)
    normalized_reference = (reference_matrix - mean) / std
    normalized_vector = (vector - mean) / std
    return normalized_reference, normalized_vector


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = fraction * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
