from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_DATA_ROOT
from .data import discover_clips, parse_clip_metadata

DEFAULT_BINARY_SPLIT_DIR = Path("reports/binary_splits")
DEFAULT_BINARY_TRAIN_MANIFEST = DEFAULT_BINARY_SPLIT_DIR / "ind_binary_train_manifest.json"
DEFAULT_BINARY_EVAL_MANIFEST = DEFAULT_BINARY_SPLIT_DIR / "ind_binary_evaluation_manifest.json"
DEFAULT_BINARY_TEST_MANIFEST = DEFAULT_BINARY_SPLIT_DIR / "ind_binary_test_manifest.json"
DEFAULT_BINARY_SPLIT_SUMMARY = DEFAULT_BINARY_SPLIT_DIR / "ind_binary_split_summary.md"


def build_fixed_binary_split_manifests(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    train_output_path: Path = DEFAULT_BINARY_TRAIN_MANIFEST,
    evaluation_output_path: Path = DEFAULT_BINARY_EVAL_MANIFEST,
    test_output_path: Path = DEFAULT_BINARY_TEST_MANIFEST,
    summary_output_path: Path = DEFAULT_BINARY_SPLIT_SUMMARY,
    train_normal_per_stratum: int = 100,
    evaluation_normal_per_stratum: int = 20,
    evaluation_abnormal_per_stratum: int = 20,
    test_normal_per_stratum: int = 20,
    test_abnormal_per_stratum: int = 20,
) -> dict:
    grouped = _group_ind_clips(data_root)
    train_manifest: list[dict] = []
    evaluation_manifest: list[dict] = []
    test_manifest: list[dict] = []

    for case in ("case1", "case2", "case3"):
        for channel in (1, 2, 3, 4):
            normal_paths = grouped[(case, channel, "normal")]
            abnormal_paths = grouped[(case, channel, "anomalous")]

            required_normal = train_normal_per_stratum + evaluation_normal_per_stratum + test_normal_per_stratum
            required_abnormal = evaluation_abnormal_per_stratum + test_abnormal_per_stratum

            if len(normal_paths) < required_normal:
                raise ValueError(f"Not enough normal IND clips for {case} ch{channel}: need {required_normal}")
            if len(abnormal_paths) < required_abnormal:
                raise ValueError(f"Not enough anomalous IND clips for {case} ch{channel}: need {required_abnormal}")

            test_normals, remaining_normals = _pop_evenly_spaced(normal_paths, test_normal_per_stratum)
            eval_normals, remaining_normals = _pop_evenly_spaced(remaining_normals, evaluation_normal_per_stratum)
            train_normals, _ = _pop_evenly_spaced(remaining_normals, train_normal_per_stratum)

            test_abnormals, remaining_abnormals = _pop_evenly_spaced(abnormal_paths, test_abnormal_per_stratum)
            eval_abnormals, _ = _pop_evenly_spaced(remaining_abnormals, evaluation_abnormal_per_stratum)

            train_manifest.extend(
                _manifest_row(path, case, channel, "normal", split="train")
                for path in train_normals
            )
            evaluation_manifest.extend(
                _manifest_row(path, case, channel, "normal", split="evaluation")
                for path in eval_normals
            )
            evaluation_manifest.extend(
                _manifest_row(path, case, channel, "anomalous", split="evaluation")
                for path in eval_abnormals
            )
            test_manifest.extend(
                _manifest_row(path, case, channel, "normal", split="test")
                for path in test_normals
            )
            test_manifest.extend(
                _manifest_row(path, case, channel, "anomalous", split="test")
                for path in test_abnormals
            )

    for output_path, manifest in (
        (train_output_path, train_manifest),
        (evaluation_output_path, evaluation_manifest),
        (test_output_path, test_manifest),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "train_manifest_path": str(train_output_path),
        "evaluation_manifest_path": str(evaluation_output_path),
        "test_manifest_path": str(test_output_path),
        "train_clip_count": len(train_manifest),
        "evaluation_clip_count": len(evaluation_manifest),
        "test_clip_count": len(test_manifest),
        "train_normal_per_stratum": train_normal_per_stratum,
        "evaluation_normal_per_stratum": evaluation_normal_per_stratum,
        "evaluation_abnormal_per_stratum": evaluation_abnormal_per_stratum,
        "test_normal_per_stratum": test_normal_per_stratum,
        "test_abnormal_per_stratum": test_abnormal_per_stratum,
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(render_binary_split_summary(summary), encoding="utf-8")
    summary["summary_path"] = str(summary_output_path)
    return summary


def render_binary_split_summary(summary: dict) -> str:
    lines = [
        "# IND Binary Split Summary",
        "",
        f"- Train manifest: `{summary['train_manifest_path']}`",
        f"- Evaluation manifest: `{summary['evaluation_manifest_path']}`",
        f"- Test manifest: `{summary['test_manifest_path']}`",
        f"- Train clips: {summary['train_clip_count']}",
        f"- Evaluation clips: {summary['evaluation_clip_count']}",
        f"- Test clips: {summary['test_clip_count']}",
        "",
        "## Per-Stratum Allocation",
        "",
        f"- Train normal per case-channel stratum: {summary['train_normal_per_stratum']}",
        f"- Evaluation normal per case-channel stratum: {summary['evaluation_normal_per_stratum']}",
        f"- Evaluation anomalous per case-channel stratum: {summary['evaluation_abnormal_per_stratum']}",
        f"- Test normal per case-channel stratum: {summary['test_normal_per_stratum']}",
        f"- Test anomalous per case-channel stratum: {summary['test_abnormal_per_stratum']}",
        "",
        "Train contains normal clips only for normal-only AE training.",
    ]
    return "\n".join(lines)


def _group_ind_clips(data_root: Path) -> dict[tuple[str, int, str], list[Path]]:
    grouped: dict[tuple[str, int, str], list[Path]] = {}
    for case in ("case1", "case2", "case3"):
        for channel in (1, 2, 3, 4):
            grouped[(case, channel, "normal")] = []
            grouped[(case, channel, "anomalous")] = []

    for path in discover_clips(data_root=data_root, mode="IND"):
        metadata = parse_clip_metadata(path)
        grouped[(metadata.case, metadata.channel, metadata.category)].append(path)

    return {key: sorted(value) for key, value in grouped.items()}


def _manifest_row(path: Path, case: str, channel: int, category: str, *, split: str) -> dict:
    return {
        "path": str(path),
        "split": split,
        "case": case,
        "mode": "IND",
        "channel": channel,
        "category": category,
        "expected_label": "abnormal" if category == "anomalous" else "normal",
    }


def _pop_evenly_spaced(paths: list[Path], count: int) -> tuple[list[Path], list[Path]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return [], list(paths)
    if len(paths) < count:
        raise ValueError(f"Cannot take {count} items from a list of length {len(paths)}")
    if count == len(paths):
        return list(paths), []
    if count == 1:
        selected_indices = [0]
    else:
        step = (len(paths) - 1) / float(count - 1)
        selected_indices = sorted({round(index * step) for index in range(count)})
        while len(selected_indices) < count:
            for candidate in range(len(paths)):
                if candidate not in selected_indices:
                    selected_indices.append(candidate)
                    selected_indices.sort()
                    if len(selected_indices) == count:
                        break

    selected_index_set = set(selected_indices[:count])
    selected = [path for index, path in enumerate(paths) if index in selected_index_set]
    remaining = [path for index, path in enumerate(paths) if index not in selected_index_set]
    return selected, remaining
