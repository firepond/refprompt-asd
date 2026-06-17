from __future__ import annotations

import re
from pathlib import Path

from .config import DEFAULT_DATA_ROOT
from .models import ClipMetadata

FILENAME_RE = re.compile(
    r"^(?P<prefix>[^_]+)_"
    r"(?P<machine>Toy\w+)_"
    r"(?P<case>case\d+)_"
    r"(?P<label>normal|ab\d+)_"
    r"(?P<mode>[A-Z]+)_"
    r"ch(?P<channel>\d+)_"
    r"(?P<index>\d+)\.wav$"
)


def parse_clip_metadata(path: Path) -> ClipMetadata:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported ToyADMOS filename format: {path}")

    label = match.group("label")
    if label == "normal":
        category = "normal"
        anomaly_code = None
    else:
        category = "anomalous"
        anomaly_code = label

    return ClipMetadata(
        path=str(path),
        filename=path.name,
        machine_type=match.group("machine"),
        case=match.group("case"),
        category=category,
        mode=match.group("mode"),
        channel=int(match.group("channel")),
        anomaly_code=anomaly_code,
    )


def discover_clips(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    category: str | None = None,
    case: str | None = None,
    mode: str | None = None,
    channel: int | None = None,
) -> list[Path]:
    all_paths = sorted(path for path in data_root.rglob("*.wav") if path.is_file())
    selected: list[Path] = []

    for path in all_paths:
        try:
            metadata = parse_clip_metadata(path)
        except ValueError:
            continue

        if category and metadata.category != category:
            continue
        if case and metadata.case != case:
            continue
        if mode and metadata.mode != mode:
            continue
        if channel and metadata.channel != channel:
            continue
        selected.append(path)

    return selected


def sample_clip(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    kind: str,
    case: str = "case1",
    mode: str = "IND",
    channel: int = 1,
) -> Path:
    category = {"normal": "normal", "abnormal": "anomalous", "anomalous": "anomalous"}[kind]
    matches = discover_clips(
        data_root=data_root,
        category=category,
        case=case,
        mode=mode,
        channel=channel,
    )
    if not matches:
        raise FileNotFoundError(
            f"No sample clips found for category={category}, case={case}, mode={mode}, channel={channel}"
        )
    return matches[0]


def select_baseline_paths(
    metadata: ClipMetadata,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    limit: int | None = None,
    channel_strategy: str = "same_channel",
) -> list[Path]:
    if limit == 0:
        return []

    if not metadata.case or not metadata.mode:
        raise ValueError("Target clip metadata does not include case/mode needed for baseline selection.")

    baseline_dir = data_root / metadata.case / f"NormalSound_{metadata.mode}"
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    all_baselines = sorted(path for path in baseline_dir.glob("*.wav") if path.is_file())
    if not all_baselines:
        raise FileNotFoundError(f"No normal baseline clips found in {baseline_dir}")

    channel_matches = []
    if channel_strategy == "same_channel" and metadata.channel is not None:
        for path in all_baselines:
            baseline_meta = parse_clip_metadata(path)
            if baseline_meta.channel == metadata.channel:
                channel_matches.append(path)

    if channel_strategy == "balanced_all_channels":
        return _select_balanced_all_channels(all_baselines, limit)

    selected = channel_matches or all_baselines
    if limit and len(selected) > limit:
        selected = _take_evenly_spaced(selected, limit)
    return selected


def _take_evenly_spaced(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    step = (len(paths) - 1) / float(limit - 1)
    indices = sorted({round(i * step) for i in range(limit)})
    return [paths[index] for index in indices]


def _select_balanced_all_channels(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths

    by_channel: dict[int, list[Path]] = {}
    for path in paths:
        channel = parse_clip_metadata(path).channel
        if channel is None:
            continue
        by_channel.setdefault(channel, []).append(path)

    channels = sorted(by_channel)
    if not channels:
        return _take_evenly_spaced(paths, limit)

    base = limit // len(channels)
    remainder = limit % len(channels)

    if base == 0:
        selected_channels = channels[:limit]
        return [by_channel[channel][0] for channel in selected_channels]

    selected: list[Path] = []
    for index, channel in enumerate(channels):
        target_count = base + (1 if index < remainder else 0)
        channel_paths = by_channel[channel]
        if len(channel_paths) < target_count:
            raise ValueError(
                f"Not enough normal clips for balanced_all_channels selection: "
                f"channel={channel}, required={target_count}, found={len(channel_paths)}"
            )
        selected.extend(_take_evenly_spaced(channel_paths, target_count))
    return selected
