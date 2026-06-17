from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score

from .audio import load_audio
from .config import DEFAULT_DATA_ROOT
from .supervised_splits import (
    DEFAULT_BINARY_EVAL_MANIFEST,
    DEFAULT_BINARY_TEST_MANIFEST,
    DEFAULT_BINARY_TRAIN_MANIFEST,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


DEFAULT_AE_OUTPUT_DIR = Path("reports/ae_baseline")


@dataclass(frozen=True)
class AEBaselineConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 64
    num_bw: int = 10
    num_fw: int = 10
    hidden_dim: int = 512
    latent_dim: int = 128
    num_hidden_layers: int = 4
    train_frames_per_clip: int = 32
    eval_max_frames_per_clip: int = 0
    batch_size: int = 256
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    input_normalization: str = "batchnorm"
    clip_score_mode: str = "max"
    clip_top_k_fraction: float = 0.05


def run_normal_only_ae_baseline(
    *,
    train_manifest_path: Path = DEFAULT_BINARY_TRAIN_MANIFEST,
    evaluation_manifest_path: Path = DEFAULT_BINARY_EVAL_MANIFEST,
    test_manifest_path: Path = DEFAULT_BINARY_TEST_MANIFEST,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_AE_OUTPUT_DIR,
    config: AEBaselineConfig = AEBaselineConfig(),
) -> dict:
    _require_torch()
    del data_root
    _set_random_seed(config.seed)

    device = _default_device()
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))

    mel_filterbank = _mel_filter_bank(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        num_channels=config.n_mels,
    )

    train_matrix = _build_training_matrix(train_manifest, mel_filterbank=mel_filterbank, config=config)
    standardizer = _fit_standardizer(train_matrix, config=config)
    train_matrix = _apply_standardizer(train_matrix, standardizer)
    input_dim = train_matrix.shape[1]
    model = _FrameAutoencoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_hidden_layers=config.num_hidden_layers,
        use_batchnorm=config.input_normalization == "batchnorm",
    ).to(device)

    train_started = time.perf_counter()
    training_history = _train_autoencoder(model, train_matrix, device=device, config=config)
    training_seconds = time.perf_counter() - train_started

    evaluation_results = _score_manifest(
        evaluation_manifest,
        model=model,
        mel_filterbank=mel_filterbank,
        device=device,
        config=config,
        standardizer=standardizer,
    )
    threshold = _best_f1_threshold(evaluation_results)
    evaluation_summary = _summarize_scored_results(evaluation_results, threshold)

    test_results = _score_manifest(
        test_manifest,
        model=model,
        mel_filterbank=mel_filterbank,
        device=device,
        config=config,
        standardizer=standardizer,
    )
    test_summary = _summarize_scored_results(test_results, threshold)

    artifact = {
        "config": {
            "train_manifest_path": str(train_manifest_path),
            "evaluation_manifest_path": str(evaluation_manifest_path),
            "test_manifest_path": str(test_manifest_path),
            "device": device.type,
            **config.__dict__,
        },
        "training": {
            "num_train_clips": len(train_manifest),
            "num_train_frames": int(train_matrix.shape[0]),
            "input_dim": int(input_dim),
            "seconds": training_seconds,
            "history": training_history,
            "standardizer": {
                "enabled": bool(standardizer is not None),
                "mode": config.input_normalization,
            },
        },
        "threshold": threshold,
        "evaluation_summary": evaluation_summary,
        "test_summary": test_summary,
        "evaluation_results": evaluation_results,
        "test_results": test_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"normal_only_ae_{timestamp}.json"
    md_path = output_dir / f"normal_only_ae_{timestamp}.md"
    model_path = output_dir / f"normal_only_ae_{timestamp}.pt"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(render_ae_baseline_markdown(artifact), encoding="utf-8")
    torch.save(model.state_dict(), model_path)
    return {
        "json_path": str(json_path),
        "report_path": str(md_path),
        "model_path": str(model_path),
        "evaluation_summary": evaluation_summary,
        "test_summary": test_summary,
    }


def render_ae_baseline_markdown(report: dict) -> str:
    cfg = report["config"]
    training = report["training"]
    eval_summary = report["evaluation_summary"]
    test_summary = report["test_summary"]
    lines = [
        "# Normal-Only AE Baseline",
        "",
        "## Config",
        "",
        f"- Train manifest: `{cfg['train_manifest_path']}`",
        f"- Evaluation manifest: `{cfg['evaluation_manifest_path']}`",
        f"- Test manifest: `{cfg['test_manifest_path']}`",
        f"- Device: `{cfg['device']}`",
        f"- Mel bins: {cfg['n_mels']}",
        f"- Context: bw={cfg['num_bw']}, fw={cfg['num_fw']}",
        f"- Train frames per clip: {cfg['train_frames_per_clip']}",
        f"- Max epochs: {cfg['max_epochs']}",
        "",
        "## Training",
        "",
        f"- Train clips: {training['num_train_clips']}",
        f"- Train frames: {training['num_train_frames']}",
        f"- Input dim: {training['input_dim']}",
        f"- Training time: {training['seconds']:.2f}s",
        "",
        "## Threshold",
        "",
        f"- Selected threshold: {report['threshold']:.6f}",
        "",
    ]
    for name, summary in (("Evaluation", eval_summary), ("Test", test_summary)):
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Evaluated clips: {summary['evaluated_clips']}",
                f"- Accuracy: {summary['accuracy']:.3f}",
                f"- ROC-AUC: {summary['roc_auc']:.3f}",
                f"- Precision (abnormal): {summary['precision_abnormal']:.3f}",
                f"- Recall (abnormal): {summary['recall_abnormal']:.3f}",
                f"- F1 (abnormal): {summary['f1_abnormal']:.3f}",
                f"- TP / TN / FP / FN: {summary['tp']} / {summary['tn']} / {summary['fp']} / {summary['fn']}",
                f"- Mean clip latency: {summary['mean_latency_seconds']:.2f}s",
                "",
            ]
        )
    return "\n".join(lines)


if nn is not None:
    class _FrameAutoencoder(nn.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            num_hidden_layers: int,
            use_batchnorm: bool,
        ) -> None:
            super().__init__()
            self.input_bn = nn.BatchNorm1d(input_dim, affine=False) if use_batchnorm else nn.Identity()
            encoder_layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_hidden_layers):
                encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            encoder_layers.extend([nn.Linear(hidden_dim, latent_dim), nn.ReLU()])

            decoder_layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_hidden_layers):
                decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            decoder_layers.append(nn.Linear(hidden_dim, input_dim))

            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            normalized = self.input_bn(inputs)
            latent = self.encoder(normalized)
            reconstructed = self.decoder(latent)
            frame_scores = ((normalized - reconstructed) ** 2).mean(dim=1)
            return frame_scores, reconstructed
else:  # pragma: no cover - import guard
    class _FrameAutoencoder:  # type: ignore[no-redef]
        pass


def _train_autoencoder(
    model: _FrameAutoencoder,
    train_matrix: np.ndarray,
    *,
    device: torch.device,
    config: AEBaselineConfig,
) -> list[dict]:
    dataset = TensorDataset(torch.from_numpy(train_matrix.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        sum_loss = 0.0
        total_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            frame_scores, _ = model(batch)
            loss = frame_scores.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(batch.shape[0])
            sum_loss += float(loss.item()) * batch_size
            total_count += batch_size
        epoch_loss = sum_loss / max(total_count, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
    return history


def _score_manifest(
    manifest: list[dict],
    *,
    model: _FrameAutoencoder,
    mel_filterbank: np.ndarray,
    device: torch.device,
    config: AEBaselineConfig,
    standardizer: tuple[np.ndarray, np.ndarray] | None,
) -> list[dict]:
    results = []
    model.eval()
    for item in manifest:
        started = time.perf_counter()
        frames = _context_features_for_clip(Path(item["path"]), mel_filterbank=mel_filterbank, config=config)
        frames = _apply_standardizer(frames, standardizer)
        if config.eval_max_frames_per_clip > 0 and len(frames) > config.eval_max_frames_per_clip:
            indices = np.linspace(0, len(frames) - 1, num=config.eval_max_frames_per_clip)
            frames = frames[np.round(indices).astype(np.int64)]
        with torch.no_grad():
            tensor = torch.from_numpy(frames.astype(np.float32)).to(device)
            frame_scores, _ = model(tensor)
            clip_score = _reduce_frame_scores(frame_scores, config=config)
        latency_seconds = time.perf_counter() - started
        results.append(
            {
                **item,
                "score": clip_score,
                "latency_seconds": latency_seconds,
            }
        )
    return results


def _build_training_matrix(
    manifest: list[dict],
    *,
    mel_filterbank: np.ndarray,
    config: AEBaselineConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    matrices = []
    for item in manifest:
        frames = _context_features_for_clip(Path(item["path"]), mel_filterbank=mel_filterbank, config=config)
        if len(frames) > config.train_frames_per_clip:
            indices = rng.choice(len(frames), size=config.train_frames_per_clip, replace=False)
            frames = frames[np.sort(indices)]
        matrices.append(frames.astype(np.float32))
    return np.concatenate(matrices, axis=0)


def _fit_standardizer(
    train_matrix: np.ndarray,
    *,
    config: AEBaselineConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    if config.input_normalization != "dataset_standardize":
        return None
    mean = train_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    return mean, std


def _apply_standardizer(
    matrix: np.ndarray,
    standardizer: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if standardizer is None:
        return matrix
    mean, std = standardizer
    return ((matrix - mean) / std).astype(np.float32)


def _context_features_for_clip(
    path: Path,
    *,
    mel_filterbank: np.ndarray,
    config: AEBaselineConfig,
) -> np.ndarray:
    sample_rate, waveform = load_audio(path)
    waveform = _resample_if_needed(waveform, sample_rate, config.sample_rate)
    frames = _log_mel_frames(
        waveform,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        mel_filterbank=mel_filterbank,
    )
    return _frame_concat(frames, num_bw=config.num_bw, num_fw=config.num_fw)


def _log_mel_frames(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    mel_filterbank: np.ndarray,
) -> np.ndarray:
    _, _, stft = signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float64)
    mel = mel_filterbank @ magnitude
    log_mel = np.log(mel + 1e-8).T
    return log_mel.astype(np.float32)


def _frame_concat(frames: np.ndarray, *, num_bw: int, num_fw: int) -> np.ndarray:
    total_context = 1 + num_bw + num_fw
    padded = np.pad(frames, ((num_bw, num_fw), (0, 0)), mode="constant")
    contexts = [padded[offset : offset + len(frames)] for offset in range(total_context)]
    return np.concatenate(contexts, axis=1)


def _reduce_frame_scores(frame_scores: torch.Tensor, *, config: AEBaselineConfig) -> float:
    if config.clip_score_mode == "max":
        return float(frame_scores.max().item())
    if config.clip_score_mode == "top_k_mean":
        top_k = max(1, int(math.ceil(len(frame_scores) * config.clip_top_k_fraction)))
        values, _ = torch.topk(frame_scores, k=top_k)
        return float(values.mean().item())
    raise ValueError(f"Unsupported clip score mode: {config.clip_score_mode}")


def _mel_filter_bank(*, sample_rate: int, n_fft: int, num_channels: int) -> np.ndarray:
    fmax = sample_rate / 2.0
    melmax = _hz_to_mel(fmax)
    nmax = int(n_fft / 2 + 1)
    df = sample_rate / n_fft
    dmel = melmax / (num_channels + 1)
    melcenters = np.arange(1, num_channels + 1) * dmel
    fcenters = _mel_to_hz(melcenters)
    indexcenter = (fcenters // df).astype(np.int64)
    if indexcenter[0] == 0:
        indexcenter[0] = 1
    for index in range(1, len(indexcenter)):
        if indexcenter[index - 1] >= indexcenter[index]:
            indexcenter[index] = indexcenter[index - 1] + 1
    indexstart = np.hstack(([0], indexcenter[: num_channels - 1]))
    indexstop = np.hstack((indexcenter[1:num_channels], [nmax]))
    filterbank = np.zeros((num_channels, nmax), dtype=np.float32)
    for channel in range(num_channels):
        start = int(indexstart[channel])
        center = int(indexcenter[channel])
        stop = int(indexstop[channel])
        increment = 1.0 / max(center - start, 1)
        for index in range(start, center):
            filterbank[channel, index] = (index - start) * increment
        decrement = 1.0 / max(stop - center, 1)
        for index in range(center, stop):
            filterbank[channel, index] = 1.0 - ((index - center) * decrement)
        filterbank[channel] /= max(float(filterbank[channel].sum()), 1e-8)
    return filterbank


def _hz_to_mel(freq_hz: float | np.ndarray) -> float | np.ndarray:
    return 1127.01048 * np.log(freq_hz / 700.0 + 1.0)


def _mel_to_hz(freq_mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (np.exp(freq_mel / 1127.01048) - 1.0)


def _resample_if_needed(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32)
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(waveform, up=up, down=down).astype(np.float32)


def _best_f1_threshold(results: list[dict]) -> float:
    scored = sorted(results, key=lambda item: item["score"])
    thresholds = [item["score"] for item in scored]
    best_threshold = thresholds[0] if thresholds else 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        summary = _summarize_scored_results(results, threshold)
        if summary["f1_abnormal"] > best_f1:
            best_f1 = summary["f1_abnormal"]
            best_threshold = threshold
    return float(best_threshold)


def _summarize_scored_results(results: list[dict], threshold: float) -> dict:
    enriched = []
    for item in results:
        prediction = "abnormal" if item["score"] > threshold else "normal"
        enriched.append({**item, "prediction": prediction})

    y_true = np.array([1 if item["expected_label"] == "abnormal" else 0 for item in enriched], dtype=np.int64)
    y_score = np.array([item["score"] for item in enriched], dtype=np.float64)
    tp = sum(1 for item in enriched if item["expected_label"] == "abnormal" and item["prediction"] == "abnormal")
    tn = sum(1 for item in enriched if item["expected_label"] == "normal" and item["prediction"] == "normal")
    fp = sum(1 for item in enriched if item["expected_label"] == "normal" and item["prediction"] == "abnormal")
    fn = sum(1 for item in enriched if item["expected_label"] == "abnormal" and item["prediction"] == "normal")
    evaluated = len(enriched)
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_latency = sum(item["latency_seconds"] for item in enriched) / evaluated if evaluated else 0.0
    return {
        "threshold": float(threshold),
        "evaluated_clips": evaluated,
        "roc_auc": float(roc_auc),
        "accuracy": accuracy,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_seconds": mean_latency,
    }


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch is required for the normal-only AE baseline. "
            "Install torch first."
        ) from TORCH_IMPORT_ERROR
