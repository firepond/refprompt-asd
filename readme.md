# refprompt-asd

Reference-based prompting for low-shot acoustic anomaly detection.

This repository contains code and public-facing experiment materials for the MobiSys 2026 poster paper:

**Poster: Reference-Based Prompting for Acoustic Anomaly Detection**

The project studies whether large language models can classify machine sound clips as `normal` or `anomalous` from compact DSP summaries. The main setting is reference-based: a query clip is compared against a small set of matched normal reference clips from the same ToyADMOS ToyConveyor case and recording channel.

## What This Repository Contains

- `agent_toy_ad/`: Python implementation for feature extraction, baseline aggregation, prompting, benchmarking, and classical baselines.
- `main.py`: CLI entry point.
- `paper/`: accepted poster paper source/PDF and result table.
- `docs/`: poster copy and public-facing project notes.
- `results/`: compact public result summaries.
- `data/README.md`: expected dataset layout. The ToyADMOS audio data are not redistributed in this repository.

## Core Idea

The pipeline is:

1. Select matched normal references from the same machine case and channel.
2. Extract scalar DSP features from the query and reference clips.
3. Aggregate references into per-feature mean and standard deviation.
4. Compute query-to-baseline deltas and z-scores.
5. Ask an LLM to output one label: `normal` or `abnormal`.

This frames acoustic anomaly detection as reference-based comparison rather than pure zero-shot judgment.

## Dataset Scope

Experiments use the ToyADMOS `ToyConveyor` subset with `IND` recordings only. The task is binary anomaly detection:

- `normal`
- `anomalous`

ToyConveyor anomaly codes are treated as condition combinations, not clean single-cause root-cause labels. This repository therefore focuses on binary detection rather than root-cause classification.

## DSP Features

Each clip is summarized by 11 scalar features:

- `rms_energy`
- `zero_crossing_rate`
- `spectral_centroid`
- `spectral_bandwidth`
- `spectral_rolloff`
- `dominant_frequency`
- `harmonic_strength`
- `high_frequency_ratio`
- `burst_count`
- `log_mel_mean`
- `log_mel_std`

## Main Findings

The main public findings are:

- Pure zero-shot prompting was unreliable.
- Matched normal references made the task more stable.
- LLM prompting was competitive in the 4-shot regime.
- A classical one-class SVM became stronger at 16-shot.
- Same-channel references were more useful than mixed-channel references.
- Aggregated baseline statistics worked better than listing raw reference examples.
- Correlated DSP features can inflate false-positive evidence.

The safest interpretation is that LLM prompting is useful as a training-free low-shot comparator, not as a universal replacement for classical anomaly detectors.

## Held-Out Test Summary

| Method | Shot | Accuracy | F1 |
|---|---:|---:|---:|
| GPT-5.4-mini | 4 | 0.846 | 0.850 |
| Gemini 3.1 Flash-Lite | 4 | 0.791 | 0.812 |
| one-class SVM | 4 | 0.783 | 0.779 |
| GPT-5.4-mini | 16 | 0.865 | 0.877 |
| Gemini 3.1 Flash-Lite | 16 | 0.881 | 0.888 |
| one-class SVM | 16 | 0.940 | 0.938 |

See `results/main_results.md` for a slightly fuller version.

## Installation

This project uses Python 3.11+.

```bash
uv sync
```

If you do not use `uv`, install the dependencies from `pyproject.toml` in a virtual environment.

## Data Layout

Place ToyADMOS ToyConveyor clips under:

```text
data/ToyConveyor/
  case1/
    NormalSound_IND/
    AnomalousSound_IND/
  case2/
    NormalSound_IND/
    AnomalousSound_IND/
  case3/
    NormalSound_IND/
    AnomalousSound_IND/
```

The data are not included. See `data/README.md`.

## Example Commands

Build an evaluation manifest:

```bash
uv run python main.py build-evaluation-set \
  --sampling random \
  --seed 0 \
  --output reports/ind_evaluation_manifest_random_seed0.json
```

Run an OpenAI benchmark:

```bash
OPENAI_MODEL=gpt-5.4-mini \
uv run python main.py benchmark \
  --manifest reports/ind_evaluation_manifest_random_seed0.json \
  --reasoner openai \
  --baseline-limit 16 \
  --baseline-channel-strategy same_channel \
  --exclude-feature rms_energy \
  --workers 4 \
  --output-dir reports/openai_example
```

Run the classical one-class SVM baseline:

```bash
uv run python main.py run-classical-benchmark \
  --manifest reports/ind_evaluation_manifest_random_seed0.json \
  --baseline-limit 16 \
  --method one_class_svm \
  --output-dir reports/classical_example
```

Model API keys are read from environment variables such as `OPENAI_API_KEY` and `GEMINI_API_KEY`.

## Repository Notes

This is a research artifact for a poster paper. It is intentionally scoped to the ToyADMOS ToyConveyor experiments and should not be treated as a production anomaly detection system.

