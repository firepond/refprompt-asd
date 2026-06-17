# Reproducibility Notes

This repository contains the code used for the reference-based prompting experiments, but it does not include ToyADMOS audio data or API credentials.

## Environment

```bash
uv sync
```

## Required Environment Variables

For OpenAI:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4-mini
```

For Gemini:

```bash
export GEMINI_API_KEY=...
export GEMINI_MODEL=gemini-3.1-flash-lite-preview
```

## Example Benchmark

```bash
uv run python main.py benchmark \
  --manifest reports/ind_evaluation_manifest_random_seed0.json \
  --reasoner openai \
  --baseline-limit 16 \
  --baseline-channel-strategy same_channel \
  --exclude-feature rms_energy \
  --workers 4 \
  --output-dir reports/openai_example
```

## Notes

- Results depend on the selected LLM, model version, prompting mode, and API behavior.
- The public results are intended to document the poster experiments, not to define a stable benchmark leaderboard.
- The ToyADMOS audio data must be obtained separately.

