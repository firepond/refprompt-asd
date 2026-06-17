# Main Results

This file summarizes the main public results used in the MobiSys 2026 poster paper.

## Held-Out Test

| Method | Shot | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| GPT-5.4-mini | 4 | 0.846 | 0.873 | 0.850 |
| Gemini 3.1 Flash-Lite | 4 | 0.791 | 0.873 | 0.812 |
| one-class SVM | 4 | 0.783 | -- | 0.779 |
| GPT-5.4-mini | 16 | 0.865 | 0.952 | 0.877 |
| Gemini 3.1 Flash-Lite | 16 | 0.881 | 0.952 | 0.888 |
| one-class SVM | 16 | 0.940 | -- | 0.938 |

LLM rows use the best validation-tuned feature subset for each shot regime. The one-class SVM is a classical normal-only baseline fitted on the same number of matched normal references.

## Validation / Evaluation Split

| Method | Shot | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| GPT-5.4-mini | 1 | 0.633 | 0.596 | 0.593 |
| GPT-5.4-mini | 2 | 0.608 | 0.793 | 0.712 |
| GPT-5.4-mini | 4 | 0.858 | 0.879 | 0.860 |
| GPT-5.4-mini | 16 | 0.950 | 0.971 | 0.952 |
| one-class SVM | 2 | 0.575 | -- | 0.662 |

One-class SVM with 1-shot is undefined under the leave-one-out threshold protocol used in these experiments.

