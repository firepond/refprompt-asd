from __future__ import annotations

import json

from .config import FEATURE_LABELS
from .models import AnalysisReport


def render_text_report(report: AnalysisReport) -> str:
    return report.reasoning.prediction


def render_json_report(report: AnalysisReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
