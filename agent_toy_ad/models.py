from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ClipMetadata:
    path: str
    filename: str
    machine_type: str
    case: str | None
    category: str
    mode: str | None
    channel: int | None
    anomaly_code: str | None = None


@dataclass
class FeatureSet:
    sample_rate: int
    values: dict[str, float]


@dataclass
class BaselineProfile:
    clip_paths: list[str]
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class FeatureComparison:
    feature: str
    value: float
    baseline_mean: float
    baseline_std: float
    delta: float
    relative_delta: float
    z_score: float
    salience: float
    severity: str
    direction: str


@dataclass
class Observation:
    feature: str
    text: str
    severity: str
    salience: float


@dataclass
class ReasoningResult:
    source: str
    prediction: str
    confidence: str
    explanation: str
    predicted_cause: str | None = None
    evidence: list[str] = field(default_factory=list)
    possible_causes: list[str] = field(default_factory=list)
    suggested_checks: list[str] = field(default_factory=list)
    raw_response: str | None = None


@dataclass
class AnalysisReport:
    metadata: ClipMetadata
    baseline: BaselineProfile
    target_features: FeatureSet
    comparisons: dict[str, FeatureComparison]
    observations: list[Observation]
    reasoning: ReasoningResult
    mode_handler: str = "ind_reference_baseline"
    mode_details: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
