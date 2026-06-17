from __future__ import annotations

import argparse
from pathlib import Path

from .classical_benchmark import run_classical_zero_shot_benchmark
from .config import DEFAULT_BASELINE_LIMIT, DEFAULT_DATA_ROOT
from .data import sample_clip
from .evaluation import (
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_FALSE_POSITIVE_TRACE,
    build_fixed_ind_evaluation_manifest,
    run_fixed_ind_benchmark,
    write_false_positive_trace_report,
)
from .exemplar_experiments import (
    DEFAULT_EXEMPLAR_OUTPUT_DIR,
    run_ab02_single_anomaly_exemplar_benchmark,
)
from .pipeline import analyze_clip
from .reporting import render_json_report, render_text_report
from .reference_examples_benchmark import (
    DEFAULT_REFERENCE_EXAMPLE_OUTPUT_DIR,
    run_reference_example_benchmark,
)
from .supervised_splits import (
    DEFAULT_BINARY_EVAL_MANIFEST,
    DEFAULT_BINARY_SPLIT_DIR,
    DEFAULT_BINARY_TEST_MANIFEST,
    DEFAULT_BINARY_TRAIN_MANIFEST,
    build_fixed_binary_split_manifests,
)
from .torch_ae_baseline import (
    DEFAULT_AE_OUTPUT_DIR,
    AEBaselineConfig,
    run_normal_only_ae_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ToyADMOS IND-only reference-based acoustic anomaly detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a clip")
    analyze_parser.add_argument("--clip", type=Path, help="Path to a .wav clip")
    analyze_parser.add_argument(
        "--sample-kind",
        choices=["normal", "abnormal"],
        help="Use a built-in sample clip instead of specifying --clip",
    )
    analyze_parser.add_argument("--case", default="case1", help="Sample case to use with --sample-kind")
    analyze_parser.add_argument("--mode", default="IND", help="Sample mode to use with --sample-kind; CNT is reserved for future support")
    analyze_parser.add_argument("--channel", type=int, default=1, help="Sample channel to use with --sample-kind")
    analyze_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    analyze_parser.add_argument(
        "--baseline-limit",
        type=int,
        default=DEFAULT_BASELINE_LIMIT,
        help="Maximum number of baseline clips to use",
    )
    analyze_parser.add_argument(
        "--baseline-channel-strategy",
        choices=["same_channel", "all_channels", "balanced_all_channels"],
        default="same_channel",
        help="How to choose baselines within the same case",
    )
    analyze_parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="DSP feature name to exclude from the LLM payload; can be repeated",
    )
    analyze_parser.add_argument(
        "--reasoner",
        choices=[
            "auto",
            "rule",
            "openai",
            "openai_no_logmel",
            "openai_no_logmel_std",
            "openai_analysis",
            "openai_cause",
            "openai_cause_prior",
            "openai_image",
            "openai_image_describe_label",
            "openai_raw_waveform_image",
            "openai_waveform_image",
            "openai_dual_image",
            "openai_dsp_image",
            "openai_distribution",
            "openai_chebyshev",
            "openai_feature_confidence",
            "openai_baseline_free_dsp",
            "openai_zero_shot_dsp_conservative",
            "openai_zero_shot_dsp_normal_default",
            "openai_zero_shot_dsp_fault_plausibility",
            "openai_zero_shot_cause_scores_b",
            "openai_zero_shot_cause_scores_b_reordered",
            "openai_zero_shot_multiclass",
            "gemini",
            "ollama",
            "ollama_conservative",
        ],
        default="auto",
        help="Reasoning backend",
    )
    analyze_parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    sample_parser = subparsers.add_parser("sample", help="Print a sample clip path")
    sample_parser.add_argument("--kind", choices=["normal", "abnormal"], required=True, help="Sample type")
    sample_parser.add_argument("--case", default="case1", help="Sample case")
    sample_parser.add_argument("--mode", default="IND", help="Sample mode; current benchmark path uses IND")
    sample_parser.add_argument("--channel", type=int, default=1, help="Sample channel")
    sample_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )

    build_eval_parser = subparsers.add_parser("build-evaluation-set", help="Write a fixed IND-only evaluation manifest")
    build_eval_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    build_eval_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="Path for the generated fixed evaluation manifest",
    )
    build_eval_parser.add_argument("--normal-per-stratum", type=int, default=5, help="Normal clips per case-channel stratum")
    build_eval_parser.add_argument(
        "--abnormal-per-stratum",
        type=int,
        default=5,
        help="Abnormal clips per case-channel stratum",
    )
    build_eval_parser.add_argument(
        "--sampling",
        choices=["even", "random"],
        default="even",
        help="How to sample clips within each case-channel-category stratum",
    )
    build_eval_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sampling=random",
    )

    build_binary_parser = subparsers.add_parser(
        "build-binary-splits",
        help="Write fixed IND-only train/evaluation/test manifests for supervised binary baselines",
    )
    build_binary_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    build_binary_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BINARY_SPLIT_DIR,
        help="Directory for the generated split manifests",
    )
    build_binary_parser.add_argument("--train-normal-per-stratum", type=int, default=100)
    build_binary_parser.add_argument("--evaluation-normal-per-stratum", type=int, default=20)
    build_binary_parser.add_argument("--evaluation-abnormal-per-stratum", type=int, default=20)
    build_binary_parser.add_argument("--test-normal-per-stratum", type=int, default=20)
    build_binary_parser.add_argument("--test-abnormal-per-stratum", type=int, default=20)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the fixed IND-only benchmark")
    benchmark_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="Fixed evaluation manifest path",
    )
    benchmark_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    benchmark_parser.add_argument(
        "--baseline-limit",
        type=int,
        default=DEFAULT_BASELINE_LIMIT,
        help="Maximum number of baseline clips to use",
    )
    benchmark_parser.add_argument(
        "--baseline-channel-strategy",
        choices=["same_channel", "all_channels", "balanced_all_channels"],
        default="same_channel",
        help="How to choose baselines within the same case",
    )
    benchmark_parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="DSP feature name to exclude from the LLM payload; can be repeated",
    )
    benchmark_parser.add_argument(
        "--reasoner",
        choices=[
            "openai",
            "openai_no_logmel",
            "openai_no_logmel_std",
            "openai_cause",
            "openai_cause_prior",
            "openai_image",
            "openai_image_describe_label",
            "openai_raw_waveform_image",
            "openai_waveform_image",
            "openai_dual_image",
            "openai_dsp_image",
            "openai_distribution",
            "openai_chebyshev",
            "openai_feature_confidence",
            "openai_baseline_free_dsp",
            "openai_zero_shot_dsp_conservative",
            "openai_zero_shot_dsp_normal_default",
            "openai_zero_shot_dsp_fault_plausibility",
            "openai_zero_shot_cause_scores_b",
            "openai_zero_shot_cause_scores_b_reordered",
            "openai_zero_shot_multiclass",
            "gemini",
            "rule",
            "ollama",
            "ollama_conservative",
        ],
        default="openai",
        help="Reasoning backend",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Directory for benchmark reports",
    )
    benchmark_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
        help="Number of parallel workers for benchmark evaluation",
    )

    fp_parser = subparsers.add_parser("trace-false-positives", help="Trace false positives from a benchmark JSON")
    fp_parser.add_argument("--benchmark-json", type=Path, required=True, help="Benchmark JSON artifact path")
    fp_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    fp_parser.add_argument(
        "--baseline-limit",
        type=int,
        default=DEFAULT_BASELINE_LIMIT,
        help="Maximum number of baseline clips to use",
    )
    fp_parser.add_argument(
        "--baseline-channel-strategy",
        choices=["same_channel", "all_channels", "balanced_all_channels"],
        default="same_channel",
        help="How to choose baselines within the same case",
    )
    fp_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FALSE_POSITIVE_TRACE,
        help="Markdown path for the false-positive trace report",
    )

    classical_parser = subparsers.add_parser(
        "benchmark-classical",
        help="Run classical reference baselines on the fixed IND-only benchmark",
    )
    classical_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="Fixed evaluation manifest path",
    )
    classical_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    classical_parser.add_argument(
        "--baseline-limit",
        type=int,
        default=DEFAULT_BASELINE_LIMIT,
        help="Maximum number of baseline clips to use",
    )
    classical_parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Leave-one-out normal-score quantile used as the anomaly threshold",
    )
    classical_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Directory for benchmark reports",
    )

    ae_parser = subparsers.add_parser(
        "run-ae-baseline",
        help="Train and evaluate a PyTorch normal-only AE baseline on fixed IND manifests",
    )
    ae_parser.add_argument("--train-manifest", type=Path, default=DEFAULT_BINARY_TRAIN_MANIFEST)
    ae_parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_BINARY_EVAL_MANIFEST)
    ae_parser.add_argument("--test-manifest", type=Path, default=DEFAULT_BINARY_TEST_MANIFEST)
    ae_parser.add_argument("--output-dir", type=Path, default=DEFAULT_AE_OUTPUT_DIR)
    ae_parser.add_argument("--epochs", type=int, default=20)
    ae_parser.add_argument("--batch-size", type=int, default=256)
    ae_parser.add_argument("--train-frames-per-clip", type=int, default=32)
    ae_parser.add_argument("--eval-max-frames-per-clip", type=int, default=0)
    ae_parser.add_argument("--num-mels", type=int, default=64)
    ae_parser.add_argument("--num-bw", type=int, default=10)
    ae_parser.add_argument("--num-fw", type=int, default=10)
    ae_parser.add_argument("--hidden-dim", type=int, default=512)
    ae_parser.add_argument("--latent-dim", type=int, default=128)
    ae_parser.add_argument("--num-hidden-layers", type=int, default=4)
    ae_parser.add_argument("--learning-rate", type=float, default=1e-3)
    ae_parser.add_argument("--weight-decay", type=float, default=1e-4)
    ae_parser.add_argument("--seed", type=int, default=0)
    ae_parser.add_argument(
        "--input-normalization",
        choices=["batchnorm", "none", "dataset_standardize"],
        default="batchnorm",
    )
    ae_parser.add_argument(
        "--clip-score-mode",
        choices=["max", "top_k_mean"],
        default="max",
    )
    ae_parser.add_argument("--clip-top-k-fraction", type=float, default=0.05)

    ab02_parser = subparsers.add_parser(
        "benchmark-ab02-exemplars",
        help="Run an ab02 single-anomaly DSP exemplar benchmark with normal and optional abnormal supports",
    )
    ab02_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    ab02_parser.add_argument(
        "--abnormal-support-count",
        type=int,
        choices=[0, 1, 2],
        required=True,
        help="Number of ab02 abnormal support examples per stratum",
    )
    ab02_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXEMPLAR_OUTPUT_DIR,
        help="Directory for exemplar benchmark reports",
    )
    ab02_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
        help="Number of parallel workers",
    )

    ref_examples_parser = subparsers.add_parser(
        "benchmark-reference-examples",
        help="Run a DSP-only benchmark with explicit normal reference examples instead of aggregated baselines",
    )
    ref_examples_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="Evaluation manifest path",
    )
    ref_examples_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for ToyADMOS data",
    )
    ref_examples_parser.add_argument(
        "--reference-count",
        type=int,
        default=4,
        help="Number of normal references to provide explicitly",
    )
    ref_examples_parser.add_argument(
        "--channel-strategy",
        choices=["same_channel", "all_channels"],
        required=True,
        help="How to choose explicit references within the same case",
    )
    ref_examples_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REFERENCE_EXAMPLE_OUTPUT_DIR,
        help="Directory for reference-example benchmark reports",
    )
    ref_examples_parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
        help="Number of parallel workers",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        _run_analyze(args)
    elif args.command == "sample":
        print(
            sample_clip(
                data_root=args.data_root,
                kind=args.kind,
                case=args.case,
                mode=args.mode,
                channel=args.channel,
            )
        )
    elif args.command == "build-evaluation-set":
        artifact = build_fixed_ind_evaluation_manifest(
            data_root=args.data_root,
            output_path=args.output,
            normal_per_stratum=args.normal_per_stratum,
            abnormal_per_stratum=args.abnormal_per_stratum,
            sampling=args.sampling,
            seed=args.seed,
        )
        print(f"manifest_path={artifact['manifest_path']}")
        print(f"clip_count={artifact['clip_count']}")
    elif args.command == "build-binary-splits":
        artifact = build_fixed_binary_split_manifests(
            data_root=args.data_root,
            train_output_path=args.output_dir / DEFAULT_BINARY_TRAIN_MANIFEST.name,
            evaluation_output_path=args.output_dir / DEFAULT_BINARY_EVAL_MANIFEST.name,
            test_output_path=args.output_dir / DEFAULT_BINARY_TEST_MANIFEST.name,
            summary_output_path=args.output_dir / "ind_binary_split_summary.md",
            train_normal_per_stratum=args.train_normal_per_stratum,
            evaluation_normal_per_stratum=args.evaluation_normal_per_stratum,
            evaluation_abnormal_per_stratum=args.evaluation_abnormal_per_stratum,
            test_normal_per_stratum=args.test_normal_per_stratum,
            test_abnormal_per_stratum=args.test_abnormal_per_stratum,
        )
        print(f"train_manifest_path={artifact['train_manifest_path']}")
        print(f"evaluation_manifest_path={artifact['evaluation_manifest_path']}")
        print(f"test_manifest_path={artifact['test_manifest_path']}")
        print(f"summary_path={artifact['summary_path']}")
    elif args.command == "benchmark":
        artifact = run_fixed_ind_benchmark(
            manifest_path=args.manifest,
            reasoner_mode=args.reasoner,
            data_root=args.data_root,
            baseline_limit=args.baseline_limit,
            output_dir=args.output_dir,
            max_workers=args.workers,
            baseline_channel_strategy=args.baseline_channel_strategy,
            exclude_features=args.exclude_feature,
        )
        print(f"report_path={artifact['report_path']}")
        print(f"json_path={artifact['json_path']}")
    elif args.command == "trace-false-positives":
        artifact = write_false_positive_trace_report(
            benchmark_json_path=args.benchmark_json,
            output_path=args.output,
            data_root=args.data_root,
            baseline_limit=args.baseline_limit,
        )
        print(f"trace_report_path={artifact['trace_report_path']}")
        print(f"trace_json_path={artifact['trace_json_path']}")
        print(f"false_positive_count={artifact['false_positive_count']}")
    elif args.command == "benchmark-classical":
        artifact = run_classical_zero_shot_benchmark(
            manifest_path=args.manifest,
            data_root=args.data_root,
            baseline_limit=args.baseline_limit,
            output_dir=args.output_dir,
            quantile=args.quantile,
        )
        print(f"report_path={artifact['report_path']}")
        print(f"json_path={artifact['json_path']}")
    elif args.command == "run-ae-baseline":
        artifact = run_normal_only_ae_baseline(
            train_manifest_path=args.train_manifest,
            evaluation_manifest_path=args.evaluation_manifest,
            test_manifest_path=args.test_manifest,
            output_dir=args.output_dir,
            config=AEBaselineConfig(
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                train_frames_per_clip=args.train_frames_per_clip,
                eval_max_frames_per_clip=args.eval_max_frames_per_clip,
                n_mels=args.num_mels,
                num_bw=args.num_bw,
                num_fw=args.num_fw,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                num_hidden_layers=args.num_hidden_layers,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                input_normalization=args.input_normalization,
                clip_score_mode=args.clip_score_mode,
                clip_top_k_fraction=args.clip_top_k_fraction,
            ),
        )
        print(f"report_path={artifact['report_path']}")
        print(f"json_path={artifact['json_path']}")
        print(f"model_path={artifact['model_path']}")
    elif args.command == "benchmark-ab02-exemplars":
        artifact = run_ab02_single_anomaly_exemplar_benchmark(
            data_root=args.data_root,
            abnormal_support_count=args.abnormal_support_count,
            output_dir=args.output_dir,
            max_workers=args.workers,
        )
        print(f"report_path={artifact['report_path']}")
        print(f"json_path={artifact['json_path']}")
    elif args.command == "benchmark-reference-examples":
        artifact = run_reference_example_benchmark(
            manifest_path=args.manifest,
            data_root=args.data_root,
            reference_count=args.reference_count,
            channel_strategy=args.channel_strategy,
            output_dir=args.output_dir,
            max_workers=args.workers,
        )
        print(f"report_path={artifact['report_path']}")
        print(f"json_path={artifact['json_path']}")


def _run_analyze(args: argparse.Namespace) -> None:
    clip_path = args.clip
    if clip_path is None:
        if not args.sample_kind:
            raise SystemExit("Provide --clip or --sample-kind.")
        clip_path = sample_clip(
            data_root=args.data_root,
            kind=args.sample_kind,
            case=args.case,
            mode=args.mode,
            channel=args.channel,
        )

    try:
        report = analyze_clip(
            clip_path=clip_path,
            data_root=args.data_root,
            baseline_limit=args.baseline_limit,
            reasoner_mode=args.reasoner,
            baseline_channel_strategy=args.baseline_channel_strategy,
            exclude_features=set(args.exclude_feature),
        )
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(render_json_report(report))
    else:
        print(render_text_report(report))
