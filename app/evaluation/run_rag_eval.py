"""RAG 离线评测 CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.loaders import load_jsonl_cases
from app.evaluation.report import summarize_results, write_reports


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        argparse 参数解析器。
    """
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument(
        "--dataset",
        default="data/eval/rag_cases.jsonl",
        help="Path to JSONL eval dataset.",
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        help="Evaluation profile name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="TopK used by Recall/Hit/MRR metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
        help="Directory for Markdown and JSON reports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run first N cases; 0 means all.",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Send eval traces and scores to Langfuse.",
    )
    parser.add_argument(
        "--baseline-report",
        default="",
        help="Path to previous JSON report for baseline comparison.",
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Path to metric thresholds JSON. Empty value uses built-in defaults.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with code 2 when any metric is below threshold.",
    )
    return parser


def main() -> int:
    """执行 RAG 离线评测命令。

    Returns:
        进程退出码，评测运行完成返回 0。
    """
    args = build_parser().parse_args()
    cases = load_jsonl_cases(Path(args.dataset))
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    from app.evaluation.langfuse_eval import flush
    from app.evaluation.rag_runner import run_eval_cases
    from app.evaluation.baseline import compare_summary, load_report_summary
    from app.evaluation.thresholds import evaluate_thresholds, load_thresholds

    print(f"loaded_cases={len(cases)}")
    results = run_eval_cases(
        cases=cases,
        profile=args.profile,
        k=args.top_k,
        enable_langfuse=args.langfuse,
    )
    summary = summarize_results(results)
    baseline_comparison = {}
    if args.baseline_report:
        baseline_summary = load_report_summary(args.baseline_report)
        baseline_comparison = compare_summary(summary, baseline_summary)

    thresholds = load_thresholds(args.thresholds or None)
    threshold_violations = evaluate_thresholds(summary, thresholds)

    paths = write_reports(
        results,
        profile=args.profile,
        output_dir=args.output_dir,
        baseline_comparison=baseline_comparison,
        threshold_violations=threshold_violations,
    )
    flush(args.langfuse)

    print("RAG evaluation completed.")
    print(f"total_cases={summary['total_cases']}")
    print(f"pass_rate={summary['pass_rate']:.4f}")
    for metric_name, value in summary["metrics"].items():
        print(f"{metric_name}={value:.4f}")
    if baseline_comparison:
        print(f"baseline_pass_rate_delta={baseline_comparison['pass_rate']['delta']:.4f}")
    if threshold_violations:
        print("threshold_violations:")
        for item in threshold_violations:
            print(
                "  {metric}: current={current:.4f}, threshold={threshold:.4f}, delta={delta:.4f}".format(
                    metric=item["metric"],
                    current=item["current"],
                    threshold=item["threshold"],
                    delta=item["delta"],
                )
            )
    print(f"markdown_report={paths['markdown']}")
    print(f"json_report={paths['json']}")
    if args.fail_on_threshold and threshold_violations:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
