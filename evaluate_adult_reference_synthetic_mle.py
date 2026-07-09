"""Evaluate Adult reference-synthetic outputs with the MALLM-GAN MLE/F1 protocol.

This is a convenience wrapper for the direction-2 pipeline outputs. It keeps
``evaluate_adult_mle.py`` unchanged and defaults to evaluating:

    gen/adult_reference_synthetic/{sample_size}/df_{seed}.csv

against the held-out real Adult test split in:

    sample/Adult/data_test.csv

Example:
    python evaluate_adult_reference_synthetic_mle.py

Useful options:
    python evaluate_adult_reference_synthetic_mle.py --sample-sizes 100
    python evaluate_adult_reference_synthetic_mle.py --allow-missing-xgboost
    python evaluate_adult_reference_synthetic_mle.py --strict-paper-protocol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_adult_mle import (
    ADULT_NOTEBOOK_SAMPLE_SIZES,
    EXPECTED_RUNS,
    EXPECTED_TEST_SIZE,
    PAPER_CLASSIFIERS,
    XGBClassifier,
    clean_adult_frame,
    default_sample_sizes,
    evaluate_train_on_test,
    make_models,
    paper_mle_result,
    read_required_csv,
    summarize_group,
    synthetic_files_for_size,
    warn_or_raise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("sample/Adult"))
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path("gen/adult_reference_synthetic"),
        help="Root folder containing the reference-synthetic outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adult_reference_synthetic_mle_eval"),
    )
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.set_defaults(require_xgboost=True)
    parser.add_argument(
        "--require-xgboost",
        dest="require_xgboost",
        action="store_true",
        help="Fail if xgboost is not installed. This is the paper-style default.",
    )
    parser.add_argument(
        "--allow-missing-xgboost",
        dest="require_xgboost",
        action="store_false",
        help="Evaluate without XGBoost if it is unavailable.",
    )
    parser.add_argument("--expected-test-size", type=int, default=EXPECTED_TEST_SIZE)
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    parser.add_argument(
        "--allow-size-mismatch",
        action="store_true",
        help="Allow |Dsyn| to differ from |Dtrain|.",
    )
    parser.add_argument(
        "--strict-paper-protocol",
        action="store_true",
        help="Raise errors instead of warnings for protocol mismatches.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = args.sample_sizes or default_sample_sizes(
        args.sample_dir,
        args.synthetic_root,
    )
    if not sample_sizes:
        raise FileNotFoundError(
            "No matching Adult sample sizes found in sample-dir and synthetic-root."
        )

    test_path = args.test_file or (args.sample_dir / "data_test.csv")
    raw_test = read_required_csv(test_path)
    if args.expected_test_size and len(raw_test) != args.expected_test_size:
        warn_or_raise(
            (
                f"Held-out test size is {len(raw_test)}, expected "
                f"{args.expected_test_size} under the paper protocol."
            ),
            args.strict_paper_protocol,
        )
    test_df = clean_adult_frame(raw_test, raw_test.copy())

    run_rows: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    real_baseline_rows: list[dict[str, Any]] = []

    for sample_size in sample_sizes:
        train_path = args.sample_dir / f"data{sample_size}.csv"
        raw_train = read_required_csv(train_path)
        real_train = clean_adult_frame(raw_train, raw_train.copy())

        real_results = evaluate_train_on_test(
            real_train,
            test_df,
            seed=args.seed,
            require_xgboost=args.require_xgboost,
        )
        real_best_model, real_best_f1 = paper_mle_result(real_results)
        real_baseline_rows.append(
            {
                "sample_size": sample_size,
                "train_file": str(train_path),
                "paper_mle_model": real_best_model,
                "paper_mle_f1": real_best_f1,
                "paper_mle_model_accuracy": real_results[real_best_model]["accuracy"],
                **{
                    f"{model_name}_f1": metrics["f1"]
                    for model_name, metrics in real_results.items()
                },
                **{
                    f"{model_name}_accuracy": metrics["accuracy"]
                    for model_name, metrics in real_results.items()
                },
            }
        )

        synthetic_paths = synthetic_files_for_size(args.synthetic_root, sample_size)
        if args.expected_runs and len(synthetic_paths) != args.expected_runs:
            warn_or_raise(
                (
                    f"Sample size {sample_size} has {len(synthetic_paths)} "
                    f"synthetic runs, expected {args.expected_runs}."
                ),
                args.strict_paper_protocol,
            )

        for run_index, synthetic_path in enumerate(synthetic_paths):
            raw_synthetic = read_required_csv(synthetic_path)
            synthetic = clean_adult_frame(raw_synthetic, raw_train.copy())

            if not args.allow_size_mismatch and len(synthetic) != len(real_train):
                warn_or_raise(
                    (
                        f"{synthetic_path} has {len(synthetic)} rows but the "
                        f"matching real training split has {len(real_train)} rows."
                    ),
                    args.strict_paper_protocol,
                )

            model_results = evaluate_train_on_test(
                synthetic,
                test_df,
                seed=args.seed + run_index,
                require_xgboost=args.require_xgboost,
            )
            best_model, best_f1 = paper_mle_result(model_results)
            best_accuracy = model_results[best_model]["accuracy"]

            row = {
                "sample_size": sample_size,
                "run_index": run_index,
                "synthetic_file": str(synthetic_path),
                "n_synthetic": int(len(synthetic)),
                "paper_mle_model": best_model,
                "paper_mle_f1": best_f1,
                "paper_mle_model_accuracy": best_accuracy,
            }
            for model_name, metrics in model_results.items():
                row[f"{model_name}_f1"] = metrics["f1"]
                row[f"{model_name}_accuracy"] = metrics["accuracy"]

            run_rows.append(row)
            detail_records.append(
                {
                    "sample_size": sample_size,
                    "synthetic_file": str(synthetic_path),
                    "n_synthetic": int(len(synthetic)),
                    "paper_mle_model": best_model,
                    "paper_mle_f1": best_f1,
                    "paper_mle_model_accuracy": best_accuracy,
                    "model_results": model_results,
                }
            )

    runs_df = pd.DataFrame(run_rows)
    summary_df = pd.DataFrame(
        [summarize_group(group) for _, group in runs_df.groupby("sample_size")]
    )
    real_baseline_df = pd.DataFrame(real_baseline_rows)

    runs_path = args.output_dir / "adult_reference_synthetic_mle_runs.csv"
    summary_path = args.output_dir / "adult_reference_synthetic_mle_summary.csv"
    real_baseline_path = args.output_dir / "adult_real_baseline.csv"
    details_path = args.output_dir / "adult_reference_synthetic_mle_details.json"
    summary_json_path = args.output_dir / "adult_reference_synthetic_mle_summary.json"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    real_baseline_df.to_csv(real_baseline_path, index=False)
    write_json(details_path, detail_records)
    write_json(
        summary_json_path,
        {
            "protocol": (
                "Reference-synthetic Adult MLE/F1 evaluation: train downstream "
                "classifiers on gen/adult_reference_synthetic outputs and test "
                "on the held-out real Adult test split."
            ),
            "sample_dir": str(args.sample_dir),
            "synthetic_root": str(args.synthetic_root),
            "test_file": str(test_path),
            "sample_sizes": sample_sizes,
            "adult_notebook_sample_sizes": ADULT_NOTEBOOK_SAMPLE_SIZES,
            "expected_test_size": args.expected_test_size,
            "expected_runs": args.expected_runs,
            "allow_size_mismatch": args.allow_size_mismatch,
            "paper_classifiers": [
                model_name
                for model_name in PAPER_CLASSIFIERS
                if model_name in make_models(args.seed, args.require_xgboost)
            ],
            "xgboost_available": XGBClassifier is not None,
            "summary": summary_df.to_dict(orient="records"),
            "real_data_baseline": real_baseline_df.to_dict(orient="records"),
        },
    )

    print("Adult reference-synthetic MLE/F1 evaluation complete.")
    print(f"Wrote: {runs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {real_baseline_path}")
    print(f"Wrote: {details_path}")
    print(f"Wrote: {summary_json_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
