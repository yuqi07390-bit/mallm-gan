"""Evaluate Adult synthetic data with the MALLM-GAN MLE/F1 protocol.

This script evaluates synthetic Adult CSV files using the MALLM-GAN MLE/F1
protocol and the released `adult-glm.ipynb` Adult setup: train downstream
classifiers on synthetic data and test on a held-out real test set. For
classification tasks, the paper trains Logistic Regression, Random Forest, SVM,
and XGBoost classifiers, reports F1, and averages the best score across
repeated random seeds.

The paper-style metric in this script is `paper_mle_f1`, which is the best F1
among the downstream classifiers for one synthetic run. Per-classifier scores
are also retained for diagnostics and ablations.

Example:
    python evaluate_adult_mle.py

Useful options:
    python evaluate_adult_mle.py --output-dir results/adult_mle_eval
    python evaluate_adult_mle.py --sample-sizes 100 200 400 800
    python evaluate_adult_mle.py --require-xgboost
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

TARGET_COLUMN = "Income"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_FEATURES = [
    col for col in FEATURE_COLUMNS if col not in set(CATEGORICAL_FEATURES)
]

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
ADULT_NOTEBOOK_SAMPLE_SIZES = [100, 200, 400, 800]
PAPER_CLASSIFIERS = ["Logistic Regression", "Random Forest", "SVC", "XGB"]
EXPECTED_TEST_SIZE = 200
EXPECTED_RUNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=Path("sample/Adult"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("gen/adult"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adult_mle_eval"))
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sample sizes to evaluate. Defaults to the released adult-glm.ipynb "
            "sizes that exist in both sample-dir and synthetic-root: "
            "100 200 400 800."
        ),
    )
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
        help="Evaluate without XGBoost if it is unavailable. This deviates from the paper.",
    )
    parser.add_argument(
        "--expected-test-size",
        type=int,
        default=EXPECTED_TEST_SIZE,
        help="Expected held-out real test size from the paper. Use 0 to disable.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
        help="Expected repeated synthetic runs per sample size from the paper.",
    )
    parser.add_argument(
        "--allow-size-mismatch",
        action="store_true",
        help="Allow |Dsyn| to differ from |Dtrain|. This deviates from the paper.",
    )
    parser.add_argument(
        "--strict-paper-protocol",
        action="store_true",
        help="Raise errors instead of warnings for paper-protocol mismatches.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    missing = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[ALL_COLUMNS].copy()


def clean_adult_frame(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Coerce one Adult dataframe into a stable schema for evaluation."""
    cleaned = df[ALL_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        fill_value = pd.to_numeric(reference[col], errors="coerce").median()
        cleaned[col] = cleaned[col].fillna(fill_value)

    for col in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for col in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        mode = reference[col].astype(str).str.strip().mode()
        fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
        cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )


def make_models(seed: int, require_xgboost: bool) -> dict[str, Any]:
    if require_xgboost and XGBClassifier is None:
        raise RuntimeError(
            "xgboost is required but not installed. Install it with: pip install xgboost"
        )

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=5000),
        "Random Forest": RandomForestClassifier(max_depth=3, n_jobs=1, random_state=seed),
        "SVC": SVC(random_state=seed),
    }

    if XGBClassifier is not None:
        models["XGB"] = XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            eval_metric="logloss",
            random_state=seed,
        )

    return models


def warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def available_sample_sizes(sample_dir: Path, synthetic_root: Path) -> list[int]:
    sizes: set[int] = set()
    for train_file in sample_dir.glob("data*.csv"):
        suffix = train_file.stem.removeprefix("data")
        if suffix.isdigit() and (synthetic_root / suffix).exists():
            sizes.add(int(suffix))
    return sorted(sizes)


def default_sample_sizes(
    sample_dir: Path,
    synthetic_root: Path,
) -> list[int]:
    available = available_sample_sizes(sample_dir, synthetic_root)
    notebook_available = [
        size for size in ADULT_NOTEBOOK_SAMPLE_SIZES if size in set(available)
    ]
    if notebook_available:
        return notebook_available
    return available


def encode_labels(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train.astype(str), y_test.astype(str)], ignore_index=True))
    return encoder.transform(y_train.astype(str)), encoder.transform(y_test.astype(str))


def evaluate_train_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    require_xgboost: bool,
) -> dict[str, dict[str, float]]:
    y_train, y_test = encode_labels(train_df[TARGET_COLUMN], test_df[TARGET_COLUMN])
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    results: dict[str, dict[str, float]] = {}
    for model_name, estimator in make_models(seed, require_xgboost).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[model_name] = {
            "f1": float(f1_score(y_test, predictions, average="weighted")),
            "accuracy": float(accuracy_score(y_test, predictions)),
        }

    return results


def paper_mle_result(model_results: dict[str, dict[str, float]]) -> tuple[str, float]:
    best_model = max(model_results, key=lambda name: model_results[name]["f1"])
    return best_model, model_results[best_model]["f1"]


def synthetic_files_for_size(synthetic_root: Path, sample_size: int) -> list[Path]:
    folder = synthetic_root / str(sample_size)
    if not folder.exists():
        raise FileNotFoundError(f"Missing synthetic folder: {folder}")

    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No synthetic CSV files found in: {folder}")
    return files


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_size": int(group["sample_size"].iloc[0]),
        "n_runs": int(len(group)),
        "paper_mle_f1_mean": float(group["paper_mle_f1"].mean()),
        "paper_mle_f1_std": float(group["paper_mle_f1"].std(ddof=0)),
        "paper_mle_f1_standard_error": float(
            group["paper_mle_f1"].std(ddof=1) / math.sqrt(len(group))
        )
        if len(group) > 1
        else 0.0,
        "paper_mle_f1_min": float(group["paper_mle_f1"].min()),
        "paper_mle_f1_max": float(group["paper_mle_f1"].max()),
        "paper_mle_model_accuracy_mean": float(
            group["paper_mle_model_accuracy"].mean()
        ),
        "paper_mle_model_accuracy_std": float(group["paper_mle_model_accuracy"].std(ddof=0)),
    }

    for col in group.columns:
        if col.endswith("_f1") or col.endswith("_accuracy"):
            summary[f"{col}_mean"] = float(group[col].mean())
            summary[f"{col}_std"] = float(group[col].std(ddof=0))

    return summary


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

    reference_for_test = raw_test.copy()
    test_df = clean_adult_frame(raw_test, reference_for_test)

    run_rows: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    real_baseline_rows: list[dict[str, Any]] = []

    for sample_size in sample_sizes:
        train_path = args.sample_dir / f"data{sample_size}.csv"
        raw_train = read_required_csv(train_path)
        reference = raw_train.copy()
        real_train = clean_adult_frame(raw_train, reference)

        real_results = evaluate_train_on_test(
            real_train,
            test_df,
            seed=args.seed,
            require_xgboost=args.require_xgboost,
        )
        paper_mle_model, paper_mle_f1 = paper_mle_result(real_results)
        real_baseline_rows.append(
            {
                "sample_size": sample_size,
                "train_file": str(train_path),
                "paper_mle_model": paper_mle_model,
                "paper_mle_f1": paper_mle_f1,
                "paper_mle_model_accuracy": real_results[paper_mle_model]["accuracy"],
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
                    f"synthetic runs, expected {args.expected_runs} under the "
                    "paper protocol."
                ),
                args.strict_paper_protocol,
            )

        for run_index, synthetic_path in enumerate(synthetic_paths):
            raw_synthetic = read_required_csv(synthetic_path)
            synthetic = clean_adult_frame(raw_synthetic, reference)
            if not args.allow_size_mismatch and len(synthetic) != len(real_train):
                warn_or_raise(
                    (
                        f"{synthetic_path} has {len(synthetic)} rows but the "
                        f"matching real training split has {len(real_train)} rows. "
                        "The paper uses |Dtrain| = |Dsyn|."
                    ),
                    args.strict_paper_protocol,
                )

            model_results = evaluate_train_on_test(
                synthetic,
                test_df,
                seed=args.seed + run_index,
                require_xgboost=args.require_xgboost,
            )
            paper_mle_model, paper_mle_f1 = paper_mle_result(model_results)
            paper_mle_model_accuracy = model_results[paper_mle_model]["accuracy"]

            row = {
                "sample_size": sample_size,
                "run_index": run_index,
                "synthetic_file": str(synthetic_path),
                "n_synthetic": int(len(synthetic)),
                "paper_mle_model": paper_mle_model,
                "paper_mle_f1": paper_mle_f1,
                "paper_mle_model_accuracy": paper_mle_model_accuracy,
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
                    "paper_mle_model": paper_mle_model,
                    "paper_mle_f1": paper_mle_f1,
                    "paper_mle_model_accuracy": paper_mle_model_accuracy,
                    "model_results": model_results,
                }
            )

    runs_df = pd.DataFrame(run_rows)
    summary_df = pd.DataFrame(
        [summarize_group(group) for _, group in runs_df.groupby("sample_size")]
    )
    real_baseline_df = pd.DataFrame(real_baseline_rows)

    runs_path = args.output_dir / "adult_mle_runs.csv"
    summary_path = args.output_dir / "adult_mle_summary.csv"
    real_baseline_path = args.output_dir / "adult_real_baseline.csv"
    details_path = args.output_dir / "adult_mle_details.json"
    summary_json_path = args.output_dir / "adult_mle_summary.json"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    real_baseline_df.to_csv(real_baseline_path, index=False)
    write_json(details_path, detail_records)
    write_json(
        summary_json_path,
        {
            "protocol": (
                "Paper-style MLE/F1: train Logistic Regression, Random Forest, "
                "SVC, and XGBoost classifiers on synthetic Adult data, evaluate "
                "weighted F1 on the held-out real test set, and average the best "
                "per-run classifier score across repeated random seeds."
            ),
            "sample_dir": str(args.sample_dir),
            "synthetic_root": str(args.synthetic_root),
            "test_file": str(test_path),
            "sample_sizes": sample_sizes,
            "adult_notebook_sample_sizes": ADULT_NOTEBOOK_SAMPLE_SIZES,
            "expected_test_size": args.expected_test_size,
            "expected_runs": args.expected_runs,
            "allow_size_mismatch": args.allow_size_mismatch,
            "cleaning_reference": (
                "Synthetic and real training rows are cleaned using only the "
                "matching real training split as the imputation reference. The "
                "held-out test split is cleaned independently."
            ),
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

    print("Adult MLE/F1 evaluation complete.")
    print(f"Wrote: {runs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {real_baseline_path}")
    print(f"Wrote: {details_path}")
    print(f"Wrote: {summary_json_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
