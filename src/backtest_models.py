"""Expanding-window walk-forward evaluation for the final CatBoost design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling import (
        MODEL_FEATURES,
        SEED,
        augment_team_swap,
        bootstrap_interval,
        catboost_classifier,
        load_feature_dataset,
        metric_row,
        sha256_file,
        symmetrized_probability,
        tune_elo,
        update_artifact_manifest,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from modeling import (  # type: ignore[no-redef]
        MODEL_FEATURES,
        SEED,
        augment_team_swap,
        bootstrap_interval,
        catboost_classifier,
        load_feature_dataset,
        metric_row,
        sha256_file,
        symmetrized_probability,
        tune_elo,
        update_artifact_manifest,
        write_json,
    )


DEFAULT_FOLDS = [
    ("2025-Q3", "2025-07-01", "2025-10-01"),
    ("2025-Q4", "2025-10-01", "2026-01-01"),
    ("2026-Q1", "2026-01-01", "2026-04-01"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an expanding-window walk-forward diagnostic with a separate "
            "inner validation window for Elo K and CatBoost early stopping."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--inner-validation-days", type=int, default=90)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def fold_masks(
    frame: pd.DataFrame,
    *,
    test_start: str,
    test_end: str,
    validation_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if validation_days <= 0:
        raise ValueError("validation_days must be positive")
    start = pd.Timestamp(test_start, tz="UTC")
    end = pd.Timestamp(test_end, tz="UTC")
    if end <= start:
        raise ValueError("test_end must be later than test_start")
    validation_start = start - pd.Timedelta(days=validation_days)
    timestamps = frame["match_datetime_utc"]
    train = (timestamps < validation_start).to_numpy()
    validation = ((timestamps >= validation_start) & (timestamps < start)).to_numpy()
    test = ((timestamps >= start) & (timestamps < end)).to_numpy()
    counts = tuple(int(mask.sum()) for mask in (train, validation, test))
    if min(counts) == 0:
        raise ValueError(f"Walk-forward fold contains an empty part: {counts}")
    return train, validation, test


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    frame = load_feature_dataset(args.features)
    target = frame["team1_win"].astype(int).to_numpy()
    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []

    for fold_index, (fold_name, test_start, test_end) in enumerate(DEFAULT_FOLDS):
        train_mask, validation_mask, test_mask = fold_masks(
            frame,
            test_start=test_start,
            test_end=test_end,
            validation_days=args.inner_validation_days,
        )
        best_k, elo_probability, elo_team1, elo_team2, elo_candidates = tune_elo(
            frame,
            validation_mask,
        )
        features = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).copy()
        features["diff_elo_pre"] = elo_team1 - elo_team2
        x_train = features.loc[train_mask]
        x_validation = features.loc[validation_mask]
        x_test = features.loc[test_mask]
        y_train = target[train_mask]
        y_validation = target[validation_mask]
        y_test = target[test_mask]
        if min(len(np.unique(values)) for values in (y_train, y_validation, y_test)) < 2:
            raise ValueError(f"Fold {fold_name} contains a single-class split")

        x_train_augmented, y_train_augmented = augment_team_swap(x_train, y_train)
        x_validation_augmented, y_validation_augmented = augment_team_swap(
            x_validation, y_validation
        )
        model = catboost_classifier(
            iterations=args.catboost_iterations,
            seed=args.seed + fold_index,
        )
        model.fit(
            x_train_augmented,
            y_train_augmented,
            eval_set=(x_validation_augmented, y_validation_augmented),
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=False,
        )
        catboost_probability = symmetrized_probability(model, x_test)
        fold_predictions = frame.loc[
            test_mask,
            ["match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"],
        ].copy()
        fold_predictions.insert(0, "fold", fold_name)
        fold_predictions["prob_elo"] = elo_probability[test_mask]
        fold_predictions["prob_catboost"] = catboost_probability
        prediction_parts.append(fold_predictions)

        fold_metrics: dict[str, dict[str, float]] = {}
        for model_name, probability in [
            ("Elo", elo_probability[test_mask]),
            ("CatBoost", catboost_probability),
        ]:
            values = metric_row(y_test, probability)
            fold_metrics[model_name] = values
            metric_rows.append(
                {
                    "scope": "fold",
                    "fold": fold_name,
                    "model": model_name,
                    "n": int(test_mask.sum()),
                    **values,
                }
            )
        fold_summaries.append(
            {
                "fold": fold_name,
                "train_end_exclusive": (
                    pd.Timestamp(test_start, tz="UTC")
                    - pd.Timedelta(days=args.inner_validation_days)
                ).isoformat(),
                "validation_start": (
                    pd.Timestamp(test_start, tz="UTC")
                    - pd.Timedelta(days=args.inner_validation_days)
                ).isoformat(),
                "validation_end_exclusive": pd.Timestamp(test_start, tz="UTC").isoformat(),
                "test_start": pd.Timestamp(test_start, tz="UTC").isoformat(),
                "test_end_exclusive": pd.Timestamp(test_end, tz="UTC").isoformat(),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "elo_k": best_k,
                "elo_validation_candidates": elo_candidates,
                "catboost_best_iteration": int(model.get_best_iteration()),
                "test_metrics": fold_metrics,
            }
        )
        print(
            f"{fold_name}: train={int(train_mask.sum())}, "
            f"validation={int(validation_mask.sum())}, test={int(test_mask.sum())}"
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    combined_target = predictions["team1_win"].astype(int).to_numpy()
    aggregate: dict[str, dict[str, object]] = {}
    for model_name, column in [("Elo", "prob_elo"), ("CatBoost", "prob_catboost")]:
        probability = predictions[column].to_numpy(dtype=float)
        values: dict[str, object] = metric_row(combined_target, probability)
        values.update(
            bootstrap_interval(
                combined_target,
                probability,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
        )
        aggregate[model_name] = values
        metric_rows.append(
            {
                "scope": "aggregate",
                "fold": "all",
                "model": model_name,
                "n": len(predictions),
                **{key: value for key, value in values.items() if isinstance(value, float)},
            }
        )

    metrics_path = args.output_dir / "walk_forward_metrics.csv"
    predictions_path = args.output_dir / "walk_forward_predictions.csv"
    summary_path = args.output_dir / "walk_forward_summary.json"
    write_csv_atomic(pd.DataFrame(metric_rows), metrics_path)
    write_csv_atomic(predictions, predictions_path)
    write_json(
        summary_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "Stability diagnostic. Every test quarter is evaluated after an expanding "
                "training window and a disjoint 90-day inner validation window."
            ),
            "dataset_sha256": sha256_file(args.features),
            "feature_count": len(MODEL_FEATURES),
            "team_swap_augmentation": True,
            "symmetric_inference": True,
            "folds": fold_summaries,
            "aggregate": aggregate,
        },
    )
    update_artifact_manifest(
        args.output_dir,
        [metrics_path.name, predictions_path.name, summary_path.name],
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
