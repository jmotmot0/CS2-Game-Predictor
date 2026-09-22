"""Legacy September-3 training protocol (retained for historical reproduction).

The current research protocol is ``python -m src.research_revision``. Calibration
below belongs to the archived diagnostic, not the revised model comparison.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

# В некоторых версиях Windows joblib не может получить число ядер через WMIC.
# Явное ограничение числа потоков исключает этот запрос и стабилизирует запуск.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(os.cpu_count() or 1, 8)))

import numpy as np
import pandas as pd

try:
    from src.modeling import (
        ABLATION_GROUPS,
        MODEL_SCHEMA_VERSION,
        MODEL_FEATURES,
        SEED,
        TRAIN_END,
        VALIDATION_END,
        augment_team_swap,
        bootstrap_interval,
        catboost_classifier,
        chronological_masks,
        library_versions,
        load_feature_dataset,
        metric_row,
        segment_metrics,
        sha256_file,
        symmetrized_probability,
        tune_elo,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from modeling import (  # type: ignore[no-redef]
        ABLATION_GROUPS,
        MODEL_SCHEMA_VERSION,
        MODEL_FEATURES,
        SEED,
        TRAIN_END,
        VALIDATION_END,
        augment_team_swap,
        bootstrap_interval,
        catboost_classifier,
        chronological_masks,
        library_versions,
        load_feature_dataset,
        metric_row,
        segment_metrics,
        sha256_file,
        symmetrized_probability,
        tune_elo,
        write_json,
    )


MODEL_ORDER = [
    "Historical prior",
    "Elo",
    "Logistic regression",
    "Histogram gradient boosting",
    "CatBoost",
    "CatBoost + Platt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline and CatBoost models with chronological validation."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/interim/hltv_final_clean"),
        help="Clean relational tables used to build the fast inference state.",
    )
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--validation-end", default=VALIDATION_END)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the four additional CatBoost fits used for the ablation table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    print(f"Loading features from {args.features}")
    frame = load_feature_dataset(args.features)
    train_mask, validation_mask, test_mask = chronological_masks(
        frame,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    target = frame["team1_win"].astype(int).to_numpy()

    print("Selecting Elo K on validation data")
    best_k, elo_probability, elo_team1, elo_team2, elo_candidates = tune_elo(
        frame,
        validation_mask,
    )
    frame["team1_elo_pre"] = elo_team1
    frame["team2_elo_pre"] = elo_team2
    frame["diff_elo_pre"] = elo_team1 - elo_team2

    features = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    x_train = features.loc[train_mask]
    x_validation = features.loc[validation_mask]
    x_test = features.loc[test_mask]
    y_train = target[train_mask]
    y_validation = target[validation_mask]
    y_test = target[test_mask]
    x_train_augmented, y_train_augmented = augment_team_swap(x_train, y_train)
    x_validation_augmented, y_validation_augmented = augment_team_swap(
        x_validation,
        y_validation,
    )

    predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    prior = float(y_train.mean())
    predictions["Historical prior"] = np.full(len(y_test), prior)
    validation_predictions["Historical prior"] = np.full(len(y_validation), prior)
    predictions["Elo"] = elo_probability[test_mask]
    validation_predictions["Elo"] = elo_probability[validation_mask]

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("Training logistic regression")
    logistic_candidates: dict[str, float] = {}
    best_logistic: Pipeline | None = None
    best_logistic_loss = float("inf")
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(C=c_value, max_iter=2000, random_state=args.seed),
                ),
            ]
        )
        model.fit(x_train_augmented, y_train_augmented)
        probability = symmetrized_probability(model, x_validation)
        loss = float(log_loss(y_validation, probability))
        logistic_candidates[str(c_value)] = loss
        if loss < best_logistic_loss:
            best_logistic_loss = loss
            best_logistic = model
    if best_logistic is None:
        raise RuntimeError("Unable to fit logistic regression")
    validation_predictions["Logistic regression"] = symmetrized_probability(
        best_logistic, x_validation
    )
    predictions["Logistic regression"] = symmetrized_probability(best_logistic, x_test)

    print("Training histogram gradient boosting")
    histogram = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=350,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    histogram.fit(x_train_augmented, y_train_augmented)
    validation_predictions["Histogram gradient boosting"] = symmetrized_probability(
        histogram, x_validation
    )
    predictions["Histogram gradient boosting"] = symmetrized_probability(histogram, x_test)

    print("Training CatBoost")
    catboost = catboost_classifier(iterations=args.catboost_iterations, seed=args.seed)
    catboost.fit(
        x_train_augmented,
        y_train_augmented,
        eval_set=(x_validation_augmented, y_validation_augmented),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=100,
    )
    validation_predictions["CatBoost"] = symmetrized_probability(catboost, x_validation)
    predictions["CatBoost"] = symmetrized_probability(catboost, x_test)

    # Параметры калибровки оцениваются только на валидационном периоде.
    epsilon = 1e-6
    validation_logit = np.log(
        np.clip(validation_predictions["CatBoost"], epsilon, 1 - epsilon)
        / np.clip(1 - validation_predictions["CatBoost"], epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    test_logit = np.log(
        np.clip(predictions["CatBoost"], epsilon, 1 - epsilon)
        / np.clip(1 - predictions["CatBoost"], epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, random_state=args.seed)
    platt.fit(validation_logit, y_validation)
    predictions["CatBoost + Platt"] = platt.predict_proba(test_logit)[:, 1]

    metrics = {name: metric_row(y_test, probability) for name, probability in predictions.items()}
    metrics_frame = pd.DataFrame(
        [{"model": name, **metrics[name]} for name in MODEL_ORDER]
    )
    metrics_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)

    print("Calculating feature importance and segment metrics")
    importance = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": catboost.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    segments = segment_metrics(
        frame.loc[test_mask].reset_index(drop=True),
        y_test,
        predictions["CatBoost"],
    )
    pd.DataFrame(segments).to_csv(args.output_dir / "segment_metrics.csv", index=False)

    ablation_rows: list[dict[str, object]] = []
    if not args.skip_ablation:
        print("Running nested feature-family ablation")
        cumulative: list[str] = []
        for label, additions in ABLATION_GROUPS.items():
            cumulative.extend(additions)
            cumulative = list(dict.fromkeys(cumulative))
            model = catboost_classifier(iterations=700, seed=args.seed)
            model.fit(
                x_train_augmented[cumulative],
                y_train_augmented,
                eval_set=(x_validation_augmented[cumulative], y_validation_augmented),
                use_best_model=True,
                early_stopping_rounds=80,
                verbose=False,
            )
            probability = symmetrized_probability(model, x_test[cumulative])
            row: dict[str, object] = {
                "feature_set": label,
                "feature_count": len(cumulative),
            }
            row.update(metric_row(y_test, probability))
            ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(args.output_dir / "ablation_metrics.csv", index=False)

    prediction_output = frame.loc[
        test_mask,
        ["match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"],
    ].reset_index(drop=True)
    for name, probability in predictions.items():
        safe_name = (
            name.lower()
            .replace(" + ", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        prediction_output[f"prob_{safe_name}"] = probability
    prediction_output.to_csv(args.output_dir / "test_predictions.csv", index=False)

    catboost.save_model(str(args.output_dir / "catboost_model.cbm"))
    best_iteration = int(catboost.get_best_iteration())
    schema = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_name": "catboost_v3_symmetric",
        "target": "team1_win",
        "feature_order": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "feature_dtype": "float64",
        "missing_values": "native CatBoost NaN handling",
        "elo_k": best_k,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "seed": args.seed,
        "catboost_best_iteration": best_iteration,
        "team_swap_augmentation": True,
        "symmetric_inference": True,
    }
    write_json(args.output_dir / "feature_schema.json", schema)

    print("Building compact inference state")
    try:
        from src.inference_state import save_inference_state
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from inference_state import save_inference_state  # type: ignore[no-redef]
    save_inference_state(
        args.clean_dir,
        args.output_dir / "inference_state.json",
        elo_k=best_k,
    )

    split_counts = {
        "train": int(train_mask.sum()),
        "validation": int(validation_mask.sum()),
        "test": int(test_mask.sum()),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.features),
            "rows": len(frame),
            "columns": len(frame.columns),
            "date_min": frame["match_datetime_utc"].min(),
            "date_max": frame["match_datetime_utc"].max(),
            "target_rate_team1_win": float(target.mean()),
            "split_counts": split_counts,
            "feature_count": len(MODEL_FEATURES),
            "augmented_train_rows": len(x_train_augmented),
            "sha256": sha256_file(args.features),
        },
        "versions": library_versions(),
        "elo": {"best_k": best_k, "validation_candidates": elo_candidates},
        "logistic": {"validation_logloss_by_c": logistic_candidates},
        "catboost": {
            "best_iteration": best_iteration,
            "platt_intercept": float(platt.intercept_[0]),
            "platt_coefficient": float(platt.coef_[0, 0]),
        },
        "selected_model": "CatBoost",
        "selection_reason": (
            "The deployable CatBoost probability is averaged across both team perspectives; "
            "calibration is reported as an experiment because its parameters were fitted on "
            "the single validation window."
        ),
        "test_metrics": metrics,
        "catboost_bootstrap": bootstrap_interval(
            y_test,
            predictions["CatBoost"],
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "segment_metrics": segments,
        "ablation": ablation_rows,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    write_json(args.output_dir / "experiment_summary.json", summary)

    artifact_names = [
        "catboost_model.cbm",
        "feature_schema.json",
        "inference_state.json",
        "experiment_summary.json",
        "model_metrics.csv",
        "feature_importance.csv",
        "segment_metrics.csv",
        "ablation_metrics.csv",
        "test_predictions.csv",
    ]
    source_paths = [
        args.features,
        *(args.clean_dir / name for name in [
            "matches_final.csv",
            "match_lineups.csv",
            "veto_steps.csv",
            "match_maps.csv",
            "map_player_stats.csv",
        ]),
    ]
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        },
        "artifacts": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in artifact_names
        },
    }
    write_json(args.output_dir / "artifact_manifest.json", manifest)

    print(f"Artifacts saved to {args.output_dir}")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
