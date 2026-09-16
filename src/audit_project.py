"""Run reproducible integrity checks across data, features and model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.inference_state import INFERENCE_STATE_SCHEMA_VERSION
    from src.modeling import (
        MODEL_FEATURES,
        load_feature_dataset,
        load_schema,
        metric_row,
        sha256_file,
        swap_team_perspective,
        symmetrized_probability,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from inference_state import INFERENCE_STATE_SCHEMA_VERSION  # type: ignore[no-redef]
    from modeling import (  # type: ignore[no-redef]
        MODEL_FEATURES,
        load_feature_dataset,
        load_schema,
        metric_row,
        sha256_file,
        swap_team_perspective,
        symmetrized_probability,
        write_json,
    )


PREDICTION_COLUMNS = {
    "Historical prior": "prob_historical_prior",
    "Elo": "prob_elo",
    "Logistic regression": "prob_logistic_regression",
    "Histogram gradient boosting": "prob_histogram_gradient_boosting",
    "CatBoost": "prob_catboost",
    "CatBoost + Platt": "prob_catboost_platt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CS2 data and trained artifacts.")
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/interim/hltv_final_clean"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_dataset.csv"),
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audit_report.json"),
    )
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    matches_path = args.clean_dir / "matches_final.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Missing clean match table: {matches_path}")
    matches = pd.read_csv(matches_path, low_memory=False)
    required_match_columns = {
        "match_id",
        "match_datetime_utc",
        "team1_id",
        "team2_id",
        "team1_win",
    }
    missing_match_columns = sorted(required_match_columns - set(matches.columns))
    if missing_match_columns:
        failures.append(f"matches_final missing columns: {missing_match_columns}")
    else:
        match_time = pd.to_datetime(matches["match_datetime_utc"], errors="coerce", utc=True)
        duplicate_matches = int(matches["match_id"].duplicated().sum())
        invalid_targets = int((~matches["team1_win"].isin([0, 1])).sum())
        invalid_teams = int(
            (
                matches["team1_id"].isna()
                | matches["team2_id"].isna()
                | matches["team1_id"].eq(matches["team2_id"])
            ).sum()
        )
        invalid_times = int(match_time.isna().sum())
        checks["matches"] = {
            "rows": len(matches),
            "duplicate_match_ids": duplicate_matches,
            "invalid_targets": invalid_targets,
            "invalid_team_pairs": invalid_teams,
            "invalid_timestamps": invalid_times,
            "date_min": match_time.min(),
            "date_max": match_time.max(),
        }
        if duplicate_matches or invalid_targets or invalid_teams or invalid_times:
            failures.append("matches_final failed identifier, target, or timestamp checks")

    valid_team_pairs: set[tuple[int, int]] = set()
    if not missing_match_columns:
        for row in matches[["match_id", "team1_id", "team2_id"]].itertuples(index=False):
            if pd.notna(row.match_id) and pd.notna(row.team1_id) and pd.notna(row.team2_id):
                valid_team_pairs.add((int(row.match_id), int(row.team1_id)))
                valid_team_pairs.add((int(row.match_id), int(row.team2_id)))

    child_specs = {
        "match_lineups.csv": ["match_id", "team_id", "player_id"],
        "veto_steps.csv": ["match_id", "step_number", "team_name", "action", "map_name"],
        "match_maps.csv": ["match_id", "map_no", "map_name"],
        "map_player_stats.csv": ["match_id", "map_no", "team_id", "player_id"],
    }
    child_checks: dict[str, Any] = {}
    valid_match_ids = set(pd.to_numeric(matches.get("match_id"), errors="coerce").dropna().astype(int))
    for filename, key in child_specs.items():
        path = args.clean_dir / filename
        if not path.exists():
            failures.append(f"Missing clean child table: {path}")
            continue
        extra_columns = (
            {"played", "team1_map_score", "team2_map_score"}
            if filename == "match_maps.csv"
            else set()
        )
        selected_columns = set(key) | extra_columns
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in selected_columns,
            low_memory=False,
        )
        missing_key = sorted(set(key) - set(frame.columns))
        if missing_key:
            failures.append(f"{filename} missing key columns: {missing_key}")
            continue
        duplicate_keys = int(frame.duplicated(key).sum())
        child_ids = set(pd.to_numeric(frame["match_id"], errors="coerce").dropna().astype(int))
        orphan_matches = len(child_ids - valid_match_ids)
        invalid_team_links = 0
        if "team_id" in frame.columns and valid_team_pairs:
            ids = frame[["match_id", "team_id"]].apply(pd.to_numeric, errors="coerce")
            invalid_team_links = int(
                sum(
                    pd.isna(match_id)
                    or pd.isna(team_id)
                    or (int(match_id), int(team_id)) not in valid_team_pairs
                    for match_id, team_id in ids.itertuples(index=False, name=None)
                )
            )
        details: dict[str, Any] = {
            "rows": len(frame),
            "duplicate_keys": duplicate_keys,
            "orphan_match_ids": orphan_matches,
            "invalid_match_team_links": invalid_team_links,
        }
        if filename == "match_lineups.csv":
            lineup_size = frame.groupby(["match_id", "team_id"])["player_id"].nunique()
            details["lineup_groups"] = len(lineup_size)
            details["complete_five_player_lineups"] = int(lineup_size.eq(5).sum())
            details["incomplete_lineups"] = int(lineup_size.ne(5).sum())
        elif filename == "map_player_stats.csv":
            map_lineup_size = frame.groupby(["match_id", "map_no", "team_id"])[
                "player_id"
            ].nunique()
            details["map_team_groups"] = len(map_lineup_size)
            details["five_player_map_groups"] = int(map_lineup_size.eq(5).sum())
            details["nonstandard_map_groups"] = int(map_lineup_size.ne(5).sum())
        elif filename == "match_maps.csv":
            played = frame["played"].astype(str).str.lower().isin({"true", "1"})
            score1 = pd.to_numeric(frame["team1_map_score"], errors="coerce")
            score2 = pd.to_numeric(frame["team2_map_score"], errors="coerce")
            complete = played & score1.notna() & score2.notna()
            invalid_scores = complete & ((score1 < 0) | (score2 < 0) | score1.eq(score2))
            details["played_rows"] = int(played.sum())
            details["played_rows_without_score"] = int((played & ~complete).sum())
            details["invalid_completed_scores"] = int(invalid_scores.sum())
            if invalid_scores.any():
                failures.append("match_maps.csv contains negative or tied completed scores")
        child_checks[filename] = details
        if duplicate_keys or orphan_matches or invalid_team_links:
            failures.append(f"{filename} failed key or foreign-key checks")
    checks["child_tables"] = child_checks

    raw_feature_order = pd.read_csv(
        args.features,
        usecols=["match_id", "match_datetime_utc"],
        low_memory=False,
    )
    raw_feature_order["match_datetime_utc"] = pd.to_datetime(
        raw_feature_order["match_datetime_utc"], errors="coerce", utc=True
    )
    chronological_order = raw_feature_order.sort_values(
        ["match_datetime_utc", "match_id"], kind="mergesort"
    ).index.equals(raw_feature_order.index)
    features = load_feature_dataset(args.features)
    values = features[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    all_missing = [feature for feature in MODEL_FEATURES if values[feature].isna().all()]
    constant = [
        feature for feature in MODEL_FEATURES if values[feature].nunique(dropna=True) <= 1
    ]
    duplicate_feature_pairs: list[list[str]] = []
    for index, left in enumerate(MODEL_FEATURES):
        for right in MODEL_FEATURES[index + 1 :]:
            if values[left].equals(values[right]):
                duplicate_feature_pairs.append([left, right])
    checks["features"] = {
        "rows": len(features),
        "columns": len(features.columns),
        "model_feature_count": len(MODEL_FEATURES),
        "duplicate_match_ids": int(features["match_id"].duplicated().sum()),
        "infinite_values": int(np.isinf(features[MODEL_FEATURES].to_numpy(dtype=float)).sum()),
        "all_missing_features": all_missing,
        "constant_features": constant,
        "exact_duplicate_feature_pairs": duplicate_feature_pairs,
        "stored_in_chronological_order": chronological_order,
        "minimum_feature_coverage": float(values.notna().mean(axis=1).min()),
        "mean_feature_coverage": float(values.notna().mean(axis=1).mean()),
        "sha256": sha256_file(args.features),
    }
    if len(features) != len(matches):
        failures.append("Feature and clean-match row counts differ")
    if all_missing or constant or duplicate_feature_pairs:
        failures.append(
            "Feature allow-list contains an all-missing, constant, or exact duplicate feature"
        )
    if not chronological_order:
        failures.append("Feature dataset is not stored in chronological order")

    schema = load_schema(args.artifacts)
    from catboost import CatBoostClassifier

    model_path = args.artifacts / "catboost_model.cbm"
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    model_names_match = model.feature_names_ == schema["feature_order"]
    symmetry_sample = values.head(min(512, len(values)))
    symmetry_probability = symmetrized_probability(model, symmetry_sample)
    reversed_probability = symmetrized_probability(
        model,
        swap_team_perspective(symmetry_sample),
    )
    maximum_symmetry_error = float(
        np.max(np.abs(symmetry_probability + reversed_probability - 1.0))
    )
    checks["model"] = {
        "name": schema.get("model_name"),
        "schema_version": schema.get("schema_version"),
        "tree_count": model.tree_count_,
        "feature_names_match": model_names_match,
        "symmetric_inference": schema.get("symmetric_inference") is True,
        "maximum_team_swap_symmetry_error": maximum_symmetry_error,
    }
    if (
        not model_names_match
        or schema.get("symmetric_inference") is not True
        or maximum_symmetry_error > 1e-12
    ):
        failures.append("Saved model is inconsistent with its schema")

    state_path = args.artifacts / "inference_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    team_states = state.get("teams", {})
    invalid_state_teams = 0
    total_state_matches = 0
    total_state_wins = 0
    for team in team_states.values():
        team_matches = int(team.get("matches", -1))
        team_wins = int(team.get("wins", -1))
        total_state_matches += team_matches
        total_state_wins += team_wins
        if team_matches < 0 or team_wins < 0 or team_wins > team_matches:
            invalid_state_teams += 1
    state_time = pd.Timestamp(state.get("state_as_of_utc"))
    expected_state_time = pd.to_datetime(
        matches["match_datetime_utc"], errors="coerce", utc=True
    ).max()
    state_time_matches = state_time == expected_state_time
    history_totals_match = (
        total_state_matches == 2 * len(matches) and total_state_wins == len(matches)
    )
    checks["inference_state"] = {
        "schema_version": state.get("schema_version"),
        "expected_schema_version": INFERENCE_STATE_SCHEMA_VERSION,
        "state_as_of_utc": state.get("state_as_of_utc"),
        "teams": len(team_states),
        "players": len(state.get("players", {})),
        "maps": len(state.get("maps", {})),
        "veto": len(state.get("veto", {})),
        "state_time_matches_latest_clean_match": state_time_matches,
        "team_match_appearances": total_state_matches,
        "team_wins": total_state_wins,
        "history_totals_match_clean_data": history_totals_match,
        "invalid_team_records": invalid_state_teams,
    }
    if state.get("schema_version") != INFERENCE_STATE_SCHEMA_VERSION:
        failures.append("Inference state schema is stale")
    if not state_time_matches or not history_totals_match or invalid_state_teams:
        failures.append("Inference state is inconsistent with clean match history")

    predictions = pd.read_csv(args.artifacts / "test_predictions.csv")
    stored_metrics = pd.read_csv(args.artifacts / "model_metrics.csv").set_index("model")
    target = predictions["team1_win"].astype(int).to_numpy()
    largest_metric_difference = 0.0
    for model_name, column in PREDICTION_COLUMNS.items():
        if column not in predictions or model_name not in stored_metrics.index:
            failures.append(f"Missing saved predictions or metrics for {model_name}")
            continue
        probability = predictions[column].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            failures.append(f"Invalid saved probabilities for {model_name}")
            continue
        for metric, value in metric_row(target, probability).items():
            difference = abs(float(stored_metrics.loc[model_name, metric]) - value)
            largest_metric_difference = max(largest_metric_difference, difference)
    checks["predictions"] = {
        "rows": len(predictions),
        "largest_metric_difference": largest_metric_difference,
        "tolerance": args.metric_tolerance,
    }
    if largest_metric_difference > args.metric_tolerance:
        failures.append("Stored metrics do not reproduce from test predictions")

    supplemental_paths = {
        "walk_forward_metrics": args.artifacts / "walk_forward_metrics.csv",
        "walk_forward_predictions": args.artifacts / "walk_forward_predictions.csv",
        "walk_forward_summary": args.artifacts / "walk_forward_summary.json",
        "drift_report": args.artifacts / "drift_report.json",
    }
    missing_supplemental = [
        name for name, path in supplemental_paths.items() if not path.exists()
    ]
    checks["supplemental_artifacts"] = {"missing": missing_supplemental}
    if missing_supplemental:
        failures.append(f"Missing supplemental audit artifacts: {missing_supplemental}")
    else:
        walk_predictions = pd.read_csv(supplemental_paths["walk_forward_predictions"])
        walk_metrics = pd.read_csv(supplemental_paths["walk_forward_metrics"])
        required_walk_columns = {
            "fold",
            "match_id",
            "team1_win",
            "prob_elo",
            "prob_catboost",
        }
        missing_walk_columns = sorted(required_walk_columns - set(walk_predictions.columns))
        walk_metric_difference = 0.0
        invalid_walk_probability = False
        if missing_walk_columns:
            failures.append(
                f"walk_forward_predictions.csv missing columns: {missing_walk_columns}"
            )
        else:
            for model_name, column in [("Elo", "prob_elo"), ("CatBoost", "prob_catboost")]:
                probability = pd.to_numeric(walk_predictions[column], errors="coerce").to_numpy()
                invalid_walk_probability |= bool(
                    (~np.isfinite(probability) | (probability < 0) | (probability > 1)).any()
                )
                aggregate_rows = walk_metrics[
                    (walk_metrics["scope"] == "aggregate")
                    & (walk_metrics["model"] == model_name)
                ]
                if len(aggregate_rows) != 1:
                    failures.append(f"Missing unique walk-forward aggregate for {model_name}")
                    continue
                reproduced = metric_row(
                    walk_predictions["team1_win"].astype(int).to_numpy(), probability
                )
                for metric, value in reproduced.items():
                    difference = abs(float(aggregate_rows.iloc[0][metric]) - value)
                    walk_metric_difference = max(walk_metric_difference, difference)
        if invalid_walk_probability:
            failures.append("Walk-forward predictions contain invalid probabilities")
        if walk_predictions["match_id"].duplicated().any():
            failures.append("Walk-forward predictions contain duplicate matches")
        if walk_metric_difference > args.metric_tolerance:
            failures.append("Walk-forward aggregate metrics do not reproduce")

        walk_summary = json.loads(
            supplemental_paths["walk_forward_summary"].read_text(encoding="utf-8")
        )
        drift_report = json.loads(
            supplemental_paths["drift_report"].read_text(encoding="utf-8")
        )
        drift_features = drift_report.get("features", [])
        drift_names = [row.get("feature") for row in drift_features]
        if drift_names and set(drift_names) != set(MODEL_FEATURES):
            failures.append("Drift report does not cover the exact model feature allow-list")
        if len(drift_names) != len(MODEL_FEATURES):
            failures.append("Drift report has an unexpected feature count")
        checks["supplemental_artifacts"].update(
            {
                "walk_forward_rows": len(walk_predictions),
                "walk_forward_folds": sorted(walk_predictions.get("fold", pd.Series()).unique()),
                "walk_forward_largest_metric_difference": walk_metric_difference,
                "walk_forward_summary_folds": len(walk_summary.get("folds", [])),
                "drift_status": drift_report.get("status"),
                "drift_feature_count": len(drift_names),
            }
        )

    manifest_path = args.artifacts / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches: list[str] = []
        for name, metadata in manifest.get("artifacts", {}).items():
            path = args.artifacts / name
            if not path.exists() or sha256_file(path) != metadata.get("sha256"):
                mismatches.append(name)
        for name, metadata in manifest.get("sources", {}).items():
            path = Path(name)
            if not path.exists() or sha256_file(path) != metadata.get("sha256"):
                mismatches.append(name)
        checks["manifest"] = {
            "entries": len(manifest.get("artifacts", {})) + len(manifest.get("sources", {})),
            "mismatches": mismatches,
        }
        if mismatches:
            failures.append("Artifact manifest contains checksum mismatches")
    else:
        warnings.append("artifact_manifest.json is missing")

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    write_json(args.output, report)
    print(f"Audit status: {report['status']}")
    print(f"Report: {args.output}")
    if report["failures"]:
        for failure in report["failures"]:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
