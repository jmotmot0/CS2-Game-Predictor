"""Общие функции обучения моделей, оценки качества и получения прогноза."""

from __future__ import annotations

import json
import hashlib
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEED = 42
TRAIN_END = "2025-07-01"
VALIDATION_END = "2026-01-01"
MODEL_SCHEMA_VERSION = 3

# Явный список разрешённых признаков: остальные поля не передаются модели.
MODEL_FEATURES = [
    "bo1",
    "bo3",
    "bo5",
    "is_lan",
    "is_online",
    "rank_available",
    "diff_elo_pre",
    "diff_matches_before",
    "diff_days_since_last_match",
    "diff_activity_7d",
    "diff_activity_30d",
    "diff_activity_90d",
    "diff_overall_winrate",
    "diff_winrate_last_5",
    "diff_winrate_last_10",
    "diff_winrate_last_20",
    "diff_win_streak",
    "diff_loss_streak",
    "diff_avg_opp_elo_last_10",
    "diff_h2h_wins_all",
    "diff_h2h_wins_last5",
    "diff_roster_size",
    "diff_roster_overlap_prev",
    "diff_roster_overlap_prev_ratio",
    "diff_lineup_history_coverage",
    "diff_lineup_players_with_history",
    "diff_lineup_player_rating_mean",
    "diff_lineup_player_adr_mean",
    "diff_lineup_player_kast_mean",
    "diff_lineup_player_opening_diff_mean",
    "diff_lineup_player_maps_played_mean",
    "diff_avg_map_count_before",
    "diff_avg_map_wr_before",
    "diff_avg_map_ct_wr_before",
    "diff_avg_map_t_wr_before",
    "diff_series_maps_known",
    "diff_veto_pick_rate_before",
    "diff_veto_remove_rate_before",
    "diff_veto_leftover_rate_before",
    "diff_rank",
]

ABLATION_GROUPS = {
    "Rank + context": [
        "diff_rank",
        "bo1",
        "bo3",
        "bo5",
        "is_lan",
        "is_online",
        "rank_available",
    ],
    "+ Elo and form": [
        "diff_elo_pre",
        "diff_matches_before",
        "diff_overall_winrate",
        "diff_winrate_last_5",
        "diff_winrate_last_10",
        "diff_winrate_last_20",
        "diff_activity_7d",
        "diff_activity_30d",
        "diff_activity_90d",
        "diff_days_since_last_match",
        "diff_win_streak",
        "diff_loss_streak",
        "diff_avg_opp_elo_last_10",
        "diff_h2h_wins_all",
        "diff_h2h_wins_last5",
    ],
    "+ Roster and players": [
        "diff_roster_size",
        "diff_roster_overlap_prev",
        "diff_roster_overlap_prev_ratio",
        "diff_lineup_history_coverage",
        "diff_lineup_players_with_history",
        "diff_lineup_player_rating_mean",
        "diff_lineup_player_adr_mean",
        "diff_lineup_player_kast_mean",
        "diff_lineup_player_opening_diff_mean",
        "diff_lineup_player_maps_played_mean",
    ],
    "+ Map pool and veto": [
        "diff_avg_map_wr_before",
        "diff_avg_map_ct_wr_before",
        "diff_avg_map_t_wr_before",
        "diff_avg_map_count_before",
        "diff_series_maps_known",
        "diff_veto_pick_rate_before",
        "diff_veto_remove_rate_before",
        "diff_veto_leftover_rate_before",
    ],
}


def require_model_features(frame: pd.DataFrame, feature_order: Iterable[str] = MODEL_FEATURES) -> None:
    missing = [feature for feature in feature_order if feature not in frame.columns]
    if missing:
        raise ValueError(f"Feature dataset is incompatible; missing columns: {missing}")


def swap_team_perspective(
    frame: pd.DataFrame,
    feature_order: Iterable[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    """Mirror model inputs so team 1 and team 2 exchange places.

    The allow-list contains only team-invariant context columns and signed
    ``diff_*`` columns. Mirroring therefore consists of negating every signed
    difference while leaving BO/location/availability indicators unchanged.
    """

    features = list(feature_order)
    require_model_features(frame, features)
    swapped = frame[features].copy()
    difference_columns = [feature for feature in features if feature.startswith("diff_")]
    swapped[difference_columns] = -swapped[difference_columns]
    return swapped


def augment_team_swap(
    features: pd.DataFrame,
    target: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add a mirrored observation with the inverted target for every match."""

    y = np.asarray(target, dtype=int)
    if len(features) != len(y):
        raise ValueError("Features and target must contain the same number of rows")
    mirrored = swap_team_perspective(features, features.columns)
    augmented_features = pd.concat([features, mirrored], ignore_index=True)
    augmented_target = np.concatenate([y, 1 - y])
    return augmented_features, augmented_target


def symmetrized_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict with the exact identity P(A, B) = 1 - P(B, A)."""

    forward = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    mirrored = swap_team_perspective(features, features.columns)
    reverse = np.asarray(model.predict_proba(mirrored)[:, 1], dtype=float)
    return (forward + (1.0 - reverse)) / 2.0


def load_feature_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    require_model_features(frame)
    frame["match_datetime_utc"] = pd.to_datetime(
        frame["match_datetime_utc"], errors="coerce", utc=True
    )
    frame["team1_win"] = pd.to_numeric(frame["team1_win"], errors="coerce")
    frame = frame.dropna(
        subset=["match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"]
    ).copy()
    frame = frame[frame["team1_win"].isin([0, 1])]
    if frame["match_id"].duplicated().any():
        raise ValueError("Feature dataset contains duplicate match_id values")
    return frame.sort_values(["match_datetime_utc", "match_id"]).reset_index(drop=True)


def chronological_masks(
    frame: pd.DataFrame,
    *,
    train_end: str = TRAIN_END,
    validation_end: str = VALIDATION_END,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_boundary = pd.Timestamp(train_end, tz="UTC")
    validation_boundary = pd.Timestamp(validation_end, tz="UTC")
    if validation_boundary <= train_boundary:
        raise ValueError("validation_end must be later than train_end")
    time = frame["match_datetime_utc"]
    train = (time < train_boundary).to_numpy()
    validation = ((time >= train_boundary) & (time < validation_boundary)).to_numpy()
    test = (time >= validation_boundary).to_numpy()
    counts = {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())}
    if min(counts.values()) == 0:
        raise ValueError(f"Chronological split contains an empty part: {counts}")
    return train, validation, test


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def metric_row(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece_10": expected_calibration_error(y_true, probability, bins=10),
    }


def elo_probabilities(
    frame: pd.DataFrame,
    *,
    k: float,
    base: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate simultaneous two-team Elo updates in chronological order."""

    ratings: dict[int, float] = defaultdict(lambda: float(base))
    team1_ids = frame["team1_id"].astype(int).to_numpy()
    team2_ids = frame["team2_id"].astype(int).to_numpy()
    outcomes = frame["team1_win"].astype(int).to_numpy()
    timestamps = frame["match_datetime_utc"].astype("int64").to_numpy()
    probabilities = np.empty(len(frame), dtype=float)
    team1_pre = np.empty(len(frame), dtype=float)
    team2_pre = np.empty(len(frame), dtype=float)

    start = 0
    while start < len(frame):
        stop = start + 1
        while stop < len(frame) and timestamps[stop] == timestamps[start]:
            stop += 1
        deltas: dict[int, float] = defaultdict(float)
        for index in range(start, stop):
            team1_id = int(team1_ids[index])
            team2_id = int(team2_ids[index])
            rating1 = ratings[team1_id]
            rating2 = ratings[team2_id]
            probability = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
            probabilities[index] = probability
            team1_pre[index] = rating1
            team2_pre[index] = rating2
            deltas[team1_id] += k * (int(outcomes[index]) - probability)
            deltas[team2_id] += k * ((1 - int(outcomes[index])) - (1 - probability))
        for team_id, delta in deltas.items():
            ratings[team_id] += delta
        start = stop
    return probabilities, team1_pre, team2_pre


def tune_elo(
    frame: pd.DataFrame,
    validation_mask: np.ndarray,
    *,
    candidates: Iterable[float] = (8, 16, 24, 32, 40, 48, 64),
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    target = frame["team1_win"].astype(int).to_numpy()
    best_k: float | None = None
    best_loss = math.inf
    best_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    metrics: dict[str, dict[str, float]] = {}
    for value in candidates:
        probability, team1_pre, team2_pre = elo_probabilities(frame, k=float(value))
        validation_metrics = metric_row(target[validation_mask], probability[validation_mask])
        metrics[str(int(value))] = validation_metrics
        if validation_metrics["log_loss"] < best_loss:
            best_loss = validation_metrics["log_loss"]
            best_k = float(value)
            best_values = (probability, team1_pre, team2_pre)
    if best_k is None or best_values is None:
        raise RuntimeError("Unable to select Elo K")
    return best_k, *best_values, metrics


def catboost_classifier(*, iterations: int = 1000, seed: int = SEED) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=5.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def segment_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    segments = {
        "BO1": frame["bo1"].eq(1).to_numpy(),
        "BO3": frame["bo3"].eq(1).to_numpy(),
        "LAN": frame["is_lan"].eq(1).to_numpy(),
        "Online": frame["is_online"].eq(1).to_numpy(),
        "Both ranks available": frame["rank_available"].eq(1).to_numpy(),
        "At least one rank missing": frame["rank_available"].eq(0).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in segments.items():
        if int(mask.sum()) < 30 or len(np.unique(target[mask])) < 2:
            continue
        row: dict[str, Any] = {"segment": name, "n": int(mask.sum())}
        row.update(metric_row(target[mask], probability[mask]))
        rows.append(row)
    return rows


def bootstrap_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    samples: int = 500,
    seed: int = SEED,
) -> dict[str, list[float]]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    if samples <= 0:
        raise ValueError("samples must be positive")
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Bootstrap inputs must be non-empty and have equal length")
    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    auc_values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        sample_p = probability[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        accuracies.append(float(accuracy_score(sample_y, sample_p >= 0.5)))
        auc_values.append(float(roc_auc_score(sample_y, sample_p)))
    return {
        "accuracy_95_ci": [float(np.quantile(accuracies, 0.025)), float(np.quantile(accuracies, 0.975))],
        "roc_auc_95_ci": [float(np.quantile(auc_values, 0.025)), float(np.quantile(auc_values, 0.975))],
    }


def feature_coverage(row: pd.Series, feature_order: Iterable[str] = MODEL_FEATURES) -> float:
    features = list(feature_order)
    return float(row[features].notna().mean()) if features else 0.0


def library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name in ["sklearn", "catboost", "matplotlib"]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not-installed"
    return versions


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def update_artifact_manifest(output_dir: Path, artifact_names: Iterable[str]) -> None:
    """Add or refresh generated files in the existing artifact manifest."""

    manifest_path = output_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Artifact manifest not found: {manifest_path}; run training first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for name in artifact_names:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Cannot add missing artifact to manifest: {path}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(manifest_path, manifest)


def load_schema(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature schema not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    feature_order = schema.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError("feature_schema.json does not contain a valid feature_order")
    if schema.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Feature schema is incompatible with this code; retrain the model artifacts."
        )
    if feature_order != MODEL_FEATURES:
        raise ValueError("Feature schema order does not match the current model allow-list")
    if len(feature_order) != len(set(feature_order)):
        raise ValueError("feature_schema.json contains duplicate feature names")
    if schema.get("feature_count") != len(feature_order):
        raise ValueError("feature_schema.json contains an inconsistent feature_count")
    return schema
