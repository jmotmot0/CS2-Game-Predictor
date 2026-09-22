"""Прогноз сохранённого ранкера и пересборка признаков нового матча.

С --match-id используются сохранённые предматчевые признаки.
С --team1-id/--team2-id/--match-time признаки пересчитываются по истории
и заданным текущим составам, рейтингам и выбору карт.
Оба режима работают без сбора страниц и повторного обучения.

Пример: python -m src.predict_team_ranking --match-id 2389256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.modeling import TRAIN_END, VALIDATION_END, sha256_file
from src.team_ranking import (
    EXTENDED_TEAM_FEATURES, EXTENDED_TEAM_FEATURE_GROUPS,
    TEAM_MODEL_FEATURES, TEAM_FEATURE_GROUPS, team_ranking_scores, team_rows,
)

DEFAULT_MODEL_DIR = Path("artifacts/supervisor_revision_2026-09-30")
DEFAULT_CLEAN_DIR = Path("data/interim/hltv_final_clean")


def verified_bundle(model_dir: Path, data_path: Path | None = None, *, feature_count=46):
    """Проверить сохранённые файлы, схему и порядок признаков CatBoost."""
    from catboost import CatBoostRanker

    root = Path(model_dir).resolve()
    manifest_path = root / "results_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The results manifest must contain nonempty file hashes")
    for relative, digest in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or Path(relative).is_absolute():
            raise ValueError(f"Manifest path escapes the model directory: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid SHA-256 digest in manifest: {relative}")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen research file failed its integrity check: {relative}")
    key = f"ranking_{feature_count}_s42"
    required = [f"{key}.cbm", f"{key}.json", "extended_features.csv", "protocol.json"]
    if any(name not in files for name in required):
        raise ValueError(f"The frozen manifest must include model, schema, data and protocol: {required}")
    chosen_data = Path(data_path).resolve() if data_path is not None else root / "extended_features.csv"
    if not chosen_data.is_file() or sha256_file(chosen_data) != files["extended_features.csv"]:
        raise ValueError("The supplied data must match the frozen extended_features.csv SHA-256")
    schema = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
    expected = EXTENDED_TEAM_FEATURES if feature_count == 46 else TEAM_MODEL_FEATURES
    if schema.get("features") != expected or schema.get("feature_count") != feature_count:
        raise ValueError("The model schema does not match the canonical ordered own-team feature list")
    if schema.get("params", {}).get("loss_function") != "PairLogit":
        raise ValueError("The research model must be trained with PairLogit")
    model = CatBoostRanker()
    model.load_model(str(root / f"{key}.cbm"))
    if model.feature_names_ != expected:
        raise ValueError("The model weights and stored feature schema disagree")
    return model, chosen_data, {
        "model_key": key,
        "verified_files": len(files),
        "model_sha256": files[f"{key}.cbm"],
        "schema_sha256": files[f"{key}.json"],
        "data_sha256": files["extended_features.csv"],
        "manifest_sha256": sha256_file(manifest_path),
        "protocol": json.loads((root / "protocol.json").read_text(encoding="utf-8")),
    }


def historical_prediction(match_id: int, model_dir=DEFAULT_MODEL_DIR, data_path=None, *, feature_count=46):
    """Получить прогноз по предматчевой строке, не передавая модели её исход."""
    if feature_count not in (40, 46):
        raise ValueError("Supported own-team research feature counts are 40 and 46")
    if match_id <= 0:
        raise ValueError("The historical match ID must be positive")
    model, chosen_data, metadata = verified_bundle(Path(model_dir), data_path, feature_count=feature_count)
    data = pd.read_csv(chosen_data)
    required = {"match_id", "match_datetime_utc", "team1_id", "team2_id"}
    if not required.issubset(data.columns):
        raise ValueError(f"Historical metadata is incomplete: {sorted(required - set(data.columns))}")
    row = data.loc[data.match_id.eq(match_id)]
    if len(row) != 1:
        raise ValueError(f"Expected one historical row for match {match_id}; found {len(row)}")
    timestamp = pd.to_datetime(row.iloc[0].match_datetime_utc, utc=True, errors="raise")
    if pd.isna(timestamp):
        raise ValueError("The historical match timestamp is unavailable")
    columns = EXTENDED_TEAM_FEATURES if feature_count == 46 else TEAM_MODEL_FEATURES
    scores = team_ranking_scores(model, row, columns)[0]
    ids = [int(row.iloc[0][f"team{number}_id"]) for number in (1, 2)]
    if ids[0] == ids[1]:
        raise ValueError("A historical match must have two distinct teams")
    exact_tie = bool(scores[0] == scores[1])
    chosen = (0 if ids[0] < ids[1] else 1) if exact_tie else int(scores[1] > scores[0])
    probability = float(expit(scores[0] - scores[1]))
    teams = []
    for index, number in enumerate((1, 2)):
        value = row.iloc[0].get(f"team{number}", ids[index])
        teams.append({
            "id": ids[index], "name": str(value), "score": float(scores[index]),
            "win_probability": probability if index == 0 else 1 - probability,
        })
    definitions = EXTENDED_TEAM_FEATURE_GROUPS if feature_count == 46 else TEAM_FEATURE_GROUPS
    train_end = pd.to_datetime(metadata["protocol"].get("training_end_exclusive", TRAIN_END), utc=True)
    validation_end = pd.to_datetime(metadata["protocol"].get("validation_end_exclusive", VALIDATION_END), utc=True)
    evaluation_role = ("in_sample_training" if timestamp < train_end else
                       "validation_used_for_checkpoint_selection" if timestamp < validation_end else
                       "retrospective_test_or_later")
    return {
        "mode": "historical_offline_reproduction",
        "evaluation_role": evaluation_role,
        "match_id": int(match_id), "match_datetime_utc": timestamp.isoformat(),
        "prediction_time": metadata["protocol"].get("prediction_time", "After pre-match veto, before first map"),
        "model": metadata["model_key"], "loss_function": "PairLogit",
        "feature_count_per_team": feature_count,
        "teams": teams, "predicted_winner": teams[chosen]["name"],
        "predicted_winner_id": teams[chosen]["id"], "exact_score_tie": exact_tie,
        "decision_rule": "Higher score; exact tie resolved by lower stable team ID",
        "probability_rule": "sigmoid(score_A - score_B); no post-hoc calibration",
        "own_feature_groups": {
            group: {"label": item["label"], "count": len(item["features"]), "features": list(item["features"])}
            for group, item in definitions.items()
        },
        "integrity": {key: value for key, value in metadata.items() if key.endswith("sha256")},
        "note": "Historical pre-match features are reused. This command does not collect current data or validate a new live forecast.",
    }


def prepare_forecast_features(*, team1_id: int, team2_id: int, match_time,
                              clean_dir=DEFAULT_CLEAN_DIR, protocol=None, bo=3,
                              location="online", team1_rank=None, team2_rank=None,
                              team1_players=None, team2_players=None, maps=None,
                              team1_picks=None, team2_picks=None, team1_removes=None,
                              team2_removes=None, decider=None):
    """Пересчитать признаки команд до начала матча с ещё неизвестным исходом.

    В историю входят только матчи раньше указанного времени начала.
    Текущие составы, рейтинг и выбор карт не восстанавливаются из будущих строк.
    Результат и статистика игроков прогнозируемой серии отсутствуют в таблицах.
    """
    from src.feature_engineering import (
        MAP_COLUMNS, PLAYER_COLUMNS, ROSTER_COLUMNS, VETO_COLUMNS,
        build_feature_dataset, normalize_lineups, normalize_matches,
        normalize_player_stats, read_clean_tables,
    )
    from src.predict_match import PENDING_MATCH_ID, append_lineup_rows, append_veto_rows
    from src.research_team_features import build_research_team_features
    from src.team_ranking import EXTRA_INDIVIDUAL_FEATURES, EXTRA_JOINT_EXPERIENCE_FEATURES

    protocol = protocol or {}
    if team1_id <= 0 or team2_id <= 0 or team1_id == team2_id:
        raise ValueError("Two distinct positive team identifiers are required")
    if bo not in (1, 3, 5) or location not in ("lan", "online"):
        raise ValueError("BO must be 1/3/5 and location must be lan/online")
    timestamp = pd.Timestamp(match_time)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("match-time must include an explicit UTC offset or Z suffix")
    timestamp = timestamp.tz_convert("UTC")
    train_end = pd.Timestamp(protocol.get("training_end_exclusive", TRAIN_END))
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    selection_end = pd.Timestamp(protocol.get("validation_end_exclusive", VALIDATION_END))
    selection_end = selection_end.tz_localize("UTC") if selection_end.tzinfo is None else selection_end.tz_convert("UTC")
    if timestamp < train_end:
        raise ValueError("Requested time predates the model's training cutoff; use --match-id for historical reproduction")
    if timestamp < selection_end:
        raise ValueError("Requested time predates the model-selection cutoff: its checkpoint used the later validation period; use --match-id for retrospective reproduction")
    rosters = [list(team1_players or []), list(team2_players or [])]
    for roster in rosters:
        if roster and (len(roster) != 5 or len(set(roster)) != 5 or min(roster) <= 0):
            raise ValueError("Supply exactly five distinct positive player IDs per known lineup")
    if set(rosters[0]) & set(rosters[1]):
        raise ValueError("The two current lineups cannot share a player")
    for rank in (team1_rank, team2_rank):
        if rank is not None and (not np.isfinite(rank) or rank < 1 or not float(rank).is_integer()):
            raise ValueError("Current HLTV ranks must be positive integers when supplied")
    map_names = list(maps or [])
    selections = [list(team1_picks or []), list(team2_picks or []), list(decider or [])]
    removals = [list(team1_removes or []), list(team2_removes or [])]
    if not map_names:
        map_names = [name for values in selections for name in values]
    if map_names and (len(map_names) != bo or len(set(map_names)) != len(map_names)):
        raise ValueError("The supplied pre-match map pool must contain exactly BO distinct maps, including the potential decider")
    selected = [name for values in selections for name in values]
    removed = [name for values in removals for name in values]
    if len(selected) != len(set(selected)) or len(removed) != len(set(removed)):
        raise ValueError("Veto selections/removals must not duplicate maps")
    if not set(selected).issubset(map_names) or set(removed) & set(map_names):
        raise ValueError("Selected maps must belong to the pool and removed maps must not")
    warnings = [
        "Historical outcomes are ordered by match start, not confirmed finish/publication time; end timestamps were not archived.",
        "Current lineups, HLTV ranks and veto are supplied by the caller; this command does not obtain live inputs.",
    ]
    expected = {
        Path(name).name: digest for name, digest in protocol.get("protected_hashes", {}).items()
        if "hltv_final_clean" in name.replace("\\", "/")
    }
    clean_hashes = {}
    for filename in ("matches_final.csv", "match_lineups.csv", "veto_steps.csv", "match_maps.csv", "map_player_stats.csv"):
        path = Path(clean_dir) / filename
        clean_hashes[filename] = sha256_file(path)
        if expected and clean_hashes[filename] != expected.get(filename):
            raise ValueError(f"Clean table differs from the frozen research source: {filename}")
    if not expected:
        warnings.append("The bundle has no frozen clean-table hashes; input hashes are reported but not checked against training sources.")
    matches, lineups, veto, match_maps, stats = read_clean_tables(Path(clean_dir))
    normalized = normalize_matches(matches)
    history = normalized.loc[normalized.match_datetime_utc.lt(timestamp)].copy()
    if history.empty:
        raise ValueError("No completed historical matches exist before the requested time")
    latest = history.match_datetime_utc.max()
    staleness_days = (timestamp - latest).total_seconds() / 86400
    if staleness_days > 7:
        warnings.append(f"Collected history ends {staleness_days:.1f} days before the requested time; uncollected newer matches are absent.")
    pending = {
        "match_id": PENDING_MATCH_ID, "match_date": timestamp.date().isoformat(),
        "match_datetime_utc": timestamp.isoformat(), "team1_id": team1_id, "team2_id": team2_id,
        "team1": f"team_{team1_id}", "team2": f"team_{team2_id}",
        "team1_rank": np.nan if team1_rank is None else float(team1_rank),
        "team2_rank": np.nan if team2_rank is None else float(team2_rank),
        "team1_win": np.nan, "team1_score": np.nan, "team2_score": np.nan,
        "is_valid_result": 0, "bo": f"bo{bo}", "lan_online": location,
    }
    combined = pd.concat([history, pd.DataFrame([pending])], ignore_index=True, sort=False)
    known_ids = set(history.match_id.astype(int))
    tables = [lineups, veto, match_maps, stats]
    lineups, veto, match_maps, stats = [
        table.loc[pd.to_numeric(table.match_id, errors="coerce").isin(known_ids)].copy()
        for table in tables
    ]
    for number, (team_id, roster) in enumerate(zip((team1_id, team2_id), rosters), start=1):
        lineups = append_lineup_rows(lineups, team_id=team_id, team_ordinal=number, players=roster)
    veto = append_veto_rows(
        veto, team1_id=team1_id, team2_id=team2_id,
        team1_picks=selections[0], team2_picks=selections[1],
        team1_removes=removals[0], team2_removes=removals[1], deciders=selections[2],
    )
    features = build_feature_dataset(
        combined, lineups, veto, match_maps, stats,
        elo_k=float(protocol.get("elo_k", 48.)), include_pending=True,
        map_pool_overrides={PENDING_MATCH_ID: map_names} if map_names else None,
    )
    valid_ids = set(features.match_id.astype(int))
    extra = build_research_team_features(
        features, normalize_lineups(lineups, valid_ids), normalize_player_stats(stats, features),
    )
    features = features.merge(extra, on="match_id", how="left", validate="one_to_one")
    row = features.loc[features.match_id.eq(PENDING_MATCH_ID)].copy()
    if len(row) != 1:
        raise RuntimeError("Exactly one unlabelled pending series must be constructed")
    for number, roster in enumerate(rosters, start=1):
        if int(row.iloc[0][f"team{number}_matches_before"]) == 0:
            warnings.append(f"Team {number} has no earlier series in the collected history; history-based strength estimates are poorly supported.")
        if not roster:
            fields = ROSTER_COLUMNS + PLAYER_COLUMNS + list(EXTRA_INDIVIDUAL_FEATURES) + list(EXTRA_JOINT_EXPERIENCE_FEATURES)
            row[[f"team{number}_{name}" for name in fields]] = np.nan
            warnings.append(f"Current lineup for team {number} was not supplied; individual and joint-experience measurements are unavailable.")
        elif float(row.iloc[0][f"team{number}_lineup_history_coverage"]) < 1:
            warnings.append(f"Team {number} has players with fewer than five known earlier maps; distributional Rating features are unavailable.")
    if not map_names:
        fields = [f"team{number}_{name}" for number in (1, 2) for name in MAP_COLUMNS + VETO_COLUMNS]
        row[fields] = np.nan
        warnings.append("Pre-match map pool was not supplied; map and veto-history measurements are unavailable.")
    elif not any(selections) and not any(removals):
        warnings.append("Detailed veto actions were not supplied; veto-history measurements are unavailable.")
    elif set(selected) != set(map_names):
        warnings.append("Veto roles are only partially supplied; some pick/decider-history measurements are unavailable.")
    if team1_rank is None or team2_rank is None:
        warnings.append("At least one current HLTV rank was not supplied; no historical rank is substituted.")
    for filename, digest in clean_hashes.items():
        if sha256_file(Path(clean_dir) / filename) != digest:
            raise ValueError(f"Clean table changed while the snapshot was being built: {filename}")
    return row, {
        "match_datetime_utc": timestamp.isoformat(), "history_latest_start_utc": latest.isoformat(),
        "model_selection_available_from_utc": selection_end.isoformat(),
        "history_series_count": len(history), "history_staleness_days": staleness_days,
        "clean_table_sha256": clean_hashes, "warnings": warnings,
    }


def forecast_prediction(*, model_dir=DEFAULT_MODEL_DIR, data_path=None, feature_count=46, **inputs):
    """Прогноз новой серии сохранённым ранкером по пересобранной истории."""
    model, _, metadata = verified_bundle(Path(model_dir), data_path, feature_count=feature_count)
    row, audit = prepare_forecast_features(protocol=metadata["protocol"], **inputs)
    columns = EXTENDED_TEAM_FEATURES if feature_count == 46 else TEAM_MODEL_FEATURES
    scores = team_ranking_scores(model, row, columns)[0]
    coverage = np.isfinite(team_rows(row, columns).to_numpy()).mean(axis=1)
    ids = [int(row.iloc[0][f"team{number}_id"]) for number in (1, 2)]
    exact_tie = bool(scores[0] == scores[1])
    selected = (0 if ids[0] < ids[1] else 1) if exact_tie else int(scores[1] > scores[0])
    p = float(expit(scores[0] - scores[1]))
    if min(coverage) < .75:
        audit["warnings"].append("Own-team feature coverage is below 75%; the score depends on many missing inputs.")
    definitions = EXTENDED_TEAM_FEATURE_GROUPS if feature_count == 46 else TEAM_FEATURE_GROUPS
    return {
        "mode": "new_pre_match_snapshot", "model": metadata["model_key"],
        "loss_function": "PairLogit", "feature_count_per_team": feature_count,
        "teams": [
            {"id": ids[index], "score": float(scores[index]),
             "win_probability": p if index == 0 else 1 - p,
             "feature_coverage": float(coverage[index])} for index in (0, 1)
        ],
        "predicted_winner_id": ids[selected], "exact_score_tie": exact_tie,
        "decision_rule": "Higher score; exact tie resolved by lower stable team ID",
        "probability_rule": "sigmoid(score_A - score_B); no post-hoc calibration",
        "own_feature_groups": {name: {"label": group["label"], "count": len(group["features"])}
                               for name, group in definitions.items()},
        "integrity": {key: value for key, value in metadata.items() if key.endswith("sha256")},
        **audit,
    }


def main():
    # Сохраняем UTF-8 и при перенаправлении JSON в файл или канал Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, help="Stored historical match; cannot be combined with new-series inputs")
    parser.add_argument("--verify", action="store_true", help="Read-only frozen model/schema/data/manifest verification without prediction")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data", type=Path, help="Optional exact SHA-256 matching copy of the frozen extended dataset")
    parser.add_argument("--feature-count", type=int, choices=(40, 46), default=46,
                        help="46 is the research representation; 40 reproduces its aggregate-only control")
    from src.predict_match import parse_int_list, parse_map_list
    parser.add_argument("--team1-id", type=int)
    parser.add_argument("--team2-id", type=int)
    parser.add_argument("--match-time")
    parser.add_argument("--bo", type=int, choices=(1, 3, 5), default=3)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--lan", action="store_true")
    location.add_argument("--online", action="store_true")
    parser.add_argument("--team1-rank", type=float)
    parser.add_argument("--team2-rank", type=float)
    parser.add_argument("--team1-players", type=parse_int_list, default=[])
    parser.add_argument("--team2-players", type=parse_int_list, default=[])
    for name in ("maps", "team1-picks", "team2-picks", "team1-removes", "team2-removes", "decider"):
        parser.add_argument(f"--{name}", type=parse_map_list, default=[])
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    args = parser.parse_args()
    try:
        new_series_inputs = any(value is not None for value in (args.team1_id, args.team2_id, args.match_time, args.team1_rank, args.team2_rank)) or args.lan or args.online or any((args.team1_players, args.team2_players, args.maps, args.team1_picks, args.team2_picks, args.team1_removes, args.team2_removes, args.decider))
        if args.verify:
            if args.match_id is not None or new_series_inputs:
                raise ValueError("--verify cannot be mixed with prediction inputs")
            _, _, metadata = verified_bundle(args.model_dir, args.data, feature_count=args.feature_count)
            result = {"mode": "frozen_bundle_verification", "verified": True,
                      "model": metadata["model_key"], "verified_files": metadata["verified_files"],
                      "feature_count_per_team": args.feature_count,
                      "integrity": {key: value for key, value in metadata.items() if key.endswith("sha256")}}
        elif args.match_id is not None:
            if new_series_inputs:
                raise ValueError("Historical --match-id mode cannot be mixed with new-series arguments")
            result = historical_prediction(args.match_id, args.model_dir, args.data, feature_count=args.feature_count)
        else:
            if args.team1_id is None or args.team2_id is None or args.match_time is None or not (args.lan or args.online):
                raise ValueError("New-series mode requires --team1-id, --team2-id, --match-time and --lan or --online")
            result = forecast_prediction(
                model_dir=args.model_dir, data_path=args.data, feature_count=args.feature_count,
                team1_id=args.team1_id, team2_id=args.team2_id, match_time=args.match_time,
                bo=args.bo, location="lan" if args.lan else "online", clean_dir=args.clean_dir,
                team1_rank=args.team1_rank, team2_rank=args.team2_rank,
                team1_players=args.team1_players, team2_players=args.team2_players,
                maps=args.maps, team1_picks=args.team1_picks, team2_picks=args.team2_picks,
                team1_removes=args.team1_removes, team2_removes=args.team2_removes, decider=args.decider,
            )
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
