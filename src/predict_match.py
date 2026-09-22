"""Build a point-in-time snapshot and predict a CS2 match outcome."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from src.feature_engineering import (
        MAP_COLUMNS,
        PLAYER_COLUMNS,
        ROSTER_COLUMNS,
        build_feature_dataset,
        normalize_map_name,
        normalize_matches,
        read_clean_tables,
    )
    from src.inference_state import prediction_features_from_state
    from src.modeling import feature_coverage, load_schema, symmetrized_probability
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from feature_engineering import (  # type: ignore[no-redef]
        MAP_COLUMNS,
        PLAYER_COLUMNS,
        ROSTER_COLUMNS,
        build_feature_dataset,
        normalize_map_name,
        normalize_matches,
        read_clean_tables,
    )
    from inference_state import prediction_features_from_state  # type: ignore[no-redef]
    from modeling import (  # type: ignore[no-redef]
        feature_coverage,
        load_schema,
        symmetrized_probability,
    )


PENDING_MATCH_ID = -1


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            try:
                player_id = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid player identifier: {item!r}"
                ) from exc
            if player_id <= 0:
                raise argparse.ArgumentTypeError("Player identifiers must be positive")
            result.append(player_id)
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Player identifiers must be unique within a lineup")
    return result


def parse_map_list(value: str | None) -> list[str]:
    if not value:
        return []
    result = [normalize_map_name(item) for item in value.split(",") if item.strip()]
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Map names must be unique within one option")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict a future match from historical clean tables. The command rebuilds a strict "
            "pre-match snapshot and reports feature coverage."
        )
    )
    parser.add_argument("--team1-id", required=True, type=int)
    parser.add_argument("--team2-id", required=True, type=int)
    parser.add_argument("--match-time", required=True, help="ISO-8601 timestamp, for example 2026-04-12T15:00:00Z")
    parser.add_argument("--bo", type=int, choices=[1, 3, 5], default=3)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--lan", action="store_true")
    location.add_argument("--online", action="store_true")
    parser.add_argument("--team1-rank", type=float)
    parser.add_argument("--team2-rank", type=float)
    parser.add_argument(
        "--team1-players",
        type=parse_int_list,
        default=[],
        help="Comma-separated current player IDs (exactly five when supplied)",
    )
    parser.add_argument(
        "--team2-players",
        type=parse_int_list,
        default=[],
        help="Comma-separated current player IDs (exactly five when supplied)",
    )
    maps_group = parser.add_mutually_exclusive_group()
    maps_group.add_argument(
        "--maps",
        type=parse_map_list,
        default=[],
        help="Comma-separated post-veto map pool",
    )
    maps_group.add_argument(
        "--candidate-maps",
        type=parse_map_list,
        default=[],
        help=(
            "Comma-separated maps available before veto. The prediction is averaged "
            "uniformly over every BO-sized map combination."
        ),
    )
    parser.add_argument("--team1-picks", type=parse_map_list, default=[])
    parser.add_argument("--team2-picks", type=parse_map_list, default=[])
    parser.add_argument("--team1-removes", type=parse_map_list, default=[])
    parser.add_argument("--team2-removes", type=parse_map_list, default=[])
    parser.add_argument("--decider", type=parse_map_list, default=[])
    parser.add_argument("--clean-dir", type=Path, default=Path("data/interim/hltv_final_clean"))
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("artifacts/research_revision_2026-09-30/deployment"),
        help="Research-selected classifier bundle; use --model-dir artifacts for the archived model.",
    )
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Ignore the compact state and rebuild history from clean tables.",
    )
    return parser.parse_args()


def generate_map_scenarios(candidate_maps: list[str], bo: int) -> list[list[str]]:
    if not candidate_maps:
        return []
    if bo not in {1, 3, 5}:
        raise ValueError("BO must be one of 1, 3 or 5")
    if len(candidate_maps) < bo:
        raise ValueError(
            f"At least {bo} candidate maps are required for a BO{bo} pre-veto simulation"
        )
    return [list(values) for values in combinations(candidate_maps, bo)]


def latest_rank(history: pd.DataFrame, team_id: int) -> float:
    for row in history.iloc[::-1].itertuples(index=False):
        if int(row.team1_id) == team_id and pd.notna(getattr(row, "team1_rank", np.nan)):
            return float(row.team1_rank)
        if int(row.team2_id) == team_id and pd.notna(getattr(row, "team2_rank", np.nan)):
            return float(row.team2_rank)
    return np.nan


def append_lineup_rows(
    lineups: pd.DataFrame,
    *,
    team_id: int,
    team_ordinal: int,
    players: Iterable[int],
) -> pd.DataFrame:
    additions = [
        {
            "match_id": PENDING_MATCH_ID,
            "team_ordinal": team_ordinal,
            "team_id": team_id,
            "team_name": f"team_{team_id}",
            "player_id": player,
            "player_name": f"player_{player}",
        }
        for player in players
    ]
    return pd.concat([lineups, pd.DataFrame(additions)], ignore_index=True, sort=False) if additions else lineups


def append_map_rows(maps: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    additions = [
        {
            "match_id": PENDING_MATCH_ID,
            "map_no": index,
            "map_name": name,
            "played": True,
            "team1_map_score": np.nan,
            "team2_map_score": np.nan,
            "team1_ct_rounds": np.nan,
            "team1_t_rounds": np.nan,
            "team2_ct_rounds": np.nan,
            "team2_t_rounds": np.nan,
        }
        for index, name in enumerate(names, start=1)
    ]
    return pd.concat([maps, pd.DataFrame(additions)], ignore_index=True, sort=False) if additions else maps


def append_veto_rows(
    veto: pd.DataFrame,
    *,
    team1_id: int,
    team2_id: int,
    team1_picks: list[str],
    team2_picks: list[str],
    team1_removes: list[str],
    team2_removes: list[str],
    deciders: list[str],
) -> pd.DataFrame:
    additions: list[dict[str, object]] = []
    step = 1
    for team_id, action, names in [
        (team1_id, "picked", team1_picks),
        (team2_id, "picked", team2_picks),
        (team1_id, "removed", team1_removes),
        (team2_id, "removed", team2_removes),
    ]:
        for name in names:
            additions.append(
                {
                    "match_id": PENDING_MATCH_ID,
                    "step_number": step,
                    "team_name": f"team_{team_id}",
                    "action": action,
                    "map_name": name,
                }
            )
            step += 1
    for name in deciders:
        additions.append(
            {
                "match_id": PENDING_MATCH_ID,
                "step_number": step,
                "team_name": pd.NA,
                "action": "left_over",
                "map_name": name,
            }
        )
        step += 1
    if not additions:
        return veto
    return pd.concat([veto, pd.DataFrame(additions)], ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    if args.team1_id <= 0 or args.team2_id <= 0:
        raise ValueError("Team identifiers must be positive")
    if args.team1_id == args.team2_id:
        raise ValueError("team1-id and team2-id must be different")
    for name, value in [("team1-rank", args.team1_rank), ("team2-rank", args.team2_rank)]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number")
    for name, lineup in [
        ("team1-players", args.team1_players),
        ("team2-players", args.team2_players),
    ]:
        if lineup and len(lineup) != 5:
            raise ValueError(f"{name} must contain exactly five unique player IDs")
    if set(args.team1_players) & set(args.team2_players):
        raise ValueError("The two lineups must not contain the same player ID")
    if len(args.decider) > 1:
        raise ValueError("At most one decider map can be supplied")

    veto_groups = [
        args.team1_picks,
        args.team2_picks,
        args.team1_removes,
        args.team2_removes,
        args.decider,
    ]
    veto_maps = [name for group in veto_groups for name in group]
    if len(veto_maps) != len(set(veto_maps)):
        raise ValueError("A map cannot appear in more than one veto action")
    map_names = list(args.maps)
    map_scenarios = generate_map_scenarios(list(args.candidate_maps), args.bo)
    if map_scenarios and veto_maps:
        raise ValueError("Detailed veto actions cannot be combined with --candidate-maps")
    if map_scenarios and args.rebuild_history:
        raise ValueError(
            "Pre-veto scenario simulation uses the compact inference state; "
            "remove --rebuild-history"
        )
    if veto_maps:
        if map_names and not set(args.team1_picks + args.team2_picks + args.decider).issubset(map_names):
            raise ValueError("Picked and decider maps must be included in --maps")
        if not map_names:
            map_names = list(dict.fromkeys(args.team1_picks + args.team2_picks + args.decider))
    if len(map_names) > args.bo:
        raise ValueError("The post-veto map pool cannot contain more maps than the series BO")

    try:
        match_time = pd.Timestamp(args.match_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("match-time must be a valid ISO-8601 timestamp") from exc
    if match_time.tzinfo is None:
        match_time = match_time.tz_localize("UTC")
    else:
        match_time = match_time.tz_convert("UTC")

    schema = load_schema(args.model_dir)
    feature_order = list(schema["feature_order"])
    location = "lan" if args.lan else "online" if args.online else "unknown"
    team1_players = list(args.team1_players)
    team2_players = list(args.team2_players)
    state_path = args.model_dir / "inference_state.json"
    if state_path.exists() and not args.rebuild_history:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        scenario_inputs = map_scenarios or [map_names]
        rows = [
            prediction_features_from_state(
                state,
                team1_id=args.team1_id,
                team2_id=args.team2_id,
                match_time=match_time,
                bo=args.bo,
                location=location,
                team1_rank=args.team1_rank,
                team2_rank=args.team2_rank,
                team1_players=team1_players,
                team2_players=team2_players,
                maps=scenario,
                team1_picks=args.team1_picks,
                team2_picks=args.team2_picks,
                team1_removes=args.team1_removes,
                team2_removes=args.team2_removes,
                deciders=args.decider,
            )
            for scenario in scenario_inputs
        ]
    else:
        matches, lineups, veto, maps, player_stats = read_clean_tables(args.clean_dir)
        normalized = normalize_matches(matches, include_pending=True)
        history = normalized[
            (normalized["match_datetime_utc"] < match_time)
            & normalized["team1_win"].isin([0, 1])
        ].copy()
        if history.empty:
            raise ValueError("No completed historical matches exist before match-time")
        team1_rank = (
            float(args.team1_rank)
            if args.team1_rank is not None
            else latest_rank(history, args.team1_id)
        )
        team2_rank = (
            float(args.team2_rank)
            if args.team2_rank is not None
            else latest_rank(history, args.team2_id)
        )
        pending = {
            "match_id": PENDING_MATCH_ID,
            "match_date": match_time.date().isoformat(),
            "match_datetime_utc": match_time.isoformat(),
            "event_name": "prediction",
            "team1": f"team_{args.team1_id}",
            "team2": f"team_{args.team2_id}",
            "team1_id": args.team1_id,
            "team2_id": args.team2_id,
            "team1_rank": team1_rank,
            "team2_rank": team2_rank,
            "team1_score": np.nan,
            "team2_score": np.nan,
            "team1_win": np.nan,
            "is_valid_result": 0,
            "bo": f"bo{args.bo}",
            "lan_online": location,
        }
        prediction_matches = pd.concat(
            [history, pd.DataFrame([pending])], ignore_index=True, sort=False
        )
        valid_history_ids = set(history["match_id"].astype(int))
        lineups = lineups[
            pd.to_numeric(lineups["match_id"], errors="coerce").isin(valid_history_ids)
        ].copy()
        veto = veto[pd.to_numeric(veto["match_id"], errors="coerce").isin(valid_history_ids)].copy()
        maps = maps[pd.to_numeric(maps["match_id"], errors="coerce").isin(valid_history_ids)].copy()
        player_stats = player_stats[
            pd.to_numeric(player_stats["match_id"], errors="coerce").isin(valid_history_ids)
        ].copy()
        lineups = append_lineup_rows(
            lineups,
            team_id=args.team1_id,
            team_ordinal=1,
            players=team1_players,
        )
        lineups = append_lineup_rows(
            lineups,
            team_id=args.team2_id,
            team_ordinal=2,
            players=team2_players,
        )
        maps = append_map_rows(maps, map_names)
        veto = append_veto_rows(
            veto,
            team1_id=args.team1_id,
            team2_id=args.team2_id,
            team1_picks=args.team1_picks,
            team2_picks=args.team2_picks,
            team1_removes=args.team1_removes,
            team2_removes=args.team2_removes,
            deciders=args.decider,
        )
        features = build_feature_dataset(
            prediction_matches,
            lineups,
            veto,
            maps,
            player_stats,
            elo_k=float(schema.get("elo_k", 48.0)),
            include_pending=True,
            map_pool_overrides={PENDING_MATCH_ID: map_names} if map_names else None,
        )
        pending_row = features.loc[features["match_id"].eq(PENDING_MATCH_ID)]
        if len(pending_row) != 1:
            raise RuntimeError(f"Expected one prediction row, got {len(pending_row)}")
        row = pending_row.iloc[0]

        # Отсутствие текущих данных означает неизвестные значения, а не равенство команд.
        if not team1_players or not team2_players:
            for name in ROSTER_COLUMNS + PLAYER_COLUMNS:
                row[f"diff_{name}"] = np.nan
        if not map_names:
            for name in MAP_COLUMNS:
                row[f"diff_{name}"] = np.nan
        rows = [row]

    from catboost import CatBoostClassifier

    model_path = args.model_dir / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    if model.feature_names_ and model.feature_names_ != feature_order:
        raise ValueError("Model feature order does not match feature_schema.json")
    representative_row = rows[0]
    model_input = pd.DataFrame(
        [[row.get(feature, np.nan) for feature in feature_order] for row in rows],
        columns=feature_order,
    )
    scenario_probabilities = symmetrized_probability(model, model_input)
    probability1 = float(np.mean(scenario_probabilities))
    probability2 = 1.0 - probability1
    coverages = [feature_coverage(row, feature_order) for row in rows]
    coverage = float(np.mean(coverages))

    warnings: list[str] = []
    if not team1_players or not team2_players:
        warnings.append("Current lineups were not fully supplied; player and roster coverage is reduced.")
    if map_scenarios:
        warnings.append(
            "Pre-veto estimate assumes every candidate map combination is equally likely; "
            "the actual veto can shift the probability."
        )
    elif not map_names:
        warnings.append("Post-veto maps were not supplied; map-pool features are unavailable.")
    if not veto_maps:
        warnings.append("Detailed veto actions were not supplied; veto-history features are unavailable.")
    if not bool(representative_row.get("rank_available", 0)):
        warnings.append("At least one current rank is unavailable.")
    if coverage < 0.75:
        warnings.append("Feature coverage is below 75%; interpret the probability cautiously.")

    result = {
        "team1_id": args.team1_id,
        "team2_id": args.team2_id,
        "match_time_utc": match_time.isoformat(),
        "team1_win_probability": probability1,
        "team2_win_probability": probability2,
        "predicted_winner_team_id": args.team1_id if probability1 >= 0.5 else args.team2_id,
        "model": schema.get("model_name", "catboost"),
        "feature_coverage": coverage,
        "features_available": int(round(float(model_input.notna().sum(axis=1).mean()))),
        "features_total": len(feature_order),
        "warnings": warnings,
    }
    if map_scenarios:
        result.update(
            {
                "prediction_mode": "pre_veto_uniform_map_scenarios",
                "map_scenario_count": len(map_scenarios),
                "candidate_maps": list(args.candidate_maps),
                "team1_probability_scenario_min": float(np.min(scenario_probabilities)),
                "team1_probability_scenario_p10": float(
                    np.quantile(scenario_probabilities, 0.10)
                ),
                "team1_probability_scenario_p90": float(
                    np.quantile(scenario_probabilities, 0.90)
                ),
                "team1_probability_scenario_max": float(np.max(scenario_probabilities)),
            }
        )
    else:
        result["prediction_mode"] = "post_veto" if map_names else "maps_unavailable"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
