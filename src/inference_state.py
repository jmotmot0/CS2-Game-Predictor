"""Compact historical state for fast future-match inference."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from src.feature_engineering import (
        MAP_COLUMNS,
        PLAYER_COLUMNS,
        ROSTER_COLUMNS,
        TEAM_HISTORY_COLUMNS,
        VETO_COLUMNS,
        normalize_lineups,
        normalize_maps,
        normalize_matches,
        normalize_player_stats,
        normalize_veto,
        read_clean_tables,
    )
    from src.modeling import MODEL_FEATURES, write_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from feature_engineering import (  # type: ignore[no-redef]
        MAP_COLUMNS,
        PLAYER_COLUMNS,
        ROSTER_COLUMNS,
        TEAM_HISTORY_COLUMNS,
        VETO_COLUMNS,
        normalize_lineups,
        normalize_maps,
        normalize_matches,
        normalize_player_stats,
        normalize_veto,
        read_clean_tables,
    )
    from modeling import MODEL_FEATURES, write_json  # type: ignore[no-redef]


INFERENCE_STATE_SCHEMA_VERSION = 3


def _blank_team() -> dict[str, Any]:
    return {
        "elo": 1500.0,
        "matches": 0,
        "wins": 0,
        "recent_dates": [],
        "last_match_date": None,
        "recent_results": [],
        "streak": 0,
        "opponent_elo": [],
        "rank": None,
        "roster": [],
    }


def _finite(value: Any) -> bool:
    return pd.notna(value) and np.isfinite(value)


def build_inference_state(clean_dir: Path, *, elo_k: float = 48.0) -> dict[str, Any]:
    matches_raw, lineups_raw, veto_raw, maps_raw, player_raw = read_clean_tables(clean_dir)
    matches = normalize_matches(matches_raw)
    valid_ids = set(matches["match_id"].astype(int))
    lineups = normalize_lineups(lineups_raw, valid_ids)
    maps = normalize_maps(maps_raw, matches)
    players = normalize_player_stats(player_raw, matches)
    veto = normalize_veto(veto_raw, matches)

    teams: dict[int, dict[str, Any]] = defaultdict(_blank_team)
    h2h: dict[str, dict[str, Any]] = defaultdict(lambda: {"wins": {}, "recent": []})
    timestamps = matches["match_datetime_utc"].astype("int64").to_numpy()
    timestamp_values = matches["match_datetime_utc"].to_numpy()
    team1_ids = matches["team1_id"].astype(int).to_numpy()
    team2_ids = matches["team2_id"].astype(int).to_numpy()
    outcomes = matches["team1_win"].astype(int).to_numpy()
    ranks1 = pd.to_numeric(matches.get("team1_rank"), errors="coerce").to_numpy()
    ranks2 = pd.to_numeric(matches.get("team2_rank"), errors="coerce").to_numpy()

    start = 0
    while start < len(matches):
        stop = start + 1
        while stop < len(matches) and timestamps[stop] == timestamps[start]:
            stop += 1
        deltas: dict[int, float] = defaultdict(float)
        for index in range(start, stop):
            team1_id = int(team1_ids[index])
            team2_id = int(team2_ids[index])
            outcome1 = int(outcomes[index])
            outcome2 = 1 - outcome1
            team1 = teams[team1_id]
            team2 = teams[team2_id]
            rating1 = float(team1["elo"])
            rating2 = float(team2["elo"])
            probability1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
            deltas[team1_id] += elo_k * (outcome1 - probability1)
            deltas[team2_id] += elo_k * (outcome2 - (1 - probability1))
            # Сохраняем единицу времени массива: pandas 3 часто использует
            # микросекунды. Их ошибочная трактовка как наносекунд уменьшила бы
            # все интервалы в 1000 раз.
            timestamp = pd.Timestamp(timestamp_values[index])

            for team, result, opponent_rating in [
                (team1, outcome1, rating2),
                (team2, outcome2, rating1),
            ]:
                team["matches"] += 1
                team["wins"] += int(result)
                team["recent_dates"].append(timestamp)
                team["last_match_date"] = timestamp
                team["recent_results"].append(result)
                team["recent_results"] = team["recent_results"][-20:]
                team["opponent_elo"].append(opponent_rating)
                team["opponent_elo"] = team["opponent_elo"][-10:]
                if result:
                    team["streak"] = max(0, int(team["streak"])) + 1
                else:
                    team["streak"] = min(0, int(team["streak"])) - 1
            if _finite(ranks1[index]):
                team1["rank"] = float(ranks1[index])
            if _finite(ranks2[index]):
                team2["rank"] = float(ranks2[index])

            key = f"{min(team1_id, team2_id)}:{max(team1_id, team2_id)}"
            winner = team1_id if outcome1 else team2_id
            record = h2h[key]
            record["wins"][str(winner)] = int(record["wins"].get(str(winner), 0)) + 1
            record["recent"].append(winner)
            record["recent"] = record["recent"][-5:]
        for team_id, delta in deltas.items():
            teams[team_id]["elo"] = float(teams[team_id]["elo"]) + delta
        start = stop

    state_time = matches["match_datetime_utc"].max()
    if pd.isna(state_time):
        raise ValueError("Cannot build inference state from an empty match history")
    keep_after = state_time - pd.Timedelta(days=90)
    for team in teams.values():
        team["recent_dates"] = [
            value.isoformat() for value in team["recent_dates"] if value >= keep_after
        ]
        if team["last_match_date"] is not None:
            team["last_match_date"] = team["last_match_date"].isoformat()

    lineup_sets = {
        (int(match_id), int(team_id)): sorted(set(group["player_id"].astype(int)))
        for (match_id, team_id), group in lineups.groupby(["match_id", "team_id"], sort=False)
    }
    for row in matches.itertuples(index=False):
        match_id = int(row.match_id)
        for team_id in [int(row.team1_id), int(row.team2_id)]:
            roster = lineup_sets.get((match_id, team_id))
            if roster:
                teams[team_id]["roster"] = roster

    player_state: dict[str, dict[str, float]] = {}
    for player_id, group in players.groupby("player_id", sort=False):
        opening = group["opening_kills"].fillna(0) - group["opening_deaths"].fillna(0)
        player_state[str(int(player_id))] = {
            "maps": float(len(group)),
            "rating_sum": float(group["rating"].sum(skipna=True)),
            "rating_n": float(group["rating"].notna().sum()),
            "adr_sum": float(group["adr"].sum(skipna=True)),
            "adr_n": float(group["adr"].notna().sum()),
            "kast_sum": float(group["kast"].sum(skipna=True)),
            "kast_n": float(group["kast"].notna().sum()),
            "opening_sum": float(opening.sum(skipna=True)),
            "opening_n": float((group["opening_kills"].notna() | group["opening_deaths"].notna()).sum()),
        }

    map_state: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "wins": 0.0,
            "ct_sum": 0.0,
            "ct_n": 0.0,
            "t_sum": 0.0,
            "t_n": 0.0,
        }
    )
    for row in maps.itertuples(index=False):
        if pd.isna(row.team1_map_score) or pd.isna(row.team2_map_score):
            continue
        if float(row.team1_map_score) == float(row.team2_map_score):
            continue

        def side_rate(won: object, lost: object) -> float:
            if not _finite(won) or not _finite(lost):
                return np.nan
            rounds = float(won) + float(lost)
            return float(won) / rounds if rounds > 0 else np.nan

        for team_id, win, ct_rounds, t_rounds in [
            (
                int(row.team1_id),
                int(row.team1_map_score > row.team2_map_score),
                side_rate(row.team1_ct_rounds, row.team2_t_rounds),
                side_rate(row.team1_t_rounds, row.team2_ct_rounds),
            ),
            (
                int(row.team2_id),
                int(row.team2_map_score > row.team1_map_score),
                side_rate(row.team2_ct_rounds, row.team1_t_rounds),
                side_rate(row.team2_t_rounds, row.team1_ct_rounds),
            ),
        ]:
            record = map_state[f"{team_id}:{row.map_name}"]
            record["count"] += 1
            record["wins"] += win
            if _finite(ct_rounds):
                record["ct_sum"] += float(ct_rounds)
                record["ct_n"] += 1
            if _finite(t_rounds):
                record["t_sum"] += float(t_rounds)
                record["t_n"] += 1

    veto_state: dict[str, dict[str, int]] = defaultdict(
        lambda: {"picked": 0, "removed": 0, "left_over": 0, "total": 0}
    )
    match_teams = {
        int(row.match_id): (int(row.team1_id), int(row.team2_id))
        for row in matches.itertuples(index=False)
    }
    if "team_id" in veto.columns:
        for row in veto.itertuples(index=False):
            action = str(row.action)
            team_ids: list[int] = []
            if _finite(row.team_id):
                team_ids.append(int(row.team_id))
            elif action == "left_over":
                team_ids.extend(match_teams.get(int(row.match_id), ()))
            for team_id in team_ids:
                record = veto_state[f"{team_id}:{row.map_name}"]
                if action in record:
                    record[action] += 1
                    record["total"] += 1

    return {
        "schema_version": INFERENCE_STATE_SCHEMA_VERSION,
        "state_as_of_utc": state_time.isoformat(),
        "elo_k": float(elo_k),
        "teams": {str(team_id): value for team_id, value in teams.items()},
        "h2h": dict(h2h),
        "players": player_state,
        "maps": dict(map_state),
        "veto": dict(veto_state),
    }


def save_inference_state(clean_dir: Path, output_path: Path, *, elo_k: float = 48.0) -> None:
    state = build_inference_state(clean_dir, elo_k=elo_k)
    write_json(output_path, state)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.mean(finite)) if finite else np.nan


def prediction_features_from_state(
    state: dict[str, Any],
    *,
    team1_id: int,
    team2_id: int,
    match_time: pd.Timestamp,
    bo: int,
    location: str,
    team1_rank: float | None,
    team2_rank: float | None,
    team1_players: list[int],
    team2_players: list[int],
    maps: list[str],
    min_player_history_maps: int = 5,
    team1_picks: list[str] | None = None,
    team2_picks: list[str] | None = None,
    team1_removes: list[str] | None = None,
    team2_removes: list[str] | None = None,
    deciders: list[str] | None = None,
) -> pd.Series:
    if state.get("schema_version") != INFERENCE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Inference state schema is incompatible with this code; rebuild "
            "artifacts/inference_state.json."
        )
    state_time = pd.Timestamp(state["state_as_of_utc"])
    if match_time <= state_time:
        raise ValueError(
            f"Fast inference state is valid only after {state_time.isoformat()}; "
            "use --rebuild-history for an earlier timestamp."
        )
    teams = state["teams"]
    players = state["players"]
    map_state = state["maps"]
    veto_state = state.get("veto", {})

    def team_snapshot(team_id: int, opponent_id: int) -> dict[str, float]:
        team = teams.get(str(team_id), _blank_team())
        opponent = teams.get(str(opponent_id), _blank_team())
        dates = [pd.Timestamp(value) for value in team.get("recent_dates", [])]
        last_match = team.get("last_match_date")
        recent = list(team.get("recent_results", []))
        pair_key = f"{min(team_id, opponent_id)}:{max(team_id, opponent_id)}"
        pair = state.get("h2h", {}).get(pair_key, {"wins": {}, "recent": []})
        streak = int(team.get("streak", 0))
        return {
            "elo_pre": float(team.get("elo", 1500.0)),
            "opp_elo_pre": float(opponent.get("elo", 1500.0)),
            "elo_diff_pre": float(team.get("elo", 1500.0)) - float(opponent.get("elo", 1500.0)),
            "matches_before": float(team.get("matches", 0)),
            "days_since_last_match": (
                (match_time - pd.Timestamp(last_match)).total_seconds() / 86400.0
                if last_match
                else np.nan
            ),
            "activity_7d": float(sum((match_time - value).total_seconds() <= 7 * 86400 for value in dates)),
            "activity_30d": float(sum((match_time - value).total_seconds() <= 30 * 86400 for value in dates)),
            "activity_90d": float(sum((match_time - value).total_seconds() <= 90 * 86400 for value in dates)),
            "overall_winrate": (
                float(team.get("wins", 0)) / float(team.get("matches", 0))
                if int(team.get("matches", 0)) > 0
                else np.nan
            ),
            "winrate_last_5": _mean(recent[-5:]),
            "winrate_last_10": _mean(recent[-10:]),
            "winrate_last_20": _mean(recent[-20:]),
            "win_streak": float(max(streak, 0)),
            "loss_streak": float(max(-streak, 0)),
            "avg_opp_elo_last_10": _mean(team.get("opponent_elo", [])),
            "h2h_wins_all": float(pair.get("wins", {}).get(str(team_id), 0)),
            "h2h_wins_last5": float(sum(int(winner) == team_id for winner in pair.get("recent", []))),
        }

    def roster_snapshot(team_id: int, current_players: list[int]) -> dict[str, float]:
        if not current_players:
            return {column: np.nan for column in ROSTER_COLUMNS}
        previous = set(teams.get(str(team_id), {}).get("roster", []))
        current = set(current_players)
        overlap = len(previous & current)
        return {
            "roster_size": float(len(current)),
            "roster_overlap_prev": float(overlap),
            "roster_overlap_prev_ratio": overlap / len(current) if current else np.nan,
        }

    def player_snapshot(current_players: list[int]) -> dict[str, float]:
        if not current_players:
            return {column: np.nan for column in PLAYER_COLUMNS}
        known = [
            players[str(player)]
            for player in current_players
            if str(player) in players and players[str(player)]["maps"] >= min_player_history_maps
        ]

        def average(sum_key: str, count_key: str) -> float:
            return _mean(item[sum_key] / item[count_key] for item in known if item[count_key] > 0)

        return {
            "lineup_history_coverage": (
                float(len(known)) / float(len(current_players))
                if current_players
                else np.nan
            ),
            "lineup_players_with_history": float(len(known)),
            "lineup_player_rating_mean": average("rating_sum", "rating_n"),
            "lineup_player_adr_mean": average("adr_sum", "adr_n"),
            "lineup_player_kast_mean": average("kast_sum", "kast_n"),
            "lineup_player_opening_diff_mean": average("opening_sum", "opening_n"),
            "lineup_player_maps_played_mean": _mean(item["maps"] for item in known),
        }

    def map_snapshot(team_id: int) -> dict[str, float]:
        if not maps:
            return {column: np.nan for column in MAP_COLUMNS}
        records = [map_state.get(f"{team_id}:{name}") for name in maps]
        observed = [record for record in records if record is not None and record["count"]]
        return {
            "avg_map_count_before": _mean(
                0.0 if record is None else record["count"] for record in records
            ),
            "avg_map_wr_before": _mean(
                record["wins"] / record["count"] for record in observed
            ),
            "avg_map_ct_wr_before": _mean(
                record["ct_sum"] / record["ct_n"] for record in observed if record["ct_n"]
            ),
            "avg_map_t_wr_before": _mean(
                record["t_sum"] / record["t_n"] for record in observed if record["t_n"]
            ),
            "series_maps_known": float(len(observed)),
        }

    def veto_snapshot(
        team_id: int,
        *,
        picks: list[str],
        removes: list[str],
    ) -> dict[str, float]:
        by_action = {
            "picked": picks,
            "removed": removes,
            "left_over": deciders or [],
        }
        result: dict[str, float] = {}
        output_names = {
            "picked": "veto_pick_rate_before",
            "removed": "veto_remove_rate_before",
            "left_over": "veto_leftover_rate_before",
        }
        for action, names in by_action.items():
            values: list[float] = []
            for name in dict.fromkeys(names):
                record = veto_state.get(f"{team_id}:{name}")
                if record and record.get("total", 0):
                    values.append(float(record.get(action, 0)) / float(record["total"]))
            result[output_names[action]] = _mean(values)
        return result

    snapshots = []
    for team_id, opponent_id, lineup, picks, removes in [
        (team1_id, team2_id, team1_players, team1_picks or [], team1_removes or []),
        (team2_id, team1_id, team2_players, team2_picks or [], team2_removes or []),
    ]:
        snapshots.append(
            {
                **team_snapshot(team_id, opponent_id),
                **roster_snapshot(team_id, lineup),
                **player_snapshot(lineup),
                **map_snapshot(team_id),
                **veto_snapshot(team_id, picks=picks, removes=removes),
            }
        )

    rank1 = team1_rank if team1_rank is not None else teams.get(str(team1_id), {}).get("rank")
    rank2 = team2_rank if team2_rank is not None else teams.get(str(team2_id), {}).get("rank")
    row: dict[str, float] = {
        "bo1": float(bo == 1),
        "bo3": float(bo == 3),
        "bo5": float(bo == 5),
        "is_lan": float(location == "lan"),
        "is_online": float(location == "online"),
        "rank_available": float(rank1 is not None and rank2 is not None and _finite(rank1) and _finite(rank2)),
        "diff_rank": float(rank2) - float(rank1) if _finite(rank1) and _finite(rank2) else np.nan,
    }
    for name in TEAM_HISTORY_COLUMNS + ROSTER_COLUMNS + PLAYER_COLUMNS + MAP_COLUMNS + VETO_COLUMNS:
        if name in {"opp_elo_pre", "elo_diff_pre"}:
            continue
        row[f"diff_{name}"] = snapshots[0][name] - snapshots[1][name]
    return pd.Series({feature: row.get(feature, np.nan) for feature in MODEL_FEATURES})
