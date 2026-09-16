"""Leakage-safe feature engineering for professional CS2 matches.

The central rule in this module is simple: all snapshots for timestamp ``t``
are calculated before any result observed at ``t`` updates historical state.
This rule is applied consistently to team form, Elo, head-to-head history,
rosters, player statistics, map pool and veto history.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAP_NAME_NORMALIZATION = {
    "dust 2": "dust2",
    "de_dust2": "dust2",
    "de_inferno": "inferno",
    "de_mirage": "mirage",
    "de_nuke": "nuke",
    "de_ancient": "ancient",
    "de_anubis": "anubis",
    "de_train": "train",
    "de_vertigo": "vertigo",
    "de_overpass": "overpass",
}


def normalize_map_name(value: object) -> str:
    """Return the canonical lower-case map identifier used by every pipeline stage."""

    name = str(value).strip().lower()
    return MAP_NAME_NORMALIZATION.get(name, name)

TEAM_HISTORY_COLUMNS = [
    "elo_pre",
    "opp_elo_pre",
    "elo_diff_pre",
    "matches_before",
    "days_since_last_match",
    "activity_7d",
    "activity_30d",
    "activity_90d",
    "overall_winrate",
    "winrate_last_5",
    "winrate_last_10",
    "winrate_last_20",
    "win_streak",
    "loss_streak",
    "avg_opp_elo_last_10",
    "h2h_wins_all",
    "h2h_wins_last5",
]

ROSTER_COLUMNS = [
    "roster_size",
    "roster_overlap_prev",
    "roster_overlap_prev_ratio",
]

PLAYER_COLUMNS = [
    "lineup_history_coverage",
    "lineup_players_with_history",
    "lineup_player_rating_mean",
    "lineup_player_adr_mean",
    "lineup_player_kast_mean",
    "lineup_player_opening_diff_mean",
    "lineup_player_maps_played_mean",
]

MAP_COLUMNS = [
    "avg_map_count_before",
    "avg_map_wr_before",
    "avg_map_ct_wr_before",
    "avg_map_t_wr_before",
    "series_maps_known",
]

VETO_COLUMNS = [
    "veto_pick_rate_before",
    "veto_remove_rate_before",
    "veto_leftover_rate_before",
]


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_clean_tables(
    clean_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "matches": clean_dir / "matches_final.csv",
        "lineups": clean_dir / "match_lineups.csv",
        "veto": clean_dir / "veto_steps.csv",
        "maps": clean_dir / "match_maps.csv",
        "player_stats": clean_dir / "map_player_stats.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Clean dataset is incomplete. Missing: {missing}")
    return tuple(pd.read_csv(path, low_memory=False) for path in paths.values())  # type: ignore[return-value]


def normalize_matches(matches: pd.DataFrame, *, include_pending: bool = False) -> pd.DataFrame:
    ensure_columns(
        matches,
        ["match_id", "team1_id", "team2_id", "team1_win"],
        "matches_final",
    )
    df = matches.copy()
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    for column in [
        "team1_id",
        "team2_id",
        "team1_rank",
        "team2_rank",
        "team1_score",
        "team2_score",
        "team1_win",
        "is_valid_result",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "match_date" in df.columns:
        match_date = pd.to_datetime(df["match_date"], errors="coerce", utc=True).dt.normalize()
    else:
        match_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if "match_datetime_utc" in df.columns:
        match_time = pd.to_datetime(df["match_datetime_utc"], errors="coerce", utc=True)
    else:
        match_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    missing_time = match_time.isna() & match_date.notna()
    match_time.loc[missing_time] = match_date.loc[missing_time]

    # A page occasionally contains a stale timestamp that disagrees with the
    # canonical result date. The exact time is then unknown, so noon UTC keeps
    # the match within the correct day without pretending midnight precision.
    date_gap = (match_time.dt.normalize() - match_date).abs()
    repaired = match_date.notna() & match_time.notna() & date_gap.gt(pd.Timedelta(days=2))
    match_time.loc[repaired] = match_date.loc[repaired] + pd.Timedelta(hours=12)
    df["match_datetime_utc"] = match_time
    df["datetime_repaired"] = repaired.astype(int)
    if "match_date" not in df.columns:
        df["match_date"] = match_time.dt.date.astype("string")

    bo_source = df["bo"] if "bo" in df.columns else pd.Series(pd.NA, index=df.index)
    df["bo"] = pd.to_numeric(
        bo_source.astype("string").str.lower().str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    location = (
        df["lan_online"].astype("string").str.strip().str.lower()
        if "lan_online" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    df["is_lan"] = location.eq("lan").fillna(False).astype(int)
    df["is_online"] = location.eq("online").fillna(False).astype(int)

    required_non_null = ["match_id", "match_datetime_utc", "team1_id", "team2_id"]
    df = df.dropna(subset=required_non_null).copy()
    if not include_pending:
        if "is_valid_result" in df.columns:
            df = df[df["is_valid_result"].eq(1)].copy()
        df = df[df["team1_win"].isin([0, 1])].copy()

    if df["match_id"].duplicated().any():
        duplicates = df.loc[df["match_id"].duplicated(), "match_id"].head(10).tolist()
        raise ValueError(f"matches_final contains duplicate match_id values: {duplicates}")

    return df.sort_values(["match_datetime_utc", "match_id"]).reset_index(drop=True)


def normalize_lineups(lineups: pd.DataFrame, valid_match_ids: set[int]) -> pd.DataFrame:
    ensure_columns(lineups, ["match_id", "team_id", "player_id"], "match_lineups")
    df = lineups.copy()
    for column in ["match_id", "team_ordinal", "team_id", "player_id"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["match_id", "team_id", "player_id"])
    df = df[df["match_id"].astype(int).isin(valid_match_ids)].copy()
    return df.drop_duplicates(["match_id", "team_id", "player_id"]).reset_index(drop=True)


def normalize_maps(maps: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        maps,
        ["match_id", "map_no", "map_name", "team1_map_score", "team2_map_score"],
        "match_maps",
    )
    df = maps.copy()
    numeric = [
        "match_id",
        "map_no",
        "team1_map_score",
        "team2_map_score",
        "mapstatsid",
        "team1_ct_rounds",
        "team1_t_rounds",
        "team2_ct_rounds",
        "team2_t_rounds",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["map_name"] = df["map_name"].map(normalize_map_name).replace("", pd.NA)
    for column in [
        "team1_ct_rounds",
        "team1_t_rounds",
        "team2_ct_rounds",
        "team2_t_rounds",
    ]:
        if column not in df.columns:
            df[column] = np.nan
    if "played" in df.columns:
        played = df["played"].astype("string").str.lower().isin(["true", "1", "yes"])
        played |= df["played"].eq(1)
        # Missing marker means that the old scraper did not expose this flag.
        df = df[played | df["played"].isna()].copy()
    if "is_default_forfeit_map" in df.columns:
        default_forfeit = (
            df["is_default_forfeit_map"]
            .astype("string")
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[~default_forfeit].copy()
    df = df[~df["map_name"].eq("default")].copy()
    match_info = matches[["match_id", "match_datetime_utc", "team1_id", "team2_id"]]
    df = df.merge(match_info, on="match_id", how="inner")
    df = df.dropna(subset=["match_id", "map_name", "team1_id", "team2_id"])
    return df.drop_duplicates(["match_id", "map_no", "map_name"]).reset_index(drop=True)


def normalize_player_stats(player_stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        player_stats,
        ["match_id", "team_id", "player_id", "rating", "adr", "kast"],
        "map_player_stats",
    )
    df = player_stats.copy()
    numeric = [
        "match_id",
        "map_no",
        "mapstatsid",
        "team_id",
        "player_id",
        "rating",
        "adr",
        "kast",
        "opening_kills",
        "opening_deaths",
    ]
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan
    df = df.merge(matches[["match_id", "match_datetime_utc"]], on="match_id", how="inner")
    return df.dropna(subset=["match_id", "team_id", "player_id"]).reset_index(drop=True)


def normalize_veto(veto: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(veto, ["match_id", "action", "map_name"], "veto_steps")
    df = veto.copy()
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    if "step_number" in df.columns:
        df["step_number"] = pd.to_numeric(df["step_number"], errors="coerce")
    df["action"] = df["action"].astype("string").str.strip().str.lower()
    df["map_name"] = df["map_name"].map(normalize_map_name).replace("", pd.NA)
    df = df[df["action"].isin(["picked", "removed", "left_over"])].copy()

    if "team_name" in df.columns and "team1" in matches.columns and "team2" in matches.columns:
        team_lookup = pd.concat(
            [
                matches[["match_id", "team1_id", "team1"]].rename(
                    columns={"team1_id": "team_id", "team1": "team_name"}
                ),
                matches[["match_id", "team2_id", "team2"]].rename(
                    columns={"team2_id": "team_id", "team2": "team_name"}
                ),
            ],
            ignore_index=True,
        )
        team_lookup["team_name_key"] = (
            team_lookup["team_name"].astype("string").str.strip().str.casefold()
        )
        df["team_name_key"] = df["team_name"].astype("string").str.strip().str.casefold()
        df = df.merge(
            team_lookup[["match_id", "team_id", "team_name_key"]],
            on=["match_id", "team_name_key"],
            how="left",
        )
    else:
        df["team_id"] = np.nan
    df = df.merge(matches[["match_id", "match_datetime_utc"]], on="match_id", how="inner")
    return df.dropna(subset=["match_id", "map_name"]).reset_index(drop=True)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def compute_team_history_features(
    matches: pd.DataFrame,
    *,
    elo_k: float = 48.0,
    elo_base: float = 1500.0,
) -> pd.DataFrame:
    """Return two pre-match state rows per match.

    Matches sharing the same timestamp are snapshotted as one batch. Outcomes
    from that batch are applied only after every snapshot has been recorded.
    """

    elo: dict[int, float] = defaultdict(lambda: float(elo_base))
    played_dates: dict[int, list[pd.Timestamp]] = defaultdict(list)
    total_wins: dict[int, int] = defaultdict(int)
    recent_results: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=20))
    streak: dict[int, int] = defaultdict(int)
    opponent_elo: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    h2h_winners: dict[tuple[int, int], list[int]] = defaultdict(list)
    rows: list[dict[str, float | int]] = []

    def snapshot(match: object, team_id: int, opponent_id: int) -> dict[str, float | int]:
        dt = match.match_datetime_utc
        dates = played_dates[team_id]
        recent = list(recent_results[team_id])
        pair_history = h2h_winners[tuple(sorted((team_id, opponent_id)))]
        current_streak = streak[team_id]
        return {
            "match_id": int(match.match_id),
            "team_id": team_id,
            "elo_pre": elo[team_id],
            "opp_elo_pre": elo[opponent_id],
            "elo_diff_pre": elo[team_id] - elo[opponent_id],
            "matches_before": len(dates),
            "days_since_last_match": (
                (dt - dates[-1]).total_seconds() / 86400.0 if dates else np.nan
            ),
            "activity_7d": sum((dt - past).total_seconds() <= 7 * 86400 for past in dates),
            "activity_30d": sum((dt - past).total_seconds() <= 30 * 86400 for past in dates),
            "activity_90d": sum((dt - past).total_seconds() <= 90 * 86400 for past in dates),
            "overall_winrate": (
                float(total_wins[team_id]) / float(len(dates)) if dates else np.nan
            ),
            "winrate_last_5": _mean(recent[-5:]),
            "winrate_last_10": _mean(recent[-10:]),
            "winrate_last_20": _mean(recent[-20:]),
            "win_streak": max(current_streak, 0),
            "loss_streak": max(-current_streak, 0),
            "avg_opp_elo_last_10": _mean(opponent_elo[team_id]),
            "h2h_wins_all": sum(winner == team_id for winner in pair_history),
            "h2h_wins_last5": sum(winner == team_id for winner in pair_history[-5:]),
        }

    for _, batch in matches.groupby("match_datetime_utc", sort=False):
        batch_records = list(batch.itertuples(index=False))
        for match in batch_records:
            team1_id = int(match.team1_id)
            team2_id = int(match.team2_id)
            rows.append(snapshot(match, team1_id, team2_id))
            rows.append(snapshot(match, team2_id, team1_id))

        rating_deltas: dict[int, float] = defaultdict(float)
        for match in batch_records:
            if pd.isna(match.team1_win) or int(match.team1_win) not in (0, 1):
                continue
            team1_id = int(match.team1_id)
            team2_id = int(match.team2_id)
            outcome1 = int(match.team1_win)
            outcome2 = 1 - outcome1
            rating1 = elo[team1_id]
            rating2 = elo[team2_id]
            probability1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
            probability2 = 1.0 - probability1
            rating_deltas[team1_id] += elo_k * (outcome1 - probability1)
            rating_deltas[team2_id] += elo_k * (outcome2 - probability2)

            for team_id, result, opponent_rating in [
                (team1_id, outcome1, rating2),
                (team2_id, outcome2, rating1),
            ]:
                played_dates[team_id].append(match.match_datetime_utc)
                total_wins[team_id] += int(result)
                recent_results[team_id].append(result)
                opponent_elo[team_id].append(opponent_rating)
                if result:
                    streak[team_id] = max(0, streak[team_id]) + 1
                else:
                    streak[team_id] = min(0, streak[team_id]) - 1
            winner = team1_id if outcome1 else team2_id
            h2h_winners[tuple(sorted((team1_id, team2_id)))].append(winner)
        for team_id, delta in rating_deltas.items():
            elo[team_id] += delta

    return pd.DataFrame(rows)


def _lineup_lookup(lineups: pd.DataFrame) -> dict[tuple[int, int], tuple[int, ...]]:
    return {
        (int(match_id), int(team_id)): tuple(sorted(set(group["player_id"].astype(int))))
        for (match_id, team_id), group in lineups.groupby(["match_id", "team_id"], sort=False)
    }


def build_roster_features(matches: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    lookup = _lineup_lookup(lineups)
    previous: dict[int, set[int]] = defaultdict(set)
    rows: list[dict[str, float | int]] = []
    for _, batch in matches.groupby("match_datetime_utc", sort=False):
        updates: list[tuple[int, set[int]]] = []
        for match in batch.itertuples(index=False):
            for team_id in [int(match.team1_id), int(match.team2_id)]:
                roster = set(lookup.get((int(match.match_id), team_id), ()))
                old = previous[team_id]
                overlap = len(roster & old)
                rows.append(
                    {
                        "match_id": int(match.match_id),
                        "team_id": team_id,
                        "roster_size": len(roster),
                        "roster_overlap_prev": overlap,
                        "roster_overlap_prev_ratio": overlap / len(roster) if roster else np.nan,
                    }
                )
                if roster:
                    updates.append((team_id, roster))
        for team_id, roster in updates:
            previous[team_id] = roster
    return pd.DataFrame(rows)


def build_player_features(
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    *,
    min_history_maps: int = 5,
) -> pd.DataFrame:
    lineup = _lineup_lookup(lineups)
    stat_indices = player_stats.groupby("match_id", sort=False).indices
    arrays = {
        column: player_stats[column].to_numpy()
        for column in [
            "player_id",
            "rating",
            "adr",
            "kast",
            "opening_kills",
            "opening_deaths",
        ]
    }
    state: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "maps": 0.0,
            "rating_sum": 0.0,
            "rating_n": 0.0,
            "adr_sum": 0.0,
            "adr_n": 0.0,
            "kast_sum": 0.0,
            "kast_n": 0.0,
            "opening_sum": 0.0,
            "opening_n": 0.0,
        }
    )
    rows: list[dict[str, float | int]] = []

    def snapshot(match_id: int, team_id: int) -> dict[str, float]:
        players = lineup.get((match_id, team_id), ())
        known = [player for player in players if state[player]["maps"] >= min_history_maps]

        def player_average(sum_key: str, count_key: str) -> float:
            return _mean(
                state[player][sum_key] / state[player][count_key]
                for player in known
                if state[player][count_key] > 0
            )

        return {
            "lineup_history_coverage": (
                float(len(known)) / float(len(players)) if players else np.nan
            ),
            "lineup_players_with_history": float(len(known)),
            "lineup_player_rating_mean": player_average("rating_sum", "rating_n"),
            "lineup_player_adr_mean": player_average("adr_sum", "adr_n"),
            "lineup_player_kast_mean": player_average("kast_sum", "kast_n"),
            "lineup_player_opening_diff_mean": player_average("opening_sum", "opening_n"),
            "lineup_player_maps_played_mean": _mean(state[player]["maps"] for player in known),
        }

    for _, batch in matches.groupby("match_datetime_utc", sort=False):
        match_ids: list[int] = []
        for match in batch.itertuples(index=False):
            match_id = int(match.match_id)
            match_ids.append(match_id)
            for team_id in [int(match.team1_id), int(match.team2_id)]:
                rows.append({"match_id": match_id, "team_id": team_id, **snapshot(match_id, team_id)})

        for match_id in match_ids:
            indices = stat_indices.get(match_id)
            if indices is None:
                continue
            for index in indices:
                player = state[int(arrays["player_id"][index])]
                player["maps"] += 1
                for value_column, sum_key, count_key in [
                    ("rating", "rating_sum", "rating_n"),
                    ("adr", "adr_sum", "adr_n"),
                    ("kast", "kast_sum", "kast_n"),
                ]:
                    value = arrays[value_column][index]
                    if pd.notna(value) and np.isfinite(value):
                        player[sum_key] += float(value)
                        player[count_key] += 1
                kills = arrays["opening_kills"][index]
                deaths = arrays["opening_deaths"][index]
                if pd.notna(kills) or pd.notna(deaths):
                    player["opening_sum"] += float(0 if pd.isna(kills) else kills)
                    player["opening_sum"] -= float(0 if pd.isna(deaths) else deaths)
                    player["opening_n"] += 1
    return pd.DataFrame(rows)


def build_map_features(
    matches: pd.DataFrame,
    maps: pd.DataFrame,
    veto: pd.DataFrame,
    *,
    map_pool_overrides: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    map_indices = maps.groupby("match_id", sort=False).indices
    selected_veto = veto[veto["action"].isin(["picked", "left_over"])].copy()
    selected_map_indices = selected_veto.groupby("match_id", sort=False).indices
    history: dict[tuple[int, str], list[tuple[int, float, float]]] = defaultdict(list)
    rows: list[dict[str, float | int]] = []

    def snapshot(team_id: int, selected_maps: list[str]) -> dict[str, float]:
        counts: list[float] = []
        winrates: list[float] = []
        ct_rates: list[float] = []
        t_rates: list[float] = []
        known_maps = 0
        for map_name in selected_maps:
            past = history[(team_id, map_name)]
            counts.append(float(len(past)))
            if past:
                known_maps += 1
                winrates.append(_mean(item[0] for item in past))
                ct_rates.append(_mean(item[1] for item in past))
                t_rates.append(_mean(item[2] for item in past))
        return {
            "avg_map_count_before": _mean(counts),
            "avg_map_wr_before": _mean(winrates),
            "avg_map_ct_wr_before": _mean(ct_rates),
            "avg_map_t_wr_before": _mean(t_rates),
            "series_maps_known": float(known_maps),
        }

    for _, batch in matches.groupby("match_datetime_utc", sort=False):
        updates: list[tuple[int, str, int, float, float]] = []
        for match in batch.itertuples(index=False):
            match_id = int(match.match_id)
            indices = map_indices.get(match_id)
            current = maps.iloc[indices] if indices is not None else maps.iloc[0:0]
            veto_indices = selected_map_indices.get(match_id)
            if map_pool_overrides is not None and match_id in map_pool_overrides:
                selected_maps = list(dict.fromkeys(map_pool_overrides[match_id]))
            else:
                selected_maps = (
                    selected_veto.iloc[veto_indices]["map_name"]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                    if veto_indices is not None
                    else []
                )
            for team_id in [int(match.team1_id), int(match.team2_id)]:
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_id,
                        **snapshot(team_id, selected_maps),
                    }
                )

            for map_row in current.itertuples(index=False):
                if pd.isna(map_row.team1_map_score) or pd.isna(map_row.team2_map_score):
                    continue
                if float(map_row.team1_map_score) == float(map_row.team2_map_score):
                    continue
                map_name = str(map_row.map_name)
                result1 = int(map_row.team1_map_score > map_row.team2_map_score)
                result2 = 1 - result1

                def side_rate(won: object, lost: object) -> float:
                    if pd.isna(won) or pd.isna(lost):
                        return np.nan
                    rounds = float(won) + float(lost)
                    return float(won) / rounds if rounds > 0 else np.nan

                # On a given side, one team's won rounds are the opponent's lost
                # rounds. Overtime is absent from the scraped half split and is
                # therefore deliberately excluded from these denominators.
                ct1 = side_rate(map_row.team1_ct_rounds, map_row.team2_t_rounds)
                t1 = side_rate(map_row.team1_t_rounds, map_row.team2_ct_rounds)
                ct2 = side_rate(map_row.team2_ct_rounds, map_row.team1_t_rounds)
                t2 = side_rate(map_row.team2_t_rounds, map_row.team1_ct_rounds)
                updates.append((int(match.team1_id), map_name, result1, ct1, t1))
                updates.append((int(match.team2_id), map_name, result2, ct2, t2))
        for team_id, map_name, result, ct_rate, t_rate in updates:
            history[(team_id, map_name)].append((result, ct_rate, t_rate))
    return pd.DataFrame(rows)


def build_veto_features(matches: pd.DataFrame, veto: pd.DataFrame) -> pd.DataFrame:
    veto_indices = veto.groupby("match_id", sort=False).indices
    history: dict[tuple[int, str], list[str]] = defaultdict(list)
    rows: list[dict[str, float | int]] = []

    def snapshot(team_id: int, current: pd.DataFrame) -> dict[str, float]:
        if "team_id" not in current.columns:
            return {column: np.nan for column in VETO_COLUMNS}
        own = current[current["team_id"].eq(team_id)]
        current_maps = {
            "picked": own.loc[own["action"].eq("picked"), "map_name"].dropna().astype(str),
            "removed": own.loc[own["action"].eq("removed"), "map_name"].dropna().astype(str),
            # The decider belongs to the series rather than one team. It is a
            # pre-match-known map for both participants.
            "left_over": current.loc[
                current["action"].eq("left_over"), "map_name"
            ].dropna().astype(str),
        }
        rates: dict[str, list[float]] = {"picked": [], "removed": [], "left_over": []}
        for action, map_names in current_maps.items():
            for map_name in map_names.drop_duplicates():
                past = history[(team_id, map_name)]
                if past:
                    rates[action].append(float(np.mean([value == action for value in past])))
        return {
            "veto_pick_rate_before": _mean(rates["picked"]),
            "veto_remove_rate_before": _mean(rates["removed"]),
            "veto_leftover_rate_before": _mean(rates["left_over"]),
        }

    for _, batch in matches.groupby("match_datetime_utc", sort=False):
        updates: list[tuple[int, str, str]] = []
        for match in batch.itertuples(index=False):
            match_id = int(match.match_id)
            indices = veto_indices.get(match_id)
            current = veto.iloc[indices] if indices is not None else veto.iloc[0:0]
            for team_id in [int(match.team1_id), int(match.team2_id)]:
                rows.append({"match_id": match_id, "team_id": team_id, **snapshot(team_id, current)})
            if "team_id" in current.columns:
                for row in current.dropna(subset=["team_id"]).itertuples(index=False):
                    updates.append((int(row.team_id), str(row.map_name), str(row.action)))
                for row in current[current["action"].eq("left_over")].itertuples(index=False):
                    for team_id in [int(match.team1_id), int(match.team2_id)]:
                        updates.append((team_id, str(row.map_name), "left_over"))
        for team_id, map_name, action in updates:
            history[(team_id, map_name)].append(action)
    return pd.DataFrame(rows)


def pivot_team_features(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if features.empty:
        return matches[["match_id"]].copy()
    team1 = features.merge(
        matches[["match_id", "team1_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "team1_id"],
        how="inner",
    ).drop(columns=["team_id", "team1_id"])
    team2 = features.merge(
        matches[["match_id", "team2_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "team2_id"],
        how="inner",
    ).drop(columns=["team_id", "team2_id"])
    team1 = team1.rename(columns={column: f"team1_{column}" for column in columns})
    team2 = team2.rename(columns={column: f"team2_{column}" for column in columns})
    return (
        matches[["match_id"]]
        .merge(team1, on="match_id", how="left")
        .merge(team2, on="match_id", how="left")
    )


def add_difference_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    difference_sources = [
        name
        for name in TEAM_HISTORY_COLUMNS + ROSTER_COLUMNS + PLAYER_COLUMNS + MAP_COLUMNS + VETO_COLUMNS
        if name not in {"opp_elo_pre", "elo_diff_pre"}
    ]
    for name in difference_sources:
        column1 = f"team1_{name}"
        column2 = f"team2_{name}"
        if column1 in df.columns and column2 in df.columns:
            df[f"diff_{name}"] = df[column1] - df[column2]
    if {"team1_team_rank", "team2_team_rank"}.issubset(df.columns):
        # A lower numerical rank means a stronger team.
        df["diff_rank"] = df["team2_team_rank"] - df["team1_team_rank"]
    return df


def validate_feature_dataset(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Feature dataset is empty")
    if frame["match_id"].duplicated().any():
        raise ValueError("Feature dataset contains duplicate match_id values")
    if not frame["match_datetime_utc"].is_monotonic_increasing:
        raise ValueError("Feature dataset is not sorted chronologically")
    if "team1_win" in frame.columns:
        invalid = frame["team1_win"].dropna().loc[lambda value: ~value.isin([0, 1])]
        if not invalid.empty:
            raise ValueError("Target team1_win contains values outside {0, 1}")


def build_feature_dataset(
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    veto: pd.DataFrame,
    maps: pd.DataFrame,
    player_stats: pd.DataFrame,
    *,
    elo_k: float = 48.0,
    elo_base: float = 1500.0,
    min_player_history_maps: int = 5,
    include_pending: bool = False,
    map_pool_overrides: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    matches_norm = normalize_matches(matches, include_pending=include_pending)
    valid_ids = set(matches_norm["match_id"].astype(int))
    lineups_norm = normalize_lineups(lineups, valid_ids)
    maps_norm = normalize_maps(maps, matches_norm)
    player_norm = normalize_player_stats(player_stats, matches_norm)
    veto_norm = normalize_veto(veto, matches_norm)

    history = compute_team_history_features(matches_norm, elo_k=elo_k, elo_base=elo_base)
    roster = build_roster_features(matches_norm, lineups_norm)
    players = build_player_features(
        matches_norm,
        lineups_norm,
        player_norm,
        min_history_maps=min_player_history_maps,
    )
    map_features = build_map_features(
        matches_norm,
        maps_norm,
        veto_norm,
        map_pool_overrides=map_pool_overrides,
    )
    veto_features = build_veto_features(matches_norm, veto_norm)

    base = matches_norm.copy()
    base["rank_available"] = (
        base.get("team1_rank", pd.Series(np.nan, index=base.index)).notna()
        & base.get("team2_rank", pd.Series(np.nan, index=base.index)).notna()
    ).astype(int)
    base["bo1"] = base["bo"].eq(1).fillna(False).astype(int)
    base["bo3"] = base["bo"].eq(3).fillna(False).astype(int)
    base["bo5"] = base["bo"].eq(5).fillna(False).astype(int)

    for feature_frame, columns in [
        (history, TEAM_HISTORY_COLUMNS),
        (roster, ROSTER_COLUMNS),
        (players, PLAYER_COLUMNS),
        (map_features, MAP_COLUMNS),
        (veto_features, VETO_COLUMNS),
    ]:
        base = base.merge(
            pivot_team_features(matches_norm, feature_frame, columns),
            on="match_id",
            how="left",
        )

    base = base.rename(
        columns={"team1_rank": "team1_team_rank", "team2_rank": "team2_team_rank"}
    )
    result = add_difference_features(base)
    result = result.sort_values(["match_datetime_utc", "match_id"]).reset_index(drop=True)
    validate_feature_dataset(result)
    return result


def build_feature_dataset_from_dir(
    clean_dir: Path,
    *,
    elo_k: float = 48.0,
    elo_base: float = 1500.0,
    min_player_history_maps: int = 5,
) -> pd.DataFrame:
    return build_feature_dataset(
        *read_clean_tables(clean_dir),
        elo_k=elo_k,
        elo_base=elo_base,
        min_player_history_maps=min_player_history_maps,
    )
