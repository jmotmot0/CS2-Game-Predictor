"""Additional historical player-distribution and joint-lineup measurements.

The input chronology is the same as ``feature_engineering.normalize_matches``.
Every match at time t is snapshotted before any observations from time t are
applied. As elsewhere in this project, this uses recorded match start times,
not independently archived publication or completion times.

Player Rating means have exactly the existing player's-history semantics:
one player-statistics row counts as one past map; the mean uses its finite
Rating values, and a player is eligible after ``min_history_maps`` maps.
Unlike the existing partially covered lineup mean, max/min/std require all
five current players to be eligible and to have a finite historical mean.

The 90-day window is [t - 90 days, t). Exact-five and consecutive counts are
team-specific. Pair experience follows player identities globally, including
previous joint series under another team ID. One series contributes once to
each observed pair, regardless of the number of maps. These measurements
describe observed joint experience, not communication or a causal team effect.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from src.feature_engineering import normalize_lineups, normalize_matches


EXTRA_INDIVIDUAL_FEATURES = (
    "lineup_player_rating_max",
    "lineup_player_rating_min",
    "lineup_player_rating_std",
)
EXTRA_COHESION_FEATURES = (
    "roster_same_matches_90d",
    "roster_consecutive_before",
    "roster_pair_experience_90d",
)
EXTRA_TEAM_FEATURES = EXTRA_INDIVIDUAL_FEATURES + EXTRA_COHESION_FEATURES


def _full_roster(players: Iterable[int]) -> tuple[int, ...] | None:
    roster = tuple(sorted(set(players)))
    return roster if len(roster) == 5 else None


def build_research_team_features(
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    player_stats: pd.DataFrame,
    min_history_maps: int = 5,
) -> pd.DataFrame:
    """Return match_id and six additional columns for each team perspective.

    Inputs may be clean tables or the historical feature dataset for matches.
    Pending matches can be snapshotted but do not update past observations.
    Incomplete current lineups produce NaN for all six measurements. Missing
    historical Rating produces NaN for its three distribution measurements,
    not for independently observable joint-lineup counts.

    Consecutive counts break at an observed incomplete roster. If one team
    appears with different rosters at one identical timestamp, the order is
    unknowable: the subsequent streak is NaN until a known roster change.
    Same-timestamp identical rosters count all observed series after the batch.
    """
    if (
        not isinstance(min_history_maps, (int, np.integer))
        or isinstance(min_history_maps, (bool, np.bool_))
        or min_history_maps < 1
    ):
        raise ValueError("min_history_maps must be a positive integer")
    required_stats = {"match_id", "team_id", "player_id", "rating"}
    missing_stats = required_stats.difference(player_stats.columns)
    if missing_stats:
        raise ValueError(f"player_stats lacks columns: {sorted(missing_stats)}")

    chronology = normalize_matches(matches, include_pending=True)
    valid_match_ids = set(chronology["match_id"].astype(int))
    normalized_lineups = normalize_lineups(lineups, valid_match_ids)
    lineup_lookup = {
        (int(match_id), int(team_id)): tuple(sorted(set(group["player_id"].astype(int))))
        for (match_id, team_id), group in normalized_lineups.groupby(
            ["match_id", "team_id"], sort=False
        )
    }
    stats = player_stats.copy()
    for column in required_stats:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    stats = stats.dropna(subset=["match_id", "team_id", "player_id"])
    stats = stats[stats["match_id"].isin(valid_match_ids)]
    stat_lookup = {
        int(match_id): tuple(zip(group["player_id"].astype(int), group["rating"]))
        for match_id, group in stats.groupby("match_id", sort=False)
    }

    # [number of observed maps, finite rating sum, number of finite ratings].
    player_history: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    team_history: dict[int, deque[tuple[pd.Timestamp, tuple[int, ...] | None]]] = defaultdict(deque)
    pair_history: dict[tuple[int, int], deque[pd.Timestamp]] = defaultdict(deque)
    last_roster: dict[int, tuple[int, ...] | None] = {}
    streak: dict[int, float] = defaultdict(float)
    ambiguous_last: set[int] = set()
    rows: list[dict[str, int | float]] = []

    def snapshot(timestamp: pd.Timestamp, team_id: int, players: tuple[int, ...]) -> dict[str, float]:
        values = {name: np.nan for name in EXTRA_TEAM_FEATURES}
        roster = _full_roster(players)
        if roster is None:
            return values
        ratings: list[float] = []
        for player_id in roster:
            maps, total, count = player_history[player_id]
            if maps < min_history_maps or count <= 0:
                break
            ratings.append(total / count)
        if len(ratings) == 5 and np.isfinite(ratings).all():
            values.update(
                lineup_player_rating_max=float(np.max(ratings)),
                lineup_player_rating_min=float(np.min(ratings)),
                lineup_player_rating_std=float(np.std(ratings, ddof=0)),
            )

        cutoff = timestamp - pd.Timedelta(days=90)
        past_rosters = team_history[team_id]
        while past_rosters and past_rosters[0][0] < cutoff:
            past_rosters.popleft()
        values["roster_same_matches_90d"] = float(
            sum(past_roster == roster for _, past_roster in past_rosters)
        )
        if team_id in ambiguous_last:
            values["roster_consecutive_before"] = np.nan
        else:
            values["roster_consecutive_before"] = (
                streak[team_id] if last_roster.get(team_id) == roster else 0.0
            )
        pair_counts: list[int] = []
        for pair in combinations(roster, 2):
            previous = pair_history[pair]
            while previous and previous[0] < cutoff:
                previous.popleft()
            pair_counts.append(len(previous))
        values["roster_pair_experience_90d"] = float(np.mean(pair_counts))
        return values

    for timestamp, batch in chronology.groupby("match_datetime_utc", sort=False):
        # Stage 1: every row sees precisely the same pre-batch history.
        for match in batch.itertuples(index=False):
            match_id = int(match.match_id)
            row: dict[str, int | float] = {"match_id": match_id}
            for ordinal, team_id in enumerate((int(match.team1_id), int(match.team2_id)), 1):
                values = snapshot(timestamp, team_id, lineup_lookup.get((match_id, team_id), ()))
                row.update({f"team{ordinal}_{name}": value for name, value in values.items()})
            rows.append(row)

        # Stage 2: completed series become observations for strictly later times.
        updates: dict[int, list[tuple[int, ...] | None]] = defaultdict(list)
        for match in batch.itertuples(index=False):
            outcome = match.team1_win
            is_valid = getattr(match, "is_valid_result", 1)
            if pd.isna(outcome) or outcome not in (0, 1) or is_valid != 1:
                continue
            match_id = int(match.match_id)
            observed_pairs: set[tuple[int, int]] = set()
            for team_id in (int(match.team1_id), int(match.team2_id)):
                players = lineup_lookup.get((match_id, team_id), ())
                roster = _full_roster(players)
                team_history[team_id].append((timestamp, roster))
                updates[team_id].append(roster)
                # Observed pairs in a partial old roster remain observable pairs.
                observed_pairs.update(combinations(players, 2))
            for pair in observed_pairs:
                pair_history[pair].append(timestamp)
            for player_id, rating in stat_lookup.get(match_id, ()):
                state = player_history[player_id]
                state[0] += 1.0
                if pd.notna(rating) and np.isfinite(rating):
                    state[1] += float(rating)
                    state[2] += 1.0

        for team_id, rosters in updates.items():
            different = set(rosters)
            if len(different) > 1:
                last_roster[team_id] = None
                streak[team_id] = np.nan
                ambiguous_last.add(team_id)
                continue
            roster = rosters[0]
            if roster is None:
                last_roster[team_id] = None
                streak[team_id] = 0.0
                ambiguous_last.discard(team_id)
                continue
            if team_id in ambiguous_last:
                # At least these series are known, but their predecessor is not.
                streak[team_id] = np.nan
                ambiguous_last.discard(team_id)
            elif last_roster.get(team_id) == roster:
                streak[team_id] += len(rosters)
            else:
                streak[team_id] = float(len(rosters))
            last_roster[team_id] = roster

    columns = ["match_id"] + [
        f"team{ordinal}_{name}" for ordinal in (1, 2) for name in EXTRA_TEAM_FEATURES
    ]
    return pd.DataFrame(rows, columns=columns)
