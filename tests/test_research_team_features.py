from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_team_features import EXTRA_TEAM_FEATURES, build_research_team_features


def matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {"team1_id": 1, "team2_id": 2, "team1_win": 1, "is_valid_result": 1}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def lineups(rosters: dict[tuple[int, int], list[int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": match, "team_id": team, "player_id": player}
            for (match, team), roster in rosters.items()
            for player in roster
        ],
        columns=["match_id", "team_id", "player_id"],
    )


def ratings(match: int, roster: list[int], values: list[float], maps_count: int = 5) -> list[dict[str, object]]:
    return [
        {"match_id": match, "team_id": 1, "player_id": player, "map_no": map_no, "rating": value}
        for player, value in zip(roster, values)
        for map_no in range(1, maps_count + 1)
    ]


def empty_stats() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", "team_id", "player_id", "rating"])


def test_distribution_uses_all_five_prior_player_means_and_population_std() -> None:
    frame = matches([
        {"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    roster = [1, 2, 3, 4, 5]
    source = lineups({(1, 1): roster, (2, 1): roster})
    values = [1.5, 1.1, 1.0, 0.9, 0.8]
    result = build_research_team_features(frame, source, pd.DataFrame(ratings(1, roster, values)))
    before, after = result.iloc[0], result.iloc[1]
    assert np.isnan(before.team1_lineup_player_rating_max)
    assert after.team1_lineup_player_rating_max == pytest.approx(1.5)
    assert after.team1_lineup_player_rating_min == pytest.approx(0.8)
    assert after.team1_lineup_player_rating_std == pytest.approx(np.std(values, ddof=0))
    assert after.team1_roster_same_matches_90d == 1
    assert after.team1_roster_consecutive_before == 1
    assert after.team1_roster_pair_experience_90d == 1


def test_current_and_future_ratings_cannot_change_current_snapshot() -> None:
    frame = matches([
        {"match_id": number, "match_datetime_utc": f"2025-01-0{number}T12:00:00Z"}
        for number in (1, 2, 3)
    ])
    roster = [1, 2, 3, 4, 5]
    source = lineups({(number, 1): roster for number in (1, 2, 3)})
    stats = pd.DataFrame(sum((ratings(number, roster, [1.0] * 5) for number in (1, 2, 3)), []))
    baseline = build_research_team_features(frame, source, stats)
    changed = stats.copy()
    changed.loc[changed.match_id.ge(2), "rating"] = 99.0
    alternative = build_research_team_features(frame, source, changed)
    pd.testing.assert_frame_equal(baseline.iloc[:2], alternative.iloc[:2])
    assert alternative.iloc[2].team1_lineup_player_rating_max > baseline.iloc[2].team1_lineup_player_rating_max


def test_same_timestamp_is_one_snapshot_batch_not_sequential_updates() -> None:
    frame = matches([
        {"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 3, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    roster = [1, 2, 3, 4, 5]
    source = lineups({(number, 1): roster for number in (1, 2, 3)})
    stats = pd.DataFrame(ratings(1, roster, [1.0] * 5) + ratings(2, roster, [2.0] * 5))
    result = build_research_team_features(frame, source, stats)
    assert result.iloc[:2].team1_lineup_player_rating_max.isna().all()
    assert result.iloc[:2].team1_roster_same_matches_90d.eq(0).all()
    assert result.iloc[:2].team1_roster_pair_experience_90d.eq(0).all()
    assert result.iloc[:2].team1_roster_consecutive_before.eq(0).all()
    assert result.iloc[2].team1_lineup_player_rating_max == pytest.approx(1.5)
    assert result.iloc[2].team1_roster_same_matches_90d == 2
    assert result.iloc[2].team1_roster_pair_experience_90d == 2
    assert result.iloc[2].team1_roster_consecutive_before == 2
    # Порядок строк не должен передавать историю между матчами с одинаковым временем.
    shuffled = build_research_team_features(frame.iloc[::-1], source.iloc[::-1], stats.iloc[::-1])
    pd.testing.assert_frame_equal(result, shuffled)


def test_partial_history_does_not_turn_one_known_player_into_a_star() -> None:
    frame = matches([
        {"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    roster = [1, 2, 3, 4, 5]
    source = lineups({(1, 1): roster, (2, 1): roster})
    stats = pd.DataFrame(ratings(1, [1], [2.5]))
    result = build_research_team_features(frame, source, stats)
    assert result.iloc[1][[f"team1_lineup_player_rating_{name}" for name in ("max", "min", "std")]].isna().all()
    assert result.iloc[1].team1_roster_same_matches_90d == 1


@pytest.mark.parametrize("roster", [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
def test_invalid_current_roster_has_no_six_measurements(roster: list[int]) -> None:
    frame = matches([{"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"}])
    result = build_research_team_features(frame, lineups({(1, 1): roster}), empty_stats())
    assert result.iloc[0][[f"team1_{name}" for name in EXTRA_TEAM_FEATURES]].isna().all()


def test_pair_experience_follows_players_between_teams_and_exact_roster_is_team_specific() -> None:
    roster = [1, 2, 3, 4, 5]
    frame = matches([
        {"match_id": 1, "team1_id": 10, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "team1_id": 20, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    result = build_research_team_features(frame, lineups({(1, 10): roster, (2, 20): roster}), empty_stats())
    current = result.iloc[1]
    assert current.team1_roster_pair_experience_90d == 1
    assert current.team1_roster_same_matches_90d == 0
    assert current.team1_roster_consecutive_before == 0


def test_changed_roster_has_six_retained_pairs_and_resets_streak() -> None:
    old, changed = [1, 2, 3, 4, 5], [1, 2, 3, 4, 6]
    frame = matches([
        {"match_id": number, "match_datetime_utc": f"2025-01-0{number}T12:00:00Z"}
        for number in (1, 2, 3, 4)
    ])
    source = lineups({(1, 1): old, (2, 1): old, (3, 1): changed, (4, 1): changed})
    result = build_research_team_features(frame, source, empty_stats())
    before_change = result.iloc[2]
    assert before_change.team1_roster_same_matches_90d == 0
    assert before_change.team1_roster_consecutive_before == 0
    assert before_change.team1_roster_pair_experience_90d == pytest.approx(6 * 2 / 10)
    after_change = result.iloc[3]
    assert after_change.team1_roster_same_matches_90d == 1
    assert after_change.team1_roster_consecutive_before == 1
    assert after_change.team1_roster_pair_experience_90d == pytest.approx((6 * 3 + 4) / 10)


def test_90_day_window_is_lower_inclusive_and_streak_has_no_window_limit() -> None:
    roster = [1, 2, 3, 4, 5]
    frame = matches([
        {"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-04-01T12:00:00Z"},
        {"match_id": 3, "match_datetime_utc": "2025-04-01T12:00:01Z"},
    ])
    result = build_research_team_features(frame, lineups({(number, 1): roster for number in (1, 2, 3)}), empty_stats())
    assert result.iloc[1].team1_roster_same_matches_90d == 1
    assert result.iloc[1].team1_roster_pair_experience_90d == 1
    assert result.iloc[2].team1_roster_same_matches_90d == 1  # Jan 1 expired; Apr 1 remains.
    assert result.iloc[2].team1_roster_pair_experience_90d == 1
    assert result.iloc[2].team1_roster_consecutive_before == 2


def test_pending_series_do_not_update_player_or_joint_history() -> None:
    roster = [1, 2, 3, 4, 5]
    frame = matches([
        {"match_id": 1, "team1_win": np.nan, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    result = build_research_team_features(frame, lineups({(1, 1): roster, (2, 1): roster}), pd.DataFrame(ratings(1, roster, [2.0] * 5)))
    assert np.isnan(result.iloc[1].team1_lineup_player_rating_max)
    assert result.iloc[1].team1_roster_same_matches_90d == 0
    assert result.iloc[1].team1_roster_pair_experience_90d == 0


def test_observed_partial_old_roster_breaks_consecutive_full_roster_streak() -> None:
    roster = [1, 2, 3, 4, 5]
    frame = matches([
        {"match_id": number, "match_datetime_utc": f"2025-01-0{number}T12:00:00Z"}
        for number in (1, 2, 3)
    ])
    result = build_research_team_features(frame, lineups({(1, 1): roster, (2, 1): roster[:4], (3, 1): roster}), empty_stats())
    assert result.iloc[2].team1_roster_consecutive_before == 0
    assert result.iloc[2].team1_roster_same_matches_90d == 1
    assert result.iloc[2].team1_roster_pair_experience_90d == pytest.approx(1.6)


def test_different_same_time_rosters_make_following_streak_ambiguous() -> None:
    first, second = [1, 2, 3, 4, 5], [1, 2, 3, 4, 6]
    frame = matches([
        {"match_id": 1, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 2, "match_datetime_utc": "2025-01-01T12:00:00Z"},
        {"match_id": 3, "match_datetime_utc": "2025-01-02T12:00:00Z"},
    ])
    result = build_research_team_features(frame, lineups({(1, 1): first, (2, 1): second, (3, 1): first}), empty_stats())
    assert np.isnan(result.iloc[2].team1_roster_consecutive_before)
    assert result.iloc[2].team1_roster_same_matches_90d == 1


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_minimum_fails_before_feature_generation(minimum: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_research_team_features(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), min_history_maps=minimum)
