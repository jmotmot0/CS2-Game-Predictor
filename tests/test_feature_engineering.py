from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    build_feature_dataset,
    build_map_features,
    build_veto_features,
    compute_team_history_features,
    normalize_matches,
)


def match_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "team1": "Team 1",
        "team2": "Team 2",
        "team1_rank": 10,
        "team2_rank": 20,
        "team1_score": 2,
        "team2_score": 1,
        "bo": "bo3",
        "lan_online": "lan",
        "is_valid_result": 1,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def empty_lineups() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", "team_id", "player_id"])


def empty_veto() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", "step_number", "team_name", "action", "map_name"])


def empty_maps() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "map_no",
            "map_name",
            "team1_map_score",
            "team2_map_score",
            "team1_ct_rounds",
            "team1_t_rounds",
            "team2_ct_rounds",
            "team2_t_rounds",
        ]
    )


def empty_player_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "team_id",
            "player_id",
            "rating",
            "adr",
            "kast",
            "opening_kills",
            "opening_deaths",
        ]
    )


def test_both_teams_are_snapshotted_before_current_result() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 20,
                    "team2_id": 10,
                    "team1_win": 1,
                }
            ]
        )
    )

    history = compute_team_history_features(matches, elo_k=48)
    assert history["elo_pre"].tolist() == [1500.0, 1500.0]
    assert history["opp_elo_pre"].tolist() == [1500.0, 1500.0]
    assert history["h2h_wins_all"].tolist() == [0, 0]


def test_overall_winrate_uses_full_history_not_last_twenty() -> None:
    dates = pd.date_range("2025-01-01", periods=22, freq="D", tz="UTC")
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": index + 1,
                    "match_date": timestamp.date().isoformat(),
                    "match_datetime_utc": timestamp.isoformat(),
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 0 if index == 0 else 1,
                }
                for index, timestamp in enumerate(dates)
            ]
        )
    )

    history = compute_team_history_features(matches)
    current = history[(history["match_id"].eq(22)) & (history["team_id"].eq(1))].iloc[0]

    assert current["matches_before"] == 21
    assert current["overall_winrate"] == pytest.approx(20 / 21)
    assert current["winrate_last_20"] == pytest.approx(1.0)


def test_changing_outcome_does_not_change_current_or_past_features() -> None:
    source = match_frame(
        [
            {
                "match_id": 1,
                "match_date": "2025-01-01",
                "match_datetime_utc": "2025-01-01T12:00:00Z",
                "team1_id": 1,
                "team2_id": 2,
                "team1_win": 1,
            },
            {
                "match_id": 2,
                "match_date": "2025-01-02",
                "match_datetime_utc": "2025-01-02T12:00:00Z",
                "team1_id": 1,
                "team2_id": 2,
                "team1_win": 1,
            },
            {
                "match_id": 3,
                "match_date": "2025-01-03",
                "match_datetime_utc": "2025-01-03T12:00:00Z",
                "team1_id": 1,
                "team2_id": 2,
                "team1_win": 0,
            },
        ]
    )
    flipped = source.copy()
    flipped.loc[flipped["match_id"].eq(2), "team1_win"] = 0

    original_features = build_feature_dataset(
        source,
        empty_lineups(),
        empty_veto(),
        empty_maps(),
        empty_player_stats(),
    ).set_index("match_id")
    flipped_features = build_feature_dataset(
        flipped,
        empty_lineups(),
        empty_veto(),
        empty_maps(),
        empty_player_stats(),
    ).set_index("match_id")

    checked = ["diff_elo_pre", "diff_h2h_wins_all", "diff_winrate_last_5"]
    pd.testing.assert_frame_equal(
        original_features.loc[[1, 2], checked],
        flipped_features.loc[[1, 2], checked],
    )
    assert original_features.loc[3, "diff_elo_pre"] != flipped_features.loc[3, "diff_elo_pre"]


def test_equal_timestamps_are_one_snapshot_batch() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                },
                {
                    "match_id": 2,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 3,
                    "team1_win": 0,
                },
            ]
        )
    )
    history = compute_team_history_features(matches)
    team1_rows = history[history["team_id"].eq(1)]
    assert team1_rows["matches_before"].tolist() == [0, 0]
    assert team1_rows["elo_pre"].tolist() == [1500.0, 1500.0]


def test_maps_of_current_series_do_not_update_its_snapshot() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                },
                {
                    "match_id": 2,
                    "match_date": "2025-01-02",
                    "match_datetime_utc": "2025-01-02T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 0,
                },
            ]
        )
    )
    maps = pd.DataFrame(
        [
            {
                "match_id": 1,
                "map_no": 1,
                "map_name": "mirage",
                "team1_map_score": 13,
                "team2_map_score": 8,
                "team1_ct_rounds": 7,
                "team1_t_rounds": 6,
                "team2_ct_rounds": 4,
                "team2_t_rounds": 4,
            },
            {
                "match_id": 1,
                "map_no": 2,
                "map_name": "mirage",
                "team1_map_score": 13,
                "team2_map_score": 10,
                "team1_ct_rounds": 8,
                "team1_t_rounds": 5,
                "team2_ct_rounds": 5,
                "team2_t_rounds": 5,
            },
            {
                "match_id": 2,
                "map_no": 1,
                "map_name": "mirage",
                "team1_map_score": 8,
                "team2_map_score": 13,
                "team1_ct_rounds": 4,
                "team1_t_rounds": 4,
                "team2_ct_rounds": 7,
                "team2_t_rounds": 6,
            },
        ]
    )
    maps = maps.merge(
        matches[["match_id", "match_datetime_utc", "team1_id", "team2_id"]],
        on="match_id",
        how="left",
    )
    veto = pd.DataFrame(
        [
            {"match_id": 1, "action": "picked", "map_name": "mirage"},
            {"match_id": 2, "action": "picked", "map_name": "mirage"},
        ]
    )
    features = build_map_features(matches, maps, veto)
    first_team = features[(features["match_id"].eq(1)) & (features["team_id"].eq(1))].iloc[0]
    second_team = features[(features["match_id"].eq(2)) & (features["team_id"].eq(1))].iloc[0]
    assert first_team["avg_map_count_before"] == pytest.approx(0.0)
    assert second_team["avg_map_count_before"] == pytest.approx(2.0)
    assert second_team["avg_map_ct_wr_before"] == pytest.approx(
        (7 / (7 + 4) + 8 / (8 + 5)) / 2
    )
    assert second_team["avg_map_t_wr_before"] == pytest.approx(
        (6 / (6 + 4) + 5 / (5 + 5)) / 2
    )


def test_current_map_pool_uses_veto_even_when_decider_was_not_played() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                },
                {
                    "match_id": 2,
                    "match_date": "2025-01-02",
                    "match_datetime_utc": "2025-01-02T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                },
            ]
        )
    )
    maps = pd.DataFrame(
        [
            {
                "match_id": 1,
                "map_no": 1,
                "map_name": "ancient",
                "team1_map_score": 13,
                "team2_map_score": 8,
                "team1_ct_rounds": 7,
                "team1_t_rounds": 6,
                "team2_ct_rounds": 4,
                "team2_t_rounds": 4,
            },
            {
                "match_id": 2,
                "map_no": 1,
                "map_name": "mirage",
                "team1_map_score": 13,
                "team2_map_score": 8,
                "team1_ct_rounds": 7,
                "team1_t_rounds": 6,
                "team2_ct_rounds": 4,
                "team2_t_rounds": 4,
            },
            {
                "match_id": 2,
                "map_no": 2,
                "map_name": "nuke",
                "team1_map_score": 13,
                "team2_map_score": 9,
                "team1_ct_rounds": 7,
                "team1_t_rounds": 6,
                "team2_ct_rounds": 5,
                "team2_t_rounds": 4,
            },
        ]
    ).merge(
        matches[["match_id", "match_datetime_utc", "team1_id", "team2_id"]],
        on="match_id",
        how="left",
    )
    veto = pd.DataFrame(
        [
            {"match_id": 1, "action": "left_over", "map_name": "ancient"},
            {"match_id": 2, "action": "picked", "map_name": "mirage"},
            {"match_id": 2, "action": "picked", "map_name": "nuke"},
            {"match_id": 2, "action": "left_over", "map_name": "ancient"},
        ]
    )

    features = build_map_features(matches, maps, veto)
    snapshot = features[(features["match_id"].eq(2)) & (features["team_id"].eq(1))].iloc[0]

    assert snapshot["avg_map_count_before"] == pytest.approx(1 / 3)
    assert snapshot["series_maps_known"] == pytest.approx(1.0)


def test_explicit_inference_map_pool_override_does_not_require_fake_results() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-01-01",
                    "match_datetime_utc": "2025-01-01T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                },
                {
                    "match_id": 2,
                    "match_date": "2025-01-02",
                    "match_datetime_utc": "2025-01-02T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 0,
                },
            ]
        )
    )
    maps = pd.DataFrame(
        [
            {
                "match_id": 1,
                "map_no": 1,
                "map_name": "mirage",
                "team1_map_score": 13,
                "team2_map_score": 8,
                "team1_ct_rounds": 7,
                "team1_t_rounds": 6,
                "team2_ct_rounds": 4,
                "team2_t_rounds": 4,
            }
        ]
    ).merge(
        matches[["match_id", "match_datetime_utc", "team1_id", "team2_id"]],
        on="match_id",
        how="left",
    )

    features = build_map_features(
        matches,
        maps,
        empty_veto(),
        map_pool_overrides={2: ["mirage"]},
    )
    snapshot = features[(features["match_id"].eq(2)) & (features["team_id"].eq(1))].iloc[0]
    assert snapshot["avg_map_count_before"] == pytest.approx(1.0)
    assert snapshot["series_maps_known"] == pytest.approx(1.0)


def test_decider_history_is_attributed_to_both_teams() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": match_id,
                    "match_date": f"2025-01-0{match_id}",
                    "match_datetime_utc": f"2025-01-0{match_id}T12:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                }
                for match_id in [1, 2, 3]
            ]
        )
    )
    veto = pd.DataFrame(
        [
            {"match_id": 1, "team_id": 1, "action": "picked", "map_name": "mirage"},
            {"match_id": 2, "team_id": pd.NA, "action": "left_over", "map_name": "mirage"},
            {"match_id": 3, "team_id": pd.NA, "action": "left_over", "map_name": "mirage"},
        ]
    )
    features = build_veto_features(matches, veto)
    current = features[features["match_id"].eq(3)].set_index("team_id")
    assert current.loc[1, "veto_leftover_rate_before"] == pytest.approx(0.5)
    assert current.loc[2, "veto_leftover_rate_before"] == pytest.approx(1.0)


def test_large_date_disagreement_is_repaired() -> None:
    matches = normalize_matches(
        match_frame(
            [
                {
                    "match_id": 1,
                    "match_date": "2025-02-10",
                    "match_datetime_utc": "2025-01-01T09:00:00Z",
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_win": 1,
                }
            ]
        )
    )
    assert matches.loc[0, "datetime_repaired"] == 1
    assert matches.loc[0, "match_datetime_utc"] == pd.Timestamp("2025-02-10T12:00:00Z")
