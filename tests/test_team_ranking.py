import numpy as np
import pandas as pd
import pytest

from src.team_ranking import (
    LEGACY_TO_TEAM_FEATURE, SHARED_FEATURES, TEAM_FEATURE_GROUPS,
    TEAM_MODEL_FEATURES, EXTENDED_TEAM_FEATURES, EXTENDED_TEAM_FEATURE_GROUPS,
    swap_team_columns, team_columns_for,
    team_ranking_decision, team_ranking_pool, team_ranking_probability,
    team_ranking_scores, team_rows,
)


def example_frame():
    return pd.DataFrame({
        "team1_id": [10, 30, 10], "team2_id": [20, 20, 20],
        "team1_elo_pre": [1700., 1600., 1500.],
        "team2_elo_pre": [1500., 1800., 1500.],
        "team1_team_rank": [4., 8., np.nan], "team2_team_rank": [9., 2., 5.],
        "bo3": [1., 1., 0.], "team1_win": [1, 0, 1],
        # Deliberately inconsistent legacy values prove these are never consumed.
        "diff_elo_pre": [99999., 99999., 99999.],
    })


def test_own_team_groups_are_a_disjoint_40_feature_partition():
    assert len(TEAM_MODEL_FEATURES) == len(set(TEAM_MODEL_FEATURES)) == 40
    assert len(SHARED_FEATURES) == 6
    assert {group: len(value["features"]) for group, value in TEAM_FEATURE_GROUPS.items()} == {
        "individual": 4, "team": 21, "cohesion": 2, "controls": 13,
    }
    assert team_columns_for(TEAM_FEATURE_GROUPS) == TEAM_MODEL_FEATURES
    assert LEGACY_TO_TEAM_FEATURE["diff_rank"] == "rank_score"
    assert not any(name.startswith("diff_") for name in TEAM_MODEL_FEATURES)


def test_extended_research_schema_keeps_base_and_expands_only_the_planned_groups():
    assert EXTENDED_TEAM_FEATURES[:40] == TEAM_MODEL_FEATURES
    assert len(EXTENDED_TEAM_FEATURES) == len(set(EXTENDED_TEAM_FEATURES)) == 46
    assert {group: len(value["features"]) for group, value in EXTENDED_TEAM_FEATURE_GROUPS.items()} == {
        "individual": 7, "team": 21, "cohesion": 5, "controls": 13,
    }
    assert team_columns_for(EXTENDED_TEAM_FEATURE_GROUPS, extended=True) == EXTENDED_TEAM_FEATURES
    frame = pd.DataFrame({
        "team1_lineup_player_rating_max": [1.2], "team2_lineup_player_rating_max": [1.1],
        "team1_roster_pair_experience_90d": [5.], "team2_roster_pair_experience_90d": [10.],
    })
    rows = team_rows(frame, ["lineup_player_rating_max", "roster_pair_experience_90d"])
    np.testing.assert_allclose(rows.lineup_player_rating_max, [1.2, 1.1])
    np.testing.assert_allclose(rows.roster_pair_experience_90d, [5., 10.])


def test_rows_use_own_absolute_values_once_and_common_context_twice():
    frame = example_frame()
    rows = team_rows(frame, ["elo_pre", "rank_score", "bo3"])
    np.testing.assert_allclose(rows.elo_pre, [1700, 1500, 1600, 1800, 1500, 1500])
    np.testing.assert_allclose(rows.rank_score, [-4, -9, -8, -2, np.nan, -5], equal_nan=True)
    np.testing.assert_equal(rows.bo3.to_numpy(), [1, 1, 1, 1, 0, 0])
    pd.testing.assert_frame_equal(frame, example_frame())
    with pytest.raises(ValueError, match="missing columns"):
        team_rows(frame, ["lineup_player_adr_mean"])
    with pytest.raises(ValueError, match="Unknown"):
        team_rows(frame, ["diff_elo_pre"])
    with pytest.raises(ValueError, match="unique"):
        team_rows(frame, ["elo_pre", "elo_pre"])


def test_full_schema_ignores_current_scores_outcome_and_signed_differences():
    values = {name: [1.] for name in SHARED_FEATURES}
    for name in TEAM_MODEL_FEATURES:
        if name in SHARED_FEATURES:
            continue
        suffix = "team_rank" if name == "rank_score" else name
        values[f"team1_{suffix}"] = [10.]
        values[f"team2_{suffix}"] = [20.]
    frame = pd.DataFrame(values)
    before = team_rows(frame)
    assert before.shape == (2, 40)
    frame["team1_score"] = 2
    frame["team2_score"] = 0
    frame["team1_win"] = 1
    for name in LEGACY_TO_TEAM_FEATURE:
        if name.startswith("diff_"):
            frame[name] = -999999.
    pd.testing.assert_frame_equal(before, team_rows(frame))


def test_nullable_missing_and_infinite_values_become_nan_not_fake_zeroes():
    frame = pd.DataFrame({
        "team1_elo_pre": pd.Series([pd.NA, np.inf], dtype="Float64"),
        "team2_elo_pre": [1500., -np.inf],
    })
    rows = team_rows(frame, ["elo_pre"])
    np.testing.assert_allclose(rows.elo_pre, [np.nan, 1500., np.nan, np.nan], equal_nan=True)


def test_pool_has_one_query_and_one_outcome_pair_per_series():
    frame = example_frame()
    pool = team_ranking_pool(frame, frame.team1_win.to_numpy(), ["elo_pre", "bo3"])
    assert pool.num_row() == 6 and pool.num_pairs() == 3
    np.testing.assert_equal(pool.get_label(), [1, 0, 0, 1, 1, 0])
    groups = pool.get_group_id_hash()
    assert np.unique(groups).size == 3
    np.testing.assert_equal(groups[0::2], groups[1::2])
    for invalid in ([1, 2, 0], [1, 0], [[1], [0], [1]]):
        with pytest.raises(ValueError, match="binary outcome"):
            team_ranking_pool(frame, invalid, ["elo_pre"])
    with pytest.raises(ValueError, match="nonempty"):
        team_ranking_pool(frame.iloc[0:0], [], ["elo_pre"])


def test_exchange_only_permutes_objects_and_complements_probability_and_choice():
    class Scores:
        def predict(self, rows):
            return rows.elo_pre.to_numpy() / 100 + rows.bo3.to_numpy() * 0.2

    frame = example_frame()
    columns = ["elo_pre", "bo3"]
    reversed_frame = swap_team_columns(frame)
    pd.testing.assert_frame_equal(swap_team_columns(reversed_frame), frame)
    scores = team_ranking_scores(Scores(), frame, columns)
    reversed_scores = team_ranking_scores(Scores(), reversed_frame, columns)
    np.testing.assert_equal(reversed_scores, scores[:, ::-1])
    p = team_ranking_probability(Scores(), frame, columns)
    q = team_ranking_probability(Scores(), reversed_frame, columns)
    np.testing.assert_allclose(p + q, 1., atol=1e-15)
    winners, ties = team_ranking_decision(Scores(), frame, columns)
    reversed_winners, reversed_ties = team_ranking_decision(Scores(), reversed_frame, columns)
    np.testing.assert_equal(winners + reversed_winners, 1)
    np.testing.assert_equal(ties, [False, False, True])
    np.testing.assert_equal(ties, reversed_ties)
    assert p[-1] == 0.5
    with pytest.raises(ValueError, match="Stable team IDs"):
        team_ranking_decision(Scores(), frame.drop(columns=["team1_id"]), columns)


def test_pool_pair_orientation_controls_learning_for_both_team_orders():
    from catboost import CatBoostRanker

    frame = pd.DataFrame({
        "team1_elo_pre": np.tile([1300., 1500., 1700., 1900.], 15),
        "team2_elo_pre": np.tile([1900., 1700., 1500., 1300.], 15),
    })
    y = (frame.team1_elo_pre > frame.team2_elo_pre).astype(int).to_numpy()
    model = CatBoostRanker(
        loss_function="PairLogit", iterations=20, depth=2, verbose=False,
        allow_writing_files=False, thread_count=2, random_seed=42,
    )
    model.fit(team_ranking_pool(frame, y, ["elo_pre"]))
    p = team_ranking_probability(model, frame, ["elo_pre"])
    assert np.mean((p > 0.5) == y) == 1.
    np.testing.assert_allclose(
        p + team_ranking_probability(model, swap_team_columns(frame), ["elo_pre"]),
        1., atol=1e-15,
    )
