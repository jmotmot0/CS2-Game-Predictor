import numpy as np
import pandas as pd
import pytest

from src.feature_selection_experiment import (
    feature_bundles, split_masks, elo_features, point_losses,
    monthly_permutation, select_features, paired_weekly_interval, summarize, rank_features,
)
from src.feature_engineering import compute_team_history_features
from src.modeling import MODEL_FEATURES


def test_bundles_partition_feature_allowlist():
    flat = sum(feature_bundles(), [])
    assert len(flat) == len(set(flat)) == 40
    assert set(flat) == set(MODEL_FEATURES)


def test_four_temporal_windows_are_disjoint_and_ordered():
    frame = pd.DataFrame({"match_datetime_utc": pd.date_range("2024-01-01", "2026-01-01", tz="UTC")})
    masks = split_masks(frame, "2025-04-01")
    assert np.max(sum(masks.values())) == 1
    for left, right in zip(list(masks.values())[:-1], list(masks.values())[1:]):
        assert frame.loc[left].iloc[-1, 0] < frame.loc[right].iloc[0, 0]
    with pytest.raises(ValueError):
        split_masks(frame, "2024-01-01")


def test_elo_features_match_original_and_ignore_current_batch_labels():
    frame = pd.DataFrame({"match_id": range(1, 7), "team1_id": [1, 2, 1, 3, 1, 2],
                          "team2_id": [2, 3, 3, 2, 2, 3], "team1_win": [1, 0, 1, 0, 1, 0],
                          "match_datetime_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True)})
    for k in [8, 48, 64]:
        expected = compute_team_history_features(frame, elo_k=k).set_index(["match_id", "team_id"])
        actual = elo_features(frame, k)
        for i, row in frame.iterrows():
            left = expected.loc[(row.match_id, row.team1_id)]
            right = expected.loc[(row.match_id, row.team2_id)]
            np.testing.assert_allclose(actual.iloc[i].to_numpy(), [left.elo_pre-right.elo_pre, left.avg_opp_elo_last_10-right.avg_opp_elo_last_10], equal_nan=True)
        changed = frame.copy()
        changed.loc[2:, "team1_win"] = 1 - changed.loc[2:, "team1_win"]
        pd.testing.assert_frame_equal(actual.iloc[:4], elo_features(changed, k).iloc[:4])


def test_permutations_stay_in_month_and_preserve_atomic_encodings():
    times = pd.Series(pd.date_range("2025-01-20", periods=70, tz="UTC"))
    permutation = monthly_permutation(times, np.random.default_rng(42))
    assert sorted(permutation) == list(range(len(times)))
    assert (times.dt.month.to_numpy() == times.iloc[permutation].dt.month.to_numpy()).all()


@pytest.mark.parametrize("budget", [10, 20, 30, 40])
def test_selection_exact_budget_and_no_split_encoding(budget):
    ranking = [{"method": "test", "features": group, "bundle": "|".join(group), "score_per_feature": 1.0} for group in feature_bundles()]
    selected = select_features(ranking, "test", budget)
    assert len(selected) == budget
    assert selected == [f for f in MODEL_FEATURES if f in selected]
    for group in feature_bundles():
        assert not set(selected).intersection(group) or set(group).issubset(selected)


def test_losses_and_paired_interval():
    np.testing.assert_allclose(point_losses([0, 1], [.25, .75]), -np.log(.75))
    assert np.isfinite(point_losses([0, 1], [1, 0])).all()
    times = pd.Series(pd.date_range("2025-01-01", periods=100, tz="UTC"))
    np.testing.assert_allclose(paired_weekly_interval(np.full(100, .03), times), [.03, .03])


def test_summary_averages_seed_losses_not_probabilities():
    rows = []
    for seed, loss in [(42, .2), (43, .4)]:
        for method, budget, offset in [("PVC", 40, 0), ("PFI_LogLoss", 10, .1)]:
            for match in range(10):
                rows.append({"seed": seed, "method": method, "budget": budget, "match_id": match,
                             "match_datetime_utc": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=match),
                             "loss": loss+offset, "correct": 1})
    result = summarize(pd.DataFrame(rows))
    assert result[0]["n_matches"] == 10
    assert result[0]["log_loss"] == pytest.approx(.3)
    assert result[1]["delta_vs_40"] == pytest.approx(.1)


def test_loss_importance_detects_signal_and_does_not_modify_source():
    class SignalModel:
        def predict_proba(self, x):
            p = 1 / (1 + np.exp(-x.diff_elo_pre.to_numpy()))
            return np.column_stack([1-p, p])

        def get_feature_importance(self):
            return np.array([100 if name == "diff_elo_pre" else 0 for name in MODEL_FEATURES])

    x = pd.DataFrame(0.0, index=range(60), columns=MODEL_FEATURES)
    x["diff_elo_pre"] = np.tile([-2, 2], 30)
    original = x.copy(deep=True)
    y = (x.diff_elo_pre > 0).to_numpy().astype(int)
    times = pd.Series(pd.date_range("2025-01-01", periods=60, tz="UTC"))
    ranking = rank_features(SignalModel(), x, y, times, 42, 4)
    signal = next(row for row in ranking if row["method"] == "PFI_LogLoss" and row["bundle"] == "diff_elo_pre")
    assert signal["score"] > .1
    for row in ranking:
        if row["bundle"] != "diff_elo_pre":
            assert row["score"] == pytest.approx(0)
    pd.testing.assert_frame_equal(x, original)
