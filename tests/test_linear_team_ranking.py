import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.linear_team_ranking import (
    LinearTeamRanker, binary_target, evaluate, pair_log_loss, select_regularization,
)
from src.team_ranking import swap_team_columns, team_rows


def matches():
    return pd.DataFrame({
        "team1_id": [10, 30, 10, 30, 10, 30],
        "team2_id": [20, 20, 20, 20, 20, 20],
        "team1_elo_pre": [1700., 1400., 1700., 1400., 1700., 1400.],
        "team2_elo_pre": [1500., 1700., 1500., 1700., 1500., 1700.],
        "team1_lineup_player_rating_mean": [1.2, .9, np.nan, .9, 1.2, .9],
        "team2_lineup_player_rating_mean": [1., 1.2, 1., np.nan, 1., 1.2],
        "bo3": [1., 1., 0., 0., 1., 0.],
        "team1_win": [1, 0, 1, 0, 1, 0],
    })


FEATURES = ["elo_pre", "lineup_player_rating_mean", "bo3"]


def trained():
    frame = matches()
    return LinearTeamRanker(FEATURES).fit(frame, frame.team1_win), frame


def test_linear_model_has_one_shared_team_score_and_pairwise_margin():
    model, frame = trained()
    z = model.transform_team_rows(team_rows(frame, FEATURES))
    expected = z @ model.estimator_.coef_[0]
    np.testing.assert_allclose(model.predict(team_rows(frame, FEATURES)), expected, atol=1e-15)
    scores = model.scores(frame)
    np.testing.assert_allclose(scores.ravel(), expected, atol=1e-15)
    np.testing.assert_allclose(scores[:, 0] - scores[:, 1], model.pair_differences(frame) @ model.estimator_.coef_[0], atol=1e-15)
    assert not model.estimator_.fit_intercept
    assert model.n_iter_ < model.max_iter and model.convergence_warnings_ == []


def test_swapping_teams_exchanges_scores_and_complements_probability():
    model, frame = trained()
    reverse = swap_team_columns(frame)
    np.testing.assert_allclose(model.scores(reverse), model.scores(frame)[:, ::-1], atol=1e-15)
    np.testing.assert_allclose(model.probability(frame) + model.probability(reverse), 1., atol=1e-15)
    np.testing.assert_equal(model.decision(frame)[0] + model.decision(reverse)[0], 1)


def test_common_linear_context_cancels_and_is_not_an_interaction():
    model, frame = trained()
    assert model.estimator_.coef_[0, 2] == 0
    assert "bo3" in model.zero_pair_variance_features_
    np.testing.assert_equal(model.pair_differences(frame)[:, 2], 0)
    changed = frame.copy()
    changed["bo3"] = [10, 20, 30, 40, 50, 60]
    np.testing.assert_allclose(model.probability(changed), model.probability(frame), atol=1e-15)


def test_imputation_and_scaling_use_only_training_team_objects():
    model, frame = trained()
    own = team_rows(frame, FEATURES).to_numpy()
    np.testing.assert_allclose(model.imputer_.statistics_, np.nanmedian(own, axis=0))
    medians = model.imputer_.statistics_.copy()
    means = model.scaler_.mean_.copy()
    scales = model.scaler_.scale_.copy()
    future = frame.copy()
    future["team1_elo_pre"] = 999999
    future["team2_lineup_player_rating_mean"] = -999999
    model.probability(future)
    np.testing.assert_equal(model.imputer_.statistics_, medians)
    np.testing.assert_equal(model.scaler_.mean_, means)
    np.testing.assert_equal(model.scaler_.scale_, scales)
    assert model.training_match_count_ == 6 and model.training_team_row_count_ == 12


def test_empty_training_column_kept_and_missing_source_column_rejected():
    frame = matches()
    frame["team1_lineup_player_rating_mean"] = np.nan
    frame["team2_lineup_player_rating_mean"] = np.nan
    model = LinearTeamRanker(FEATURES).fit(frame, frame.team1_win)
    assert model.estimator_.coef_.shape == (1, 3)
    assert model.imputer_.statistics_[1] == 0
    assert model.all_missing_training_features_ == ["lineup_player_rating_mean"]
    assert np.isfinite(model.probability(frame)).all()
    with pytest.raises(ValueError, match="missing columns"):
        model.probability(frame.drop(columns=["team2_elo_pre"]))


def test_raw_score_tie_uses_lower_id_independent_of_column_order():
    model, frame = trained()
    tie = frame.iloc[[0]].copy()
    tie["team1_id"], tie["team2_id"] = 30, 20
    for name in ["elo_pre", "lineup_player_rating_mean"]:
        tie[f"team2_{name}"] = tie[f"team1_{name}"].to_numpy()
    decision, exact_tie = model.decision(tie)
    np.testing.assert_equal(decision, [0])
    np.testing.assert_equal(exact_tie, [True])
    np.testing.assert_equal(model.probability(tie), [.5])
    np.testing.assert_equal(model.decision(swap_team_columns(tie))[0], [1])
    *_, measured = evaluate(model, pd.concat([tie, swap_team_columns(tie)], ignore_index=True), [0, 1])
    assert measured["accuracy"] == 1 and measured["exact_score_ties"] == 2


def test_pair_logit_matches_binary_cross_entropy_and_is_stable():
    y = np.array([1, 0, 1, 0])
    margin = np.array([2., 2., -2., -2.])
    p = expit(margin)
    bce = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    np.testing.assert_allclose(pair_log_loss(y, margin), bce, atol=1e-15)
    np.testing.assert_allclose(pair_log_loss(y, margin), pair_log_loss(1 - y, -margin), atol=1e-15)
    np.testing.assert_allclose(pair_log_loss([1, 0], [-1000., 1000.]), [1000., 1000.])


def test_bad_inputs_and_unfitted_prediction_are_not_silent():
    for bad in [[], [2, 0], [[1], [0]], [1]]:
        with pytest.raises(ValueError, match="binary outcome"):
            binary_target(bad, 2)
    with pytest.raises(ValueError, match="both classes"):
        LinearTeamRanker(FEATURES).fit(matches(), [1] * 6)
    with pytest.raises(ValueError, match="positive"):
        LinearTeamRanker(C=0)
    with pytest.raises(ValueError, match="Fit"):
        LinearTeamRanker(FEATURES).probability(matches())


def test_regularization_selection_does_not_use_test_outcomes():
    frame = pd.concat([matches(), matches(), matches()], ignore_index=True)
    masks = {name: np.arange(18) // 6 == index for index, name in enumerate(["train", "validation", "test"])}
    original_y = frame.team1_win.to_numpy()
    model, records, C = select_regularization(frame, original_y, masks, feature_names=FEATURES, grid=[.01, 1.])
    changed_y = original_y.copy()
    changed_y[masks["test"]] = 1 - changed_y[masks["test"]]
    other, changed_records, other_C = select_regularization(frame, changed_y, masks, feature_names=FEATURES, grid=[.01, 1.])
    assert C == other_C and records == changed_records
    np.testing.assert_equal(model.estimator_.coef_, other.estimator_.coef_)
    assert model.training_team_row_count_ == 12


def test_nonconverged_fit_is_not_silently_published():
    frame = matches()
    with pytest.raises(RuntimeError, match="did not converge"):
        LinearTeamRanker(FEATURES, max_iter=1).fit(frame, frame.team1_win)


def test_saved_model_keeps_the_exact_common_scoring_function(tmp_path):
    import joblib

    model, frame = trained()
    path = tmp_path / "linear.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_equal(restored.scores(frame), model.scores(frame))
    np.testing.assert_equal(restored.probability(frame), model.probability(frame))


def test_module_cli_saves_a_model_importable_from_another_process(tmp_path):
    import json
    import subprocess
    import sys
    import joblib
    from src.modeling import sha256_file
    from src.team_ranking import EXTENDED_TEAM_FEATURES, SHARED_FEATURES

    frozen = tmp_path / "fixture_frozen"
    output = tmp_path / "new_results"
    frozen.mkdir()
    frame = pd.concat([matches(), matches(), matches()], ignore_index=True)
    frame["match_id"] = np.arange(18) + 1
    frame["match_datetime_utc"] = np.repeat(["2024-01-01", "2025-07-01", "2026-01-01"], 6)
    for name in EXTENDED_TEAM_FEATURES:
        if name in SHARED_FEATURES:
            if name not in frame:
                frame[name] = 0.
        else:
            suffix = "team_rank" if name == "rank_score" else name
            for side in ["team1", "team2"]:
                if f"{side}_{suffix}" not in frame:
                    frame[f"{side}_{suffix}"] = 1.
    frame.to_csv(frozen / "extended_features.csv", index=False)
    protocol = {
        "training_end_exclusive": "2025-07-01", "validation_end_exclusive": "2026-01-01",
        "split_counts": {"train": 6, "validation": 6, "test": 6},
        "test_status": "Self-contained CLI fixture", "availability": "Fixture only",
    }
    (frozen / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for split, indices in [("validation", slice(6, 12)), ("test", slice(12, 18))]:
        frame.iloc[indices][["match_id", "match_datetime_utc", "team1_id", "team2_id", "team1_win"]].to_csv(
            frozen / f"{split}_predictions.csv", index=False,
        )
    pd.DataFrame({"split": ["validation", "test"], "model": ["elo", "elo"], "accuracy": [.5, .5]}).to_csv(
        frozen / "comparison_metrics.csv", index=False,
    )
    manifest = {"files": {path.name: sha256_file(path) for path in frozen.iterdir()}}
    (frozen / "results_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.linear_team_ranking", "--frozen-dir", str(frozen),
         "--output", str(output), "--c-grid", ".1"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    restored = joblib.load(output / "linear_ranker.joblib")
    assert type(restored) is LinearTeamRanker
    assert np.isfinite(restored.probability(frame)).all()
