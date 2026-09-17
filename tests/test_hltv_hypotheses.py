import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.hltv_hypotheses import (
    HYPOTHESES, LAMBDA, QUALITY, eligible, elo_history, fit_logistic, losses,
    objective_gradient_hessian, time_masks, weekly_interval,
)
from src.feature_engineering import compute_team_history_features


def sample():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(600, 4)) * np.array([1, 3, .1, 50])
    y = rng.binomial(1, expit(x @ np.array([.5, -.2, 2., .01])))
    return x, y


def test_derivatives_match_finite_differences():
    rng = np.random.default_rng(51)
    x, y, beta = rng.normal(size=(100, 4)), rng.binomial(1, .5, 100), rng.normal(size=4)
    _, gradient, hessian = objective_gradient_hessian(beta, x, y)
    epsilon = 1e-5
    for j in range(4):
        step = np.eye(4)[j]*epsilon
        f_plus, g_plus, _ = objective_gradient_hessian(beta+step, x, y)
        f_minus, g_minus, _ = objective_gradient_hessian(beta-step, x, y)
        assert gradient[j] == pytest.approx((f_plus-f_minus)/(2*epsilon), abs=1e-8)
        np.testing.assert_allclose(hessian[:, j], (g_plus-g_minus)/(2*epsilon), atol=1e-8)
    assert np.linalg.eigvalsh(hessian).min() >= LAMBDA-1e-10


def test_custom_newton_matches_independent_sklearn_fit():
    x, y = sample()
    fit = fit_logistic(x, y)
    reference = LogisticRegression(C=1/(len(y)*LAMBDA), fit_intercept=False,
                                   solver="lbfgs", max_iter=2000, tol=1e-11)
    reference.fit(x/fit.scales, y)
    np.testing.assert_allclose(fit.predict(x), reference.predict_proba(x/fit.scales)[:, 1], atol=1e-6)
    assert fit.gradient_max < 1e-8


def test_probability_is_exactly_complementary_when_teams_swap():
    x, y = sample()
    fit = fit_logistic(x, y)
    np.testing.assert_allclose(fit.predict(x)+fit.predict(-x), 1., atol=1e-15)
    swapped = fit_logistic(-x, 1-y)
    np.testing.assert_allclose(fit.beta, swapped.beta, atol=1e-10)


def test_zero_feature_is_well_defined_and_no_intercept_bias():
    x = np.zeros((100, 2))
    y = np.r_[np.ones(80), np.zeros(20)]
    fit = fit_logistic(x, y)
    np.testing.assert_allclose(fit.predict(x), .5)
    assert np.isfinite(fit.beta).all()


@pytest.mark.parametrize("invalid", ["nan", "target", "empty", "ridge"])
def test_invalid_training_data_rejected(invalid):
    x, y = sample()
    ridge = LAMBDA
    if invalid == "nan":
        x[0, 0] = np.nan
    elif invalid == "target":
        y[0] = 2
    elif invalid == "empty":
        x, y = x[:0], y[:0]
    else:
        ridge = 0
    with pytest.raises(ValueError):
        fit_logistic(x, y, ridge)


def test_quarters_do_not_overlap_and_future_never_trains_current():
    times = pd.Series(pd.date_range("2024-01-01", "2026-04-05", tz="UTC"))
    evaluations = []
    for start in ["2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01"]:
        train, test = time_masks(times, start)
        train &= np.ones(len(times), dtype=bool)
        test &= np.ones(len(times), dtype=bool)
        assert not (train & test).any()
        assert times[train].max() < times[test].min()
        evaluations.append(test)
    assert np.max(sum(evaluations)) == 1


def test_fixed_elo_matches_reference_and_current_result_cannot_leak():
    frame = pd.DataFrame({"match_id": range(1, 7), "team1_id": [1, 2, 1, 3, 1, 2],
                          "team2_id": [2, 3, 3, 2, 2, 3], "team1_win": [1, 0, 1, 0, 1, 0],
                          "match_datetime_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True)})
    delta, opponent = elo_history(frame)
    expected = compute_team_history_features(frame, elo_k=32).set_index(["match_id", "team_id"])
    for i, row in frame.iterrows():
        first, second = expected.loc[(row.match_id, row.team1_id)], expected.loc[(row.match_id, row.team2_id)]
        np.testing.assert_allclose([delta[i], opponent[i]],
                                   [first.elo_pre-second.elo_pre, first.avg_opp_elo_last_10-second.avg_opp_elo_last_10], equal_nan=True)
    frame.loc[2:, "team1_win"] = 1-frame.loc[2:, "team1_win"]
    later_delta, later_opponent = elo_history(frame)
    np.testing.assert_allclose(delta[:4], later_delta[:4])
    np.testing.assert_allclose(opponent[:4], later_opponent[:4], equal_nan=True)


def test_loss_and_block_interval_keep_sign_and_scale():
    np.testing.assert_allclose(losses([1, 0], [.8, .2]), -np.log(.8))
    times = pd.Series(pd.date_range("2025-04-01", periods=120, tz="UTC"))
    np.testing.assert_allclose(weekly_interval(np.full(120, .03), times, draws=300), [.03, .03])


def test_hypotheses_are_unique_and_have_explicit_units():
    assert len(HYPOTHESES) == len({h.key for h in HYPOTHESES}) == 12
    assert all(h.scale > 0 and h.unit and h.question for h in HYPOTHESES)


def test_missing_roster_coverage_excluded_not_treated_as_zero():
    frame = pd.DataFrame({"team1_matches_before": [20, 20], "team2_matches_before": [20, 20]})
    design = pd.DataFrame(0., index=range(2), columns=QUALITY+[HYPOTHESES[0].feature])
    design.loc[0, "player_coverage"] = np.nan
    np.testing.assert_array_equal(eligible(frame, design, HYPOTHESES[0]), [False, True])
