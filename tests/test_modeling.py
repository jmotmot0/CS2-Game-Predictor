from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_difference_features
from src.modeling import (
    ABLATION_GROUPS,
    MODEL_FEATURES,
    augment_team_swap,
    chronological_masks,
    elo_probabilities,
    expected_calibration_error,
    metric_row,
    swap_team_perspective,
    symmetrized_probability,
)
from src.backtest_models import fold_masks
from src.monitor_drift import population_stability_index


def test_model_allow_list_is_unique_and_has_no_target_columns() -> None:
    assert len(MODEL_FEATURES) == 40
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
    forbidden = {"team1_win", "winner", "team1_score", "team2_score", "match_id"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_ablation_groups_cover_every_model_feature_once() -> None:
    ablation_features = [
        feature
        for group_features in ABLATION_GROUPS.values()
        for feature in group_features
    ]
    assert len(ablation_features) == len(set(ablation_features))
    assert set(ablation_features) == set(MODEL_FEATURES)


def test_chronological_split_has_expected_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "match_datetime_utc": pd.to_datetime(
                ["2025-06-30", "2025-07-01", "2025-12-31", "2026-01-01"],
                utc=True,
            )
        }
    )
    train, validation, test = chronological_masks(frame)
    assert train.tolist() == [True, False, False, False]
    assert validation.tolist() == [False, True, True, False]
    assert test.tolist() == [False, False, False, True]


def test_walk_forward_fold_uses_disjoint_time_windows() -> None:
    frame = pd.DataFrame(
        {
            "match_datetime_utc": pd.to_datetime(
                ["2025-01-01", "2025-04-01", "2025-07-01", "2025-09-30"],
                utc=True,
            )
        }
    )
    train, validation, test = fold_masks(
        frame,
        test_start="2025-07-01",
        test_end="2025-10-01",
        validation_days=91,
    )
    assert train.tolist() == [True, False, False, False]
    assert validation.tolist() == [False, True, False, False]
    assert test.tolist() == [False, False, True, True]
    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)


def test_population_stability_index_detects_distribution_shift() -> None:
    reference = pd.Series(np.linspace(0.0, 1.0, 1000))
    unchanged = pd.Series(np.linspace(0.0, 1.0, 1000))
    shifted = pd.Series(np.linspace(1.0, 2.0, 1000))
    assert population_stability_index(reference, unchanged) == pytest.approx(0.0)
    assert population_stability_index(reference, shifted) > 0.25


def test_elo_updates_are_simultaneous_for_equal_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "match_id": [1, 2],
            "match_datetime_utc": pd.to_datetime(
                ["2025-01-01T12:00:00Z", "2025-01-01T12:00:00Z"], utc=True
            ),
            "team1_id": [1, 1],
            "team2_id": [2, 3],
            "team1_win": [1, 0],
        }
    )
    probability, team1_pre, _ = elo_probabilities(frame, k=48)
    assert probability.tolist() == pytest.approx([0.5, 0.5])
    assert team1_pre.tolist() == pytest.approx([1500.0, 1500.0])


def test_metric_bundle_matches_simple_example() -> None:
    target = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.4, 0.6, 0.9])
    metrics = metric_row(target, probability)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.085)
    assert expected_calibration_error(target, probability) >= 0.0


def test_team_swap_negates_difference_features() -> None:
    original = pd.DataFrame(
        {
            "team1_elo_pre": [1600.0],
            "team2_elo_pre": [1500.0],
            "team1_team_rank": [4.0],
            "team2_team_rank": [10.0],
        }
    )
    swapped = original.rename(
        columns={
            "team1_elo_pre": "team2_elo_pre",
            "team2_elo_pre": "team1_elo_pre",
            "team1_team_rank": "team2_team_rank",
            "team2_team_rank": "team1_team_rank",
        }
    )
    first = add_difference_features(original)
    second = add_difference_features(swapped)
    assert first.loc[0, "diff_elo_pre"] == -second.loc[0, "diff_elo_pre"]
    assert first.loc[0, "diff_rank"] == -second.loc[0, "diff_rank"]


def test_team_swap_augmentation_balances_and_mirrors_training_rows() -> None:
    frame = pd.DataFrame([{feature: 0.0 for feature in MODEL_FEATURES}])
    frame.loc[0, "diff_elo_pre"] = 120.0
    augmented, target = augment_team_swap(frame, np.asarray([1]))
    assert len(augmented) == 2
    assert target.tolist() == [1, 0]
    assert augmented.loc[1, "diff_elo_pre"] == pytest.approx(-120.0)
    assert augmented.loc[1, "bo3"] == augmented.loc[0, "bo3"]


def test_symmetrized_prediction_is_exactly_complementary() -> None:
    class BiasedModel:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            score = frame["diff_elo_pre"].to_numpy() / 200.0 + 0.3
            probability = 1.0 / (1.0 + np.exp(-score))
            return np.column_stack([1 - probability, probability])

    frame = pd.DataFrame([{feature: 0.0 for feature in MODEL_FEATURES}])
    frame.loc[0, "diff_elo_pre"] = 100.0
    swapped = swap_team_perspective(frame)
    forward = symmetrized_probability(BiasedModel(), frame)[0]
    reverse = symmetrized_probability(BiasedModel(), swapped)[0]
    assert forward + reverse == pytest.approx(1.0)
