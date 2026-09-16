from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import MAP_COLUMNS, PLAYER_COLUMNS, ROSTER_COLUMNS
from src.inference_state import INFERENCE_STATE_SCHEMA_VERSION, prediction_features_from_state
from src.modeling import MODEL_FEATURES, load_schema
from src.predict_match import generate_map_scenarios, parse_int_list, parse_map_list


def minimal_state() -> dict[str, object]:
    return {
        "schema_version": INFERENCE_STATE_SCHEMA_VERSION,
        "state_as_of_utc": "2026-04-10T12:00:00+00:00",
        "elo_k": 48.0,
        "teams": {
            "1": {
                "elo": 1600.0,
                "matches": 10,
                "wins": 7,
                "recent_dates": ["2026-04-05T12:00:00+00:00"],
                "last_match_date": "2026-04-05T12:00:00+00:00",
                "recent_results": [1, 1, 0],
                "streak": -1,
                "opponent_elo": [1500.0],
                "rank": 5.0,
                "roster": [11, 12, 13, 14, 15],
            },
            "2": {
                "elo": 1500.0,
                "matches": 8,
                "wins": 3,
                "recent_dates": ["2026-04-07T12:00:00+00:00"],
                "last_match_date": "2026-04-07T12:00:00+00:00",
                "recent_results": [0, 1, 0],
                "streak": -1,
                "opponent_elo": [1550.0],
                "rank": 10.0,
                "roster": [21, 22, 23, 24, 25],
            },
        },
        "h2h": {"1:2": {"wins": {"1": 2, "2": 1}, "recent": [1, 2, 1]}},
        "players": {},
        "maps": {},
        "veto": {},
    }


def test_fast_state_preserves_real_day_intervals() -> None:
    row = prediction_features_from_state(
        minimal_state(),
        team1_id=1,
        team2_id=2,
        match_time=pd.Timestamp("2026-04-12T12:00:00Z"),
        bo=3,
        location="lan",
        team1_rank=None,
        team2_rank=None,
        team1_players=[],
        team2_players=[],
        maps=[],
    )
    # Seven days since team 1's match minus five days since team 2's match.
    assert row["diff_days_since_last_match"] == pytest.approx(2.0)
    assert row["diff_activity_7d"] == pytest.approx(0.0)
    assert row["diff_elo_pre"] == pytest.approx(100.0)
    assert row["diff_overall_winrate"] == pytest.approx(0.7 - 0.375)
    assert row["diff_winrate_last_20"] == pytest.approx(1.0 / 3.0)
    for name in ROSTER_COLUMNS + PLAYER_COLUMNS + MAP_COLUMNS:
        assert pd.isna(row[f"diff_{name}"])


def test_fast_state_counts_only_maps_with_prior_history_as_known() -> None:
    state = minimal_state()
    state["maps"] = {
        "1:mirage": {
            "count": 2,
            "wins": 1,
            "ct_sum": 1.1,
            "ct_n": 2,
            "t_sum": 0.9,
            "t_n": 2,
        }
    }
    row = prediction_features_from_state(
        state,
        team1_id=1,
        team2_id=2,
        match_time=pd.Timestamp("2026-04-12T12:00:00Z"),
        bo=3,
        location="lan",
        team1_rank=None,
        team2_rank=None,
        team1_players=[],
        team2_players=[],
        maps=["mirage", "ancient"],
    )

    assert row["diff_avg_map_count_before"] == pytest.approx(1.0)
    assert row["diff_series_maps_known"] == pytest.approx(1.0)


def test_fast_state_refuses_timestamp_inside_its_history() -> None:
    with pytest.raises(ValueError, match="valid only after"):
        prediction_features_from_state(
            minimal_state(),
            team1_id=1,
            team2_id=2,
            match_time=pd.Timestamp("2026-04-01T12:00:00Z"),
            bo=3,
            location="online",
            team1_rank=None,
            team2_rank=None,
            team1_players=[],
            team2_players=[],
            maps=[],
        )


def test_lineup_parser_rejects_duplicate_players() -> None:
    with pytest.raises(Exception, match="unique"):
        parse_int_list("11,12,11")


def test_map_parser_normalizes_aliases_and_rejects_duplicates() -> None:
    assert parse_map_list("de_dust2, Mirage") == ["dust2", "mirage"]
    with pytest.raises(Exception, match="unique"):
        parse_map_list("dust 2,de_dust2")


def test_pre_veto_map_scenarios_cover_every_bo_combination() -> None:
    scenarios = generate_map_scenarios(
        ["mirage", "nuke", "ancient", "inferno", "dust2"],
        bo=3,
    )
    assert len(scenarios) == 10
    assert all(len(scenario) == 3 for scenario in scenarios)
    assert len({tuple(scenario) for scenario in scenarios}) == 10
    with pytest.raises(ValueError, match="At least 5"):
        generate_map_scenarios(["mirage", "nuke", "ancient"], bo=5)


def test_fast_state_uses_detailed_veto_history() -> None:
    state = minimal_state()
    state["veto"] = {
        "1:mirage": {"picked": 3, "removed": 1, "left_over": 0, "total": 4},
        "2:mirage": {"picked": 1, "removed": 2, "left_over": 1, "total": 4},
    }
    row = prediction_features_from_state(
        state,
        team1_id=1,
        team2_id=2,
        match_time=pd.Timestamp("2026-04-12T12:00:00Z"),
        bo=3,
        location="lan",
        team1_rank=None,
        team2_rank=None,
        team1_players=[],
        team2_players=[],
        maps=["mirage"],
        team1_picks=["mirage"],
        team2_picks=["mirage"],
    )
    assert row["diff_veto_pick_rate_before"] == pytest.approx(0.5)


def test_saved_catboost_artifact_returns_valid_probability() -> None:
    model_dir = Path("artifacts")
    model_path = model_dir / "catboost_model.cbm"
    schema_path = model_dir / "feature_schema.json"
    if not model_path.exists() or not schema_path.exists():
        pytest.skip("Local trained artifact is not present")

    from catboost import CatBoostClassifier

    schema = load_schema(model_dir)
    assert schema["feature_order"] == MODEL_FEATURES
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    frame = pd.DataFrame(
        [[np.nan for _ in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )
    probability = model.predict_proba(frame)[0]
    assert np.isfinite(probability).all()
    assert probability.sum() == pytest.approx(1.0)
    assert (probability >= 0).all() and (probability <= 1).all()
