import json

import numpy as np
import pandas as pd
import pytest

from src.modeling import sha256_file
from src.predict_team_ranking import (
    historical_prediction, verified_bundle, prepare_forecast_features, forecast_prediction, main,
)
from src.team_ranking import EXTENDED_TEAM_FEATURES, SHARED_FEATURES, team_ranking_pool


@pytest.fixture
def frozen_bundle(tmp_path):
    from catboost import CatBoostRanker

    frame = pd.DataFrame({
        "match_id": [100, 101, 102, 103],
        "match_datetime_utc": ["2026-01-01T12:00:00Z"] * 4,
        "team1_id": [10, 20, 10, 20], "team2_id": [20, 10, 20, 10],
        "team1": ["A", "B", "A", "B"], "team2": ["B", "A", "B", "A"],
        "team1_win": [1, 0, 1, 0], "team1_score": [2, 0, 2, 0], "team2_score": [0, 2, 0, 2],
    })
    for name in EXTENDED_TEAM_FEATURES:
        if name in SHARED_FEATURES:
            frame[name] = float(name in ["bo3", "is_online", "rank_available"])
        else:
            suffix = "team_rank" if name == "rank_score" else name
            frame[f"team1_{suffix}"] = [1., 2., 1., 2.]
            frame[f"team2_{suffix}"] = [2., 1., 2., 1.]
    frame["team1_elo_pre"] = [1700., 1500., 1800., 1400.]
    frame["team2_elo_pre"] = [1500., 1700., 1400., 1800.]
    frame.to_csv(tmp_path / "extended_features.csv", index=False)
    model = CatBoostRanker(loss_function="PairLogit", iterations=5, depth=2, verbose=False,
                          allow_writing_files=False, random_seed=42, thread_count=2)
    model.fit(team_ranking_pool(frame, frame.team1_win.to_numpy(), EXTENDED_TEAM_FEATURES))
    model.save_model(str(tmp_path / "ranking_46_s42.cbm"))
    schema = {"features": EXTENDED_TEAM_FEATURES, "feature_count": 46,
              "params": {"loss_function": "PairLogit"}}
    (tmp_path / "ranking_46_s42.json").write_text(json.dumps(schema), encoding="utf-8")
    (tmp_path / "protocol.json").write_text(json.dumps({"prediction_time": "Before the first map"}), encoding="utf-8")
    (tmp_path / "results_manifest.json").write_text(json.dumps({
        "files": {path.name: sha256_file(path) for path in tmp_path.iterdir() if path.is_file()}
    }), encoding="utf-8")
    return tmp_path


def test_historical_command_uses_frozen_model_and_consistent_schema(frozen_bundle):
    result = historical_prediction(100, frozen_bundle)
    assert result["mode"] == "historical_offline_reproduction"
    assert result["feature_count_per_team"] == 46 and result["loss_function"] == "PairLogit"
    assert {name: item["count"] for name, item in result["own_feature_groups"].items()} == {
        "individual": 7, "team": 21, "cohesion": 5, "controls": 13,
    }
    assert np.isclose(sum(team["win_probability"] for team in result["teams"]), 1.)
    assert len(result["integrity"]["model_sha256"]) == 64
    assert result["match_datetime_utc"] == "2026-01-01T12:00:00+00:00"
    with pytest.raises(ValueError, match="found 0"):
        historical_prediction(999, frozen_bundle)


def test_tampered_data_or_model_is_rejected_before_inference(frozen_bundle):
    path = frozen_bundle / "extended_features.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="integrity check"):
        historical_prediction(100, frozen_bundle)


def test_manifest_path_traversal_is_rejected(frozen_bundle):
    manifest = frozen_bundle / "results_manifest.json"
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content["files"]["../escaped.txt"] = "0" * 64
    manifest.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        verified_bundle(frozen_bundle)


def test_data_override_must_be_an_exact_verified_copy(frozen_bundle, tmp_path_factory):
    folder = tmp_path_factory.mktemp("copy")
    copy = folder / "copy.csv"
    copy.write_bytes((frozen_bundle / "extended_features.csv").read_bytes())
    assert historical_prediction(100, frozen_bundle, copy)["match_id"] == 100
    copy.write_bytes(copy.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="supplied data"):
        historical_prediction(100, frozen_bundle, copy)


def test_schema_feature_order_must_match_canonical_order_even_with_new_hash(frozen_bundle):
    schema_path = frozen_bundle / "ranking_46_s42.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["features"] = list(reversed(schema["features"]))
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    manifest_path = frozen_bundle / "results_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][schema_path.name] = sha256_file(schema_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        historical_prediction(100, frozen_bundle)


def test_verify_cli_reports_success_without_predicting(frozen_bundle, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["predict_team_ranking", "--verify", "--model-dir", str(frozen_bundle)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "frozen_bundle_verification" and result["verified"] is True
    assert result["verified_files"] == 4


@pytest.fixture
def tiny_clean_history(tmp_path):
    matches, lineups, maps, stats, veto = [], [], [], [], []
    # Шестая серия позже прогноза; её показатели намеренно завышены.
    # Их изменение не должно влиять на предматчевые признаки.
    for match_id, day in enumerate([1, 2, 3, 4, 5, 7], start=1):
        matches.append({
            "match_id": match_id, "match_date": f"2026-01-{day:02}",
            "match_datetime_utc": f"2026-01-{day:02}T12:00:00Z",
            "team1_id": 10, "team2_id": 20, "team1": "A", "team2": "B",
            "team1_rank": 5, "team2_rank": 10, "team1_win": 1,
            "team1_score": 2, "team2_score": 0, "is_valid_result": 1,
            "bo": "bo3", "lan_online": "online",
        })
        maps.append({"match_id": match_id, "map_no": 1, "map_name": "mirage",
                     "team1_map_score": 13, "team2_map_score": 7})
        veto.append({"match_id": match_id, "action": "left_over", "map_name": "mirage"})
        for player in range(1, 11):
            team = 10 if player <= 5 else 20
            lineups.append({"match_id": match_id, "team_id": team, "player_id": player})
            stats.append({"match_id": match_id, "map_no": 1, "team_id": team,
                          "player_id": player, "rating": 99. if match_id == 6 else 1 + player / 100,
                          "adr": 75., "kast": 70., "opening_kills": 2, "opening_deaths": 1})
    for name, rows in {"matches_final.csv": matches, "match_lineups.csv": lineups,
                       "match_maps.csv": maps, "map_player_stats.csv": stats,
                       "veto_steps.csv": veto}.items():
        pd.DataFrame(rows).to_csv(tmp_path / name, index=False)
    return tmp_path


def forecast_inputs(clean_dir):
    return dict(team1_id=10, team2_id=20, match_time="2026-01-06T12:00:00Z",
                clean_dir=clean_dir, bo=3, location="online",
                team1_rank=5, team2_rank=10,
                team1_players=[1, 2, 3, 4, 5], team2_players=[6, 7, 8, 9, 10],
                maps=["mirage", "nuke", "ancient"],
                team1_picks=["mirage"], team2_picks=["nuke"], decider=["ancient"])


def test_forecast_rebuild_excludes_future_stats_and_retains_unlabelled_target(tiny_clean_history):
    row, audit = prepare_forecast_features(**forecast_inputs(tiny_clean_history))
    assert len(row) == 1 and row.iloc[0].match_id == -1
    assert pd.isna(row.iloc[0].team1_win)
    assert pd.isna(row.iloc[0].team1_score) and pd.isna(row.iloc[0].team2_score)
    assert row.iloc[0].team1_matches_before == 5
    assert np.isclose(row.iloc[0].team1_lineup_player_rating_max, 1.05)
    assert np.isclose(row.iloc[0].team2_lineup_player_rating_min, 1.06)
    assert row.iloc[0].team1_roster_same_matches_90d == 5
    assert row.iloc[0].team1_roster_consecutive_before == 5
    assert row.iloc[0].team1_roster_pair_experience_90d == 5
    assert audit["history_series_count"] == 5
    assert audit["history_latest_start_utc"] == "2026-01-05T12:00:00+00:00"
    assert audit["model_selection_available_from_utc"] == "2026-01-01T00:00:00+00:00"


def test_forecast_missing_current_inputs_are_unknown_not_empty_equal_rosters(tiny_clean_history):
    inputs = forecast_inputs(tiny_clean_history)
    inputs.update(team1_players=[], team1_rank=None, maps=[],
                  team1_picks=[], team2_picks=[], decider=[])
    row, audit = prepare_forecast_features(**inputs)
    assert pd.isna(row.iloc[0].team1_roster_size)
    assert pd.isna(row.iloc[0].team1_lineup_player_rating_mean)
    assert pd.isna(row.iloc[0].team1_roster_same_matches_90d)
    assert pd.isna(row.iloc[0].team1_team_rank)
    assert pd.isna(row.iloc[0].team1_avg_map_count_before)
    assert row.iloc[0].rank_available == 0
    assert any("no historical rank is substituted" in warning for warning in audit["warnings"])


def test_forecast_rejects_backdated_naive_or_impossible_current_inputs(tiny_clean_history):
    for updates, message in [
        ({"match_time": "2025-05-01T12:00:00Z"}, "training cutoff"),
        ({"match_time": "2025-08-01T12:00:00Z"}, "model-selection cutoff"),
        ({"match_time": "2026-01-06T12:00:00"}, "explicit UTC offset"),
        ({"team1_players": [1, 2]}, "exactly five"),
        ({"team2_players": [1, 6, 7, 8, 9]}, "share a player"),
        ({"maps": ["mirage"]}, "exactly BO"),
        ({"team1_removes": ["ancient"]}, "removed maps"),
    ]:
        inputs = forecast_inputs(tiny_clean_history)
        inputs.update(updates)
        with pytest.raises(ValueError, match=message):
            prepare_forecast_features(**inputs)


def test_frozen_ranker_runs_new_forecast_without_training_or_current_outcome(frozen_bundle, tiny_clean_history):
    result = forecast_prediction(model_dir=frozen_bundle, **forecast_inputs(tiny_clean_history))
    assert result["mode"] == "new_pre_match_snapshot"
    assert result["loss_function"] == "PairLogit" and result["feature_count_per_team"] == 46
    assert result["predicted_winner_id"] in (10, 20)
    assert np.isclose(sum(team["win_probability"] for team in result["teams"]), 1.)
    assert all(0 <= team["feature_coverage"] <= 1 for team in result["teams"])
    assert result["history_series_count"] == 5
    assert len(result["clean_table_sha256"]) == 5
