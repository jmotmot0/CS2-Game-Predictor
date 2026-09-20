"""Compare a fresh unlabelled forecast snapshot to the frozen research row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit

from src.feature_engineering import normalize_map_name, normalize_veto
from src.predict_team_ranking import (
    DEFAULT_CLEAN_DIR, DEFAULT_MODEL_DIR, historical_prediction,
    prepare_forecast_features, verified_bundle,
)
from src.team_ranking import EXTENDED_TEAM_FEATURES, team_ranking_scores, team_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--match-id", type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    model, data_path, metadata = verified_bundle(args.model_dir)
    example = json.loads((args.model_dir / "illustrative_match.json").read_text(encoding="utf-8"))
    match_id = args.match_id if args.match_id is not None else int(example["match_id"])
    data = pd.read_csv(data_path)
    stored = data.loc[data.match_id.eq(match_id)].copy()
    if len(stored) != 1:
        raise ValueError("Exactly one stored example is required")
    row = stored.iloc[0]
    roster_table = pd.read_csv(args.clean_dir / "match_lineups.csv")
    current_lineups = roster_table.loc[roster_table.match_id.eq(match_id)]
    veto_table = pd.read_csv(args.clean_dir / "veto_steps.csv")
    current_veto = normalize_veto(veto_table.loc[veto_table.match_id.eq(match_id)], stored)
    if "step_number" in current_veto:
        current_veto = current_veto.sort_values("step_number")
    lists = {}
    for number in (1, 2):
        team_id = int(row[f"team{number}_id"])
        lists[f"team{number}_players"] = sorted(current_lineups.loc[
            current_lineups.team_id.eq(team_id), "player_id"].astype(int).unique().tolist())
        for suffix, action in (("picks", "picked"), ("removes", "removed")):
            lists[f"team{number}_{suffix}"] = current_veto.loc[
                current_veto.team_id.eq(team_id) & current_veto.action.eq(action), "map_name"
            ].drop_duplicates().tolist()
    decider = current_veto.loc[current_veto.action.eq("left_over"), "map_name"].drop_duplicates().tolist()
    maps = current_veto.loc[current_veto.action.isin(["picked", "left_over"]), "map_name"].drop_duplicates().tolist()
    rebuilt, audit = prepare_forecast_features(
        team1_id=int(row.team1_id), team2_id=int(row.team2_id),
        match_time=row.match_datetime_utc, clean_dir=args.clean_dir,
        protocol=metadata["protocol"], bo=int(row.bo),
        location="lan" if int(row.is_lan) else "online",
        team1_rank=float(row.team1_team_rank), team2_rank=float(row.team2_team_rank),
        maps=[normalize_map_name(name) for name in maps], decider=decider, **lists,
    )
    original_objects = team_rows(stored, EXTENDED_TEAM_FEATURES)
    rebuilt_objects = team_rows(rebuilt, EXTENDED_TEAM_FEATURES)
    np.testing.assert_allclose(original_objects, rebuilt_objects, rtol=1e-10, atol=1e-10, equal_nan=True)
    original_scores = team_ranking_scores(model, stored, EXTENDED_TEAM_FEATURES)
    rebuilt_scores = team_ranking_scores(model, rebuilt, EXTENDED_TEAM_FEATURES)
    np.testing.assert_allclose(original_scores, rebuilt_scores, rtol=0, atol=1e-12)
    reproduction = historical_prediction(match_id, args.model_dir)
    probability = float(expit(rebuilt_scores[0, 0] - rebuilt_scores[0, 1]))
    assert abs(probability - reproduction["teams"][0]["win_probability"]) < 1e-12
    if match_id == example.get("match_id"):
        assert abs(probability - example["probabilities_team1"]["CTIS"]) < 1e-12
    assert pd.isna(rebuilt.iloc[0].team1_win)
    assert pd.isna(rebuilt.iloc[0].team1_score) and pd.isna(rebuilt.iloc[0].team2_score)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({
        "match_id": match_id, "all_46_own_features_reproduced": True,
        "scores_reproduced": True, "probability_reproduced": True,
        "pending_target_and_scores_are_missing": True,
        "teams": [str(row.team1), str(row.team2)],
        "probability_team1": probability,
        "history_series_count": audit["history_series_count"],
        "seconds": time.perf_counter() - started,
        "warnings": audit["warnings"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
