from __future__ import annotations

from src.enrich_hltv_matches import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_discards_uncommitted_child_rows(tmp_path) -> None:
    save_checkpoint(
        tmp_path,
        matches_rows=[{"match_id": "1", "team1_id": 10}],
        lineup_rows=[
            {"match_id": "1", "player_id": 100},
            {"match_id": "2", "player_id": 200},
        ],
        veto_rows=[{"match_id": "2", "action": "picked"}],
        map_rows=[{"match_id": "1", "map_no": 1}],
        player_rows=[{"match_id": "2", "player_id": 200}],
        failure_rows=[{"match_id": "2", "error": "temporary"}],
    )

    matches, lineups, vetoes, maps, players, failures = load_checkpoint(tmp_path)

    assert [row["match_id"] for row in matches] == ["1"]
    assert [row["match_id"] for row in lineups] == ["1"]
    assert vetoes == []
    assert [row["match_id"] for row in maps] == ["1"]
    assert players == []
    assert [row["match_id"] for row in failures] == ["2"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_replaces_previous_snapshot(tmp_path) -> None:
    arguments = {
        "out_dir": tmp_path,
        "lineup_rows": [],
        "veto_rows": [],
        "map_rows": [],
        "player_rows": [],
        "failure_rows": [],
    }
    save_checkpoint(matches_rows=[{"match_id": "1"}], **arguments)
    save_checkpoint(matches_rows=[{"match_id": "2"}], **arguments)

    matches, *_ = load_checkpoint(tmp_path)

    assert [row["match_id"] for row in matches] == ["2"]
