from __future__ import annotations

import sys

import pandas as pd
import pytest

from src import enrich_hltv_matches as enrich
from src.enrich_hltv_matches import load_checkpoint, save_checkpoint
from src.hltv_browser import BrowserActionRequired


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


def test_site_challenge_stops_batch_and_preserves_completed_matches(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame([
        {"match_id": str(number), "match_date": "2026-09-27", "source_url": f"https://www.hltv.org/matches/{number}/x"}
        for number in [1, 2, 3]
    ]).to_csv(raw_path, index=False)
    out = tmp_path / "enriched"
    out.mkdir()
    save_checkpoint(out, [{"match_id": "1", "team1_id": 10}], [], [], [], [], [])
    requested = []

    class BlockedBrowser:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_html(self, **kwargs):
            requested.append(kwargs["url"])
            raise BrowserActionRequired("challenge")

    monkeypatch.setattr(enrich, "BrowserFetcher", BlockedBrowser)
    monkeypatch.setattr(sys, "argv", [
        "enrich", "--input", str(raw_path), "--out-dir", str(out), "--limit-matches", "0", "--resume",
        "--sleep-min", "0", "--sleep-max", "0",
    ])
    with pytest.raises(BrowserActionRequired):
        enrich.main()
    assert requested == ["https://www.hltv.org/matches/2/x"]
    matches, *_, failures = load_checkpoint(out)
    assert [row["match_id"] for row in matches] == ["1"]
    assert failures == []


def test_unresolved_match_errors_fail_run_after_saving(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame([{
        "match_id": "2", "match_date": "2026-09-27", "source_url": "https://www.hltv.org/matches/2/x",
    }]).to_csv(raw_path, index=False)

    class BrokenPageBrowser:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_html(self, **kwargs):
            raise ValueError("unparseable match page")

    out = tmp_path / "enriched"
    monkeypatch.setattr(enrich, "BrowserFetcher", BrokenPageBrowser)
    monkeypatch.setattr(sys, "argv", [
        "enrich", "--input", str(raw_path), "--out-dir", str(out), "--limit-matches", "0",
        "--sleep-min", "0", "--sleep-max", "0",
    ])
    with pytest.raises(RuntimeError, match="1 unresolved failed matches"):
        enrich.main()
    matches, *_, failures = load_checkpoint(out)
    assert matches == []
    assert len(failures) == 1 and failures[0]["match_id"] == "2"
