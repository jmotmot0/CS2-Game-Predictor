from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.enrich_hltv_matches import (
    parse_lineups,
    parse_map_stats,
    parse_match_maps,
    parse_match_meta,
    parse_vetoes,
    validate_map_stats,
    validate_match_bundle,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_saved_match_page_fixture_parses_critical_entities() -> None:
    soup = BeautifulSoup((FIXTURES / "hltv_match.html").read_text(encoding="utf-8"), "lxml")
    raw = {
        "match_id": "123",
        "match_date": "2025-06-01",
        "winner": "Alpha",
        "source_url": "https://www.hltv.org/matches/123/test",
    }
    meta = parse_match_meta(soup, raw)
    lineups, rosters = parse_lineups(soup, "123")
    vetoes, picked_by_map, deciders = parse_vetoes(soup, "123")
    maps = parse_match_maps(soup, "123", picked_by_map, deciders)

    validate_match_bundle(meta, lineups, maps)
    assert meta["team1_id"] == 1 and meta["team2_id"] == 2
    assert meta["team1_rank"] == 5 and meta["team2_rank"] == 12
    assert meta["bo"] == 3 and meta["lan_online"] == "LAN"
    assert rosters == {1: [101, 102, 103, 104, 105], 2: [201, 202, 203, 204, 205]}
    assert [row["action"] for row in vetoes] == ["removed", "picked", "left_over"]
    assert maps[0]["mapstatsid"] == 777
    assert maps[0]["picked_by"] == "Bravo"
    assert maps[0]["team1_ct_rounds"] == 7
    assert maps[0]["team2_ct_rounds"] == 4


def test_saved_mapstats_fixture_parses_players_and_sides() -> None:
    html = (FIXTURES / "hltv_mapstats.html").read_text(encoding="utf-8")
    summary, players = parse_map_stats(html, "123", 1, 777)

    validate_map_stats(summary, players)
    assert summary["map_name_from_stats"] == "Mirage"
    assert summary["team_left_id"] == 1 and summary["team_right_id"] == 2
    assert summary["team_left_ct_rounds"] == 7
    assert summary["team_right_ct_rounds"] == 4
    assert len(players) == 10
    assert players[0]["player_id"] == 101
    assert players[0]["kills"] == 20 and players[0]["hs_kills"] == 10
    assert players[0]["opening_kills"] == 4 and players[0]["opening_deaths"] == 2
