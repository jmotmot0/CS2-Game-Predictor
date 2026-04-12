from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = PROJECT_ROOT / "data" / "raw" / "matches_raw.csv"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "interim" / "hltv_enriched"
PROFILE_DIR_DEFAULT = PROJECT_ROOT / "data" / "browser_profile" / "hltv"

MAPSTATS_ID_RE = re.compile(r"/mapstatsid/(\d+)/")
TEAM_ID_RE = re.compile(r"/team/(\d+)/")
PLAYER_ID_RE = re.compile(r"/player/(\d+)/")
EVENT_ID_RE = re.compile(r"/events/(\d+)/")
STATS_TEAM_ID_RE = re.compile(r"/stats/teams/(\d+)/")
STATS_PLAYER_ID_RE = re.compile(r"/stats/players/(\d+)/")

VETO_REMOVE_RE = re.compile(r"(\d+)\.\s+(.+?)\s+removed\s+(.+)")
VETO_PICK_RE = re.compile(r"(\d+)\.\s+(.+?)\s+picked\s+(.+)")
VETO_LEFTOVER_RE = re.compile(r"(\d+)\.\s+(.+?)\s+was left over")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Enrich HLTV matches with match pages and map stats using a real browser."
    )
    parser.add_argument(
        "--cdp-url",
        type=str,
        default="http://127.0.0.1:9222",
        help="CDP URL of a manually started Chrome.",
    )
    parser.add_argument("--input", type=str, default=str(RAW_DEFAULT))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--limit-matches", type=int, default=50)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--browser-profile-dir",
        type=str,
        default=str(PROFILE_DIR_DEFAULT),
        help="Persistent browser profile directory for Playwright.",
    )
    return parser.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def safe_text(node: Any) -> str:
    if node is None:
        return ""
    try:
        return safe_str(node.get_text(" ", strip=True))
    except Exception:
        return ""


def safe_int(value: Any) -> int | None:
    text = safe_str(value)
    if text in {"", "-", "—", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    text = safe_str(value)
    if text in {"", "-", "—", "null"}:
        return None
    text = text.replace("%", "").replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def extract_first_int(regex: re.Pattern[str], text: str) -> int | None:
    match = regex.search(text)
    return int(match.group(1)) if match else None


def parse_compound_stat(text: str) -> tuple[int | None, int | None]:
    text = safe_str(text)
    match = re.match(r"(\d+)\((\d+)\)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return safe_int(text), None


def parse_opkd(text: str) -> tuple[int | None, int | None]:
    text = safe_str(text)
    match = re.match(r"(\d+)\s*:\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class BrowserFetcher:
    def __init__(self, profile_dir: str) -> None:
        self.profile_dir = profile_dir
        self.playwright = None
        self.context = None
        self.page = None

    def __enter__(self) -> "BrowserFetcher":
        Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        self.playwright = sync_playwright().start()

        try:
            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                channel="chrome",
                headless=False,
                viewport={"width": 1440, "height": 1000},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized",
                ],
            )
        except Exception:
            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                headless=False,
                viewport={"width": 1440, "height": 1000},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized",
                ],
            )

        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = self.context.new_page()

        self.page.set_extra_http_headers(
            {
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.context is not None:
                self.context.close()
        finally:
            if self.playwright is not None:
                self.playwright.stop()

    def get_html(
        self,
        url: str,
        ready_selector: str,
        referer: str | None = None,
        timeout_ms: int = 45000,
    ) -> str:
        assert self.page is not None

        self.page.goto(
            url,
            wait_until="domcontentloaded",
            referer=referer,
            timeout=timeout_ms,
        )

        try:
            self.page.wait_for_selector(ready_selector, timeout=12000)
        except PlaywrightTimeoutError:
            print("\n[manual action]")
            print(f"Open in the shown browser window: {url}")
            print("If Cloudflare appears, complete it manually in the browser.")
            input("When the page is fully open, press Enter here... ")
            self.page.wait_for_selector(ready_selector, timeout=120000)

        return self.page.content()


def parse_match_meta(soup: BeautifulSoup, raw_row: dict[str, Any]) -> dict[str, Any]:
    team1_link = soup.select_one(".team1-gradient a[href*='/team/']")
    team2_link = soup.select_one(".team2-gradient a[href*='/team/']")
    date_el = soup.select_one(".timeAndEvent .date[data-unix]")
    event_link = soup.select_one(".timeAndEvent .event a[href*='/events/']")

    lineups = soup.select(".lineups .lineup.standard-box")

    team1_rank = None
    team2_rank = None
    if len(lineups) >= 1:
        rank1 = lineups[0].select_one(".teamRanking a")
        if rank1:
            m = re.search(r"#(\d+)", safe_text(rank1))
            team1_rank = int(m.group(1)) if m else None
    if len(lineups) >= 2:
        rank2 = lineups[1].select_one(".teamRanking a")
        if rank2:
            m = re.search(r"#(\d+)", safe_text(rank2))
            team2_rank = int(m.group(1)) if m else None

    pre_box = soup.select_one(".padding.preformatted-text")
    lines = []
    if pre_box is not None:
        lines = [safe_str(x) for x in pre_box.get_text("\n", strip=True).split("\n") if safe_str(x)]

    bo = None
    lan_online = ""
    match_context = ""
    forfeit_note = ""
    if lines:
        m = re.search(r"Best of (\d+)\s+\((LAN|Online)\)", lines[0])
        if m:
            bo = int(m.group(1))
            lan_online = m.group(2)

        context_lines = []
        note_lines = []
        for line in lines[1:]:
            if line.startswith("**"):
                note_lines.append(line.lstrip("* ").strip())
            elif line.startswith("*"):
                context_lines.append(line.lstrip("* ").strip())

        match_context = " | ".join(context_lines)
        forfeit_note = " | ".join(note_lines)

    match_unix_ms = safe_str(date_el.get("data-unix")) if date_el else ""
    match_datetime_utc = ""
    if match_unix_ms.isdigit():
        match_datetime_utc = datetime.fromtimestamp(
            int(match_unix_ms) / 1000, tz=timezone.utc
        ).isoformat()

    return {
        "match_id": safe_str(raw_row.get("match_id")),
        "match_date": safe_str(raw_row.get("match_date")),
        "match_datetime_utc": match_datetime_utc,
        "event_id": extract_first_int(EVENT_ID_RE, event_link.get("href", "")) if event_link else None,
        "event_name": safe_text(event_link) or safe_str(raw_row.get("event_name")),
        "team1": safe_text(soup.select_one(".team1-gradient .teamName")) or safe_str(raw_row.get("team1")),
        "team2": safe_text(soup.select_one(".team2-gradient .teamName")) or safe_str(raw_row.get("team2")),
        "team1_id": extract_first_int(TEAM_ID_RE, team1_link.get("href", "")) if team1_link else None,
        "team2_id": extract_first_int(TEAM_ID_RE, team2_link.get("href", "")) if team2_link else None,
        "team1_rank": team1_rank,
        "team2_rank": team2_rank,
        "bo": bo,
        "lan_online": lan_online,
        "match_context": match_context,
        "forfeit_note": forfeit_note,
        "winner_raw": safe_str(raw_row.get("winner")),
        "source_url": safe_str(raw_row.get("source_url")),
    }


def parse_lineups(soup: BeautifulSoup, match_id: str) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    rows: list[dict[str, Any]] = []
    roster_by_team_ordinal: dict[int, list[int]] = {1: [], 2: []}

    lineup_blocks = soup.select(".lineups .lineup.standard-box")
    for team_ordinal, block in enumerate(lineup_blocks[:2], start=1):
        team_link = block.select_one(".box-headline a[href*='/team/']")
        team_id = extract_first_int(TEAM_ID_RE, team_link.get("href", "")) if team_link else None
        team_name = safe_text(team_link)

        player_cells = block.select("[data-player-id]")
        for cell in player_cells:
            player_id = safe_int(cell.get("data-player-id"))
            player_name = safe_text(cell.select_one(".text-ellipsis"))
            country = safe_str(cell.select_one("img.flag").get("title")) if cell.select_one("img.flag") else ""

            if player_id is not None:
                roster_by_team_ordinal[team_ordinal].append(player_id)

            rows.append(
                {
                    "match_id": match_id,
                    "team_ordinal": team_ordinal,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player_name,
                    "country": country,
                }
            )

    return rows, roster_by_team_ordinal


def parse_vetoes(soup: BeautifulSoup, match_id: str) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    veto_rows: list[dict[str, Any]] = []
    picked_by_map: dict[str, str] = {}
    decider_maps: set[str] = set()

    veto_boxes = soup.select(".veto-box")
    if len(veto_boxes) < 2:
        return veto_rows, picked_by_map, decider_maps

    lines = [
        safe_text(div)
        for div in veto_boxes[1].select(".padding > div")
        if safe_text(div)
    ]

    for line in lines:
        step_number = None
        team_name = ""
        action = ""
        map_name = ""

        m_remove = VETO_REMOVE_RE.match(line)
        m_pick = VETO_PICK_RE.match(line)
        m_left = VETO_LEFTOVER_RE.match(line)

        if m_remove:
            step_number = int(m_remove.group(1))
            team_name = safe_str(m_remove.group(2))
            action = "removed"
            map_name = safe_str(m_remove.group(3))
        elif m_pick:
            step_number = int(m_pick.group(1))
            team_name = safe_str(m_pick.group(2))
            action = "picked"
            map_name = safe_str(m_pick.group(3))
            picked_by_map[map_name.lower()] = team_name
        elif m_left:
            step_number = int(m_left.group(1))
            action = "left_over"
            map_name = safe_str(m_left.group(2))
            decider_maps.add(map_name.lower())

        veto_rows.append(
            {
                "match_id": match_id,
                "step_number": step_number,
                "team_name": team_name,
                "action": action,
                "map_name": map_name,
                "raw_line": line,
            }
        )

    return veto_rows, picked_by_map, decider_maps


def parse_match_half_scores(mapholder: Any) -> dict[str, Any]:
    out = {
        "team1_ct_rounds": None,
        "team1_t_rounds": None,
        "team2_ct_rounds": None,
        "team2_t_rounds": None,
        "team1_starting_side": "",
    }

    spans = mapholder.select(".results-center-half-score span")
    side_spans: list[tuple[int, str]] = []

    for span in spans:
        value = safe_int(span.get_text(strip=True))
        classes = span.get("class", [])
        if value is None:
            continue
        if "ct" in classes:
            side_spans.append((value, "CT"))
        elif "t" in classes:
            side_spans.append((value, "T"))

    if len(side_spans) < 4:
        return out

    h1_left_val, h1_left_side = side_spans[0]
    h1_right_val, _ = side_spans[1]
    h2_left_val, _ = side_spans[2]
    h2_right_val, _ = side_spans[3]

    out["team1_starting_side"] = h1_left_side
    if h1_left_side == "CT":
        out["team1_ct_rounds"] = h1_left_val
        out["team1_t_rounds"] = h2_left_val
        out["team2_t_rounds"] = h1_right_val
        out["team2_ct_rounds"] = h2_right_val
    else:
        out["team1_t_rounds"] = h1_left_val
        out["team1_ct_rounds"] = h2_left_val
        out["team2_ct_rounds"] = h1_right_val
        out["team2_t_rounds"] = h2_right_val

    return out


def parse_match_maps(
    soup: BeautifulSoup,
    match_id: str,
    picked_by_map: dict[str, str],
    decider_maps: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, holder in enumerate(soup.select(".mapholder"), start=1):
        map_name = safe_text(holder.select_one(".mapname"))

        score_left = None
        score_right = None
        score_left_el = holder.select_one(".results-left .results-team-score")
        score_right_el = holder.select_one(".results-right .results-team-score")
        if score_left_el:
            score_left = safe_int(score_left_el.get_text(strip=True))
        if score_right_el:
            score_right = safe_int(score_right_el.get_text(strip=True))

        stats_link = holder.select_one("a.results-stats[href]")
        stats_href = safe_str(stats_link.get("href")) if stats_link else ""
        mapstatsid = extract_first_int(MAPSTATS_ID_RE, stats_href) if stats_href else None

        rows.append(
            {
                "match_id": match_id,
                "map_no": idx,
                "map_name": map_name,
                "played": holder.select_one(".results.played") is not None,
                "optional": holder.select_one(".results.optional") is not None,
                "is_default_forfeit_map": map_name.lower() == "default",
                "team1_map_score": score_left,
                "team2_map_score": score_right,
                "mapstatsid": mapstatsid,
                "stats_href": stats_href,
                "picked_by": picked_by_map.get(map_name.lower(), ""),
                "is_decider": map_name.lower() in decider_maps,
                **parse_match_half_scores(holder),
            }
        )

    return rows


def parse_map_stats(
    html: str,
    match_id: str,
    map_no: int,
    mapstatsid: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    soup = BeautifulSoup(html, "lxml")

    map_name = "Unknown"
    match_info_box = soup.select_one(".match-info-box")
    if match_info_box:
        for child in match_info_box.children:
            if isinstance(child, NavigableString) and safe_str(child):
                map_name = safe_str(child)
                break

    tl = soup.select_one(".team-left a")
    tr = soup.select_one(".team-right a")
    tl_name = safe_text(tl)
    tr_name = safe_text(tr)
    tl_id = extract_first_int(STATS_TEAM_ID_RE, tl.get("href", "")) if tl else None
    tr_id = extract_first_int(STATS_TEAM_ID_RE, tr.get("href", "")) if tr else None

    tl_score = safe_int(soup.select_one(".team-left .bold").get_text(strip=True) if soup.select_one(".team-left .bold") else "")
    tr_score = safe_int(soup.select_one(".team-right .bold").get_text(strip=True) if soup.select_one(".team-right .bold") else "")

    side_info = {
        "team_left_ct_rounds": None,
        "team_left_t_rounds": None,
        "team_right_ct_rounds": None,
        "team_right_t_rounds": None,
        "team_left_starting_side": "",
    }

    info_rows = soup.select(".match-info-row")
    if info_rows:
        right_div = info_rows[0].select_one(".right")
        if right_div:
            side_spans: list[tuple[int, str]] = []
            for span in right_div.select("span"):
                value = safe_int(span.get_text(strip=True))
                classes = span.get("class", [])
                if value is None:
                    continue
                if "ct-color" in classes:
                    side_spans.append((value, "CT"))
                elif "t-color" in classes:
                    side_spans.append((value, "T"))

            if len(side_spans) >= 4:
                h1_left_val, h1_left_side = side_spans[0]
                h1_right_val, _ = side_spans[1]
                h2_left_val, _ = side_spans[2]
                h2_right_val, _ = side_spans[3]

                side_info["team_left_starting_side"] = h1_left_side
                if h1_left_side == "CT":
                    side_info["team_left_ct_rounds"] = h1_left_val
                    side_info["team_left_t_rounds"] = h2_left_val
                    side_info["team_right_t_rounds"] = h1_right_val
                    side_info["team_right_ct_rounds"] = h2_right_val
                else:
                    side_info["team_left_t_rounds"] = h1_left_val
                    side_info["team_left_ct_rounds"] = h2_left_val
                    side_info["team_right_ct_rounds"] = h1_right_val
                    side_info["team_right_t_rounds"] = h2_right_val

    map_summary_row = {
        "match_id": match_id,
        "map_no": map_no,
        "mapstatsid": mapstatsid,
        "map_name_from_stats": map_name,
        "team_left_id": tl_id,
        "team_left_name": tl_name,
        "team_right_id": tr_id,
        "team_right_name": tr_name,
        "team_left_score_stats": tl_score,
        "team_right_score_stats": tr_score,
        **side_info,
    }

    player_rows: list[dict[str, Any]] = []
    tables = soup.select(".stats-table.totalstats")
    if not tables:
        tables = soup.select(".totalstats")

    team_ids = [tl_id, tr_id]
    for table_idx, table in enumerate(tables[:2]):
        team_id = team_ids[table_idx] if table_idx < len(team_ids) else None

        for row in table.select("tbody tr"):
            player_a = row.select_one("td.st-player a")
            if not player_a:
                continue

            player_name = safe_text(player_a)
            player_id = extract_first_int(STATS_PLAYER_ID_RE, player_a.get("href", ""))

            kills, hs_kills = parse_compound_stat(
                row.select_one("td.st-kills").get_text(strip=True) if row.select_one("td.st-kills") else ""
            )
            assists, flash_assists = parse_compound_stat(
                row.select_one("td.st-assists").get_text(strip=True) if row.select_one("td.st-assists") else ""
            )
            deaths, traded_deaths = parse_compound_stat(
                row.select_one("td.st-deaths").get_text(strip=True) if row.select_one("td.st-deaths") else ""
            )
            opening_kills, opening_deaths = parse_opkd(
                row.select_one("td.st-opkd").get_text(strip=True) if row.select_one("td.st-opkd") else ""
            )

            player_rows.append(
                {
                    "match_id": match_id,
                    "map_no": map_no,
                    "mapstatsid": mapstatsid,
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": player_name,
                    "kills": kills,
                    "hs_kills": hs_kills,
                    "assists": assists,
                    "flash_assists": flash_assists,
                    "deaths": deaths,
                    "traded_deaths": traded_deaths,
                    "adr": safe_float(
                        row.select_one("td.st-adr").get_text(strip=True) if row.select_one("td.st-adr") else ""
                    ),
                    "kast": safe_float(
                        row.select_one("td.st-kast").get_text(strip=True) if row.select_one("td.st-kast") else ""
                    ),
                    "opening_kills": opening_kills,
                    "opening_deaths": opening_deaths,
                    "rating": safe_float(
                        row.select_one("td.st-rating").get_text(strip=True) if row.select_one("td.st-rating") else ""
                    ),
                    "multi_kills": safe_int(
                        row.select_one("td.st-mks").get_text(strip=True) if row.select_one("td.st-mks") else ""
                    ),
                    "clutch_wins": safe_int(
                        row.select_one("td.st-clutches").get_text(strip=True) if row.select_one("td.st-clutches") else ""
                    ),
                    "round_swing": safe_float(
                        row.select_one("td.st-roundSwing").get_text(strip=True) if row.select_one("td.st-roundSwing") else ""
                    ),
                }
            )

    return map_summary_row, player_rows


def save_df(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, dtype={"match_id": str})
    raw = raw.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    if args.start_offset > 0:
        raw = raw.iloc[args.start_offset:].copy()
    if args.limit_matches > 0:
        raw = raw.iloc[: args.limit_matches].copy()

    matches_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    veto_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    with BrowserFetcher(args.browser_profile_dir) as browser:
        for i, raw_row in enumerate(raw.to_dict(orient="records"), start=1):
            match_id = safe_str(raw_row.get("match_id"))
            match_url = safe_str(raw_row.get("source_url"))

            print(f"[{i}/{len(raw)}] match_id={match_id}")

            try:
                match_html = browser.get_html(
                    url=match_url,
                    ready_selector=".team1-gradient .teamName",
                    referer="https://www.hltv.org/results",
                )
                soup = BeautifulSoup(match_html, "lxml")

                meta = parse_match_meta(soup, raw_row)
                lineups, roster_by_team = parse_lineups(soup, match_id)
                vetos, picked_by_map, decider_maps = parse_vetoes(soup, match_id)
                maps = parse_match_maps(soup, match_id, picked_by_map, decider_maps)

                meta["team1_roster_hash"] = "-".join(map(str, sorted(roster_by_team.get(1, []))))
                meta["team2_roster_hash"] = "-".join(map(str, sorted(roster_by_team.get(2, []))))

                matches_rows.append(meta)
                lineup_rows.extend(lineups)
                veto_rows.extend(vetos)
                map_rows.extend(maps)

                for map_row in maps:
                    mapstatsid = map_row.get("mapstatsid")
                    stats_href = safe_str(map_row.get("stats_href"))

                    if not mapstatsid:
                        continue

                    if stats_href:
                        stats_url = (
                            stats_href if stats_href.startswith("http")
                            else f"https://www.hltv.org{stats_href}"
                        )
                    else:
                        stats_url = f"https://www.hltv.org/stats/matches/mapstatsid/{int(mapstatsid)}/x"

                    stats_html = browser.get_html(
                        url=stats_url,
                        ready_selector=".stats-table.totalstats td.st-player, .totalstats td.st-player",
                        referer=match_url,
                    )

                    stats_map_row, stats_players = parse_map_stats(
                        html=stats_html,
                        match_id=match_id,
                        map_no=int(map_row["map_no"]),
                        mapstatsid=int(mapstatsid),
                    )

                    for key, value in stats_map_row.items():
                        if key not in {"match_id", "map_no", "mapstatsid"} and value not in {"", None}:
                            map_row[key] = value

                    player_rows.extend(stats_players)

            except Exception as exc:
                print(f"[skip] match_id={match_id} -> {exc}")

    save_df(matches_rows, out_dir / "matches_enriched.csv")
    save_df(lineup_rows, out_dir / "match_lineups.csv")
    save_df(veto_rows, out_dir / "veto_steps.csv")
    save_df(map_rows, out_dir / "match_maps.csv")
    save_df(player_rows, out_dir / "map_player_stats.csv")

    print(f"\nSaved files to: {out_dir}")
    print(f"matches_enriched.csv: {len(matches_rows)} rows")
    print(f"match_lineups.csv: {len(lineup_rows)} rows")
    print(f"veto_steps.csv: {len(veto_rows)} rows")
    print(f"match_maps.csv: {len(map_rows)} rows")
    print(f"map_player_stats.csv: {len(player_rows)} rows")


if __name__ == "__main__":
    main()