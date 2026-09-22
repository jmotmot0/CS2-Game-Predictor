from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from contextlib import ExitStack
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "matches_raw.csv"
PROFILE_DIR = PROJECT_ROOT / "data" / "browser_profile" / "hltv"
RAW_COLUMNS = [
    "match_id", "match_date", "event_id", "event_name", "team1", "team2",
    "team1_score", "team2_score", "winner", "game", "completed",
    "is_professional_proxy", "professional_filter_note", "source", "source_url",
]

CS2_START_DATE = date(2023, 9, 27)
RESULTS_URL = "https://www.hltv.org/results"

BAD_TEAM_NAMES = {"TBD", "Unknown", "", "-", "—"}

MATCH_ID_RE = re.compile(r"/matches/(\d+)")
ORDINAL_DAY_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)")
PAGINATION_RE = re.compile(r"^(\d+)\s*-\s*(\d+)\s+of\s+(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load historical CS2 match results from HLTV results pages into CSV."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-09-27",
        help="Start date in YYYY-MM-DD format. Default: 2023-09-27",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=date.today().isoformat(),
        help=f"End date in YYYY-MM-DD format. Default: {date.today().isoformat()}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Maximum new pages in this run; reaching the limit is not completion. Default: 500",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=1.5,
        help="Minimum delay between page requests. Default: 1.5",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=3.5,
        help="Maximum delay between page requests. Default: 3.5",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="HTTP transport retries per page request. Default: 5",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request/navigation timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.getenv("HLTV_PROXY", ""),
        help="Optional proxy URL, e.g. http://user:pass@host:port. "
             "By default uses HLTV_PROXY env var if set.",
    )
    parser.add_argument("--transport", choices=["browser", "http"], default="browser")
    parser.add_argument("--browser-profile-dir", default=str(PROFILE_DIR))
    parser.add_argument("--cdp-url", default="")
    parser.add_argument("--manual-wait-seconds", type=int, default=180)
    parser.add_argument("--checkpoint", help="JSON progress file; default: OUTPUT.checkpoint.json")
    parser.add_argument("--resume", action="store_true", help="Resume the same date window from its checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of the final output CSV.")
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def safe_int(value: Any) -> int | None:
    text = safe_str(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_result_headline_date(text: str) -> date | None:
    text = safe_str(text)
    if not text:
        return None

    # "Results for February 14th 2026" -> "February 14 2026"
    text = text.replace("Results for", "").strip()
    text = ORDINAL_DAY_RE.sub(r"\1", text)

    try:
        return datetime.strptime(text, "%B %d %Y").date()
    except ValueError:
        return None


def winner_from_scores(
    team1: str,
    team2: str,
    score1: int | None,
    score2: int | None,
) -> str:
    if score1 is None or score2 is None:
        return ""
    if score1 > score2:
        return team1
    if score2 > score1:
        return team2
    return ""


def looks_valid_match(row: dict[str, Any], start_date: date, end_date: date) -> bool:
    match_date = row.get("match_date")
    team1 = safe_str(row.get("team1"))
    team2 = safe_str(row.get("team2"))
    score1 = row.get("team1_score")
    score2 = row.get("team2_score")
    winner = safe_str(row.get("winner"))

    if match_date is None:
        return False
    if match_date < start_date or match_date > end_date:
        return False

    if team1 in BAD_TEAM_NAMES or team2 in BAD_TEAM_NAMES:
        return False
    if team1 == team2:
        return False

    if score1 is None or score2 is None:
        return False
    if score1 == score2:
        return False

    if not winner:
        return False

    return True


def build_session(proxy: str | None) -> cffi_requests.Session:
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}

    session = cffi_requests.Session(
        impersonate="chrome120",
        proxies=proxies,
    )
    session.headers.update(
        {
            "Referer": RESULTS_URL,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def fetch_html(
    session: cffi_requests.Session,
    url: str,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    timeout: int,
) -> str:
    last_error: str = ""

    for attempt in range(1, retries + 1):
        time.sleep(random.uniform(sleep_min, sleep_max))

        try:
            response = session.get(url, timeout=timeout)
            status = response.status_code

            if status == 200:
                html = response.text

                # Вместо результатов иногда возвращается HTML проверки Cloudflare.
                if (
                    "challenge-error-title" in html
                    or "Enable JavaScript and cookies to continue" in html
                ):
                    last_error = "Cloudflare challenge page returned"
                else:
                    return html
            else:
                last_error = f"HTTP {status}"

        except Exception as exc:
            last_error = repr(exc)

        backoff = min(30, 2 ** attempt)
        print(f"[retry {attempt}/{retries}] {url} -> {last_error}; sleep {backoff}s")
        time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def get_regular_results_container(soup: BeautifulSoup):
    # На первой странице HLTV избранные результаты могут дублировать основной список.
    # Последний блок .results-all содержит основной список матчей.
    containers = soup.select(".results-all")
    if containers:
        return containers[-1]

    # Поддержка прежних вариантов разметки.
    legacy = soup.select("div.allres")
    if legacy:
        return legacy[-1]

    raise RuntimeError("Could not find results container on page.")


def extract_rows_from_page(
    html: str,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], date | None, date | None]:
    soup = BeautifulSoup(html, "lxml")
    container = get_regular_results_container(soup)

    rows: list[dict[str, Any]] = []
    page_dates: list[date] = []

    for sublist in container.select(".results-sublist"):
        headline_el = sublist.select_one(".standard-headline")
        group_date = parse_result_headline_date(headline_el.get_text(" ", strip=True)) if headline_el else None

        entries = sublist.select(".result-con")
        if not entries:
            # Поддержка старой разметки с последовательным разбором ссылок.
            entries = sublist.select("a.a-reset")

        for entry in entries:
            href_el = entry.select_one("a.a-reset[href]") if entry.name != "a" else entry
            if href_el is None:
                continue

            href = safe_str(href_el.get("href"))
            match_id_match = MATCH_ID_RE.search(href)
            if not match_id_match:
                continue

            match_id = match_id_match.group(1)

            team1_el = entry.select_one(".team1 .team")
            team2_el = entry.select_one(".team2 .team")
            event_el = entry.select_one(".event-name")
            score_spans = entry.select(".result-score span")

            team1 = safe_str(team1_el.get_text(" ", strip=True) if team1_el else "")
            team2 = safe_str(team2_el.get_text(" ", strip=True) if team2_el else "")
            event_name = safe_str(event_el.get_text(" ", strip=True) if event_el else "")

            timestamp_ms = safe_str(entry.get("data-zonedgrouping-entry-unix"))
            match_date: date | None = None

            if timestamp_ms.isdigit():
                match_date = datetime.fromtimestamp(
                    int(timestamp_ms) / 1000,
                    tz=timezone.utc,
                ).date()
            elif group_date is not None:
                match_date = group_date

            if match_date is not None:
                page_dates.append(match_date)
            else:
                raise RuntimeError(f"Missing date for match {match_id}; results markup may have changed.")

            if not team1_el or not team2_el or len(score_spans) < 2:
                raise RuntimeError(f"Missing teams or scores for match {match_id}; refusing a partial page.")
            score1 = safe_int(score_spans[0].get_text(strip=True))
            score2 = safe_int(score_spans[1].get_text(strip=True))

            row = {
                "match_id": match_id,
                "match_date": match_date,
                "event_id": "",
                "event_name": event_name,
                "team1": team1,
                "team2": team2,
                "team1_score": score1,
                "team2_score": score2,
                "winner": winner_from_scores(team1, team2, score1, score2),
                "game": "CS2",
                "completed": True,
                "is_professional_proxy": True,
                "professional_filter_note": "HLTV results page scrape",
                "source": "hltv",
                "source_url": f"https://www.hltv.org{href}",
            }

            if looks_valid_match(row, start_date=start_date, end_date=end_date):
                rows.append(row)

    page_newest = max(page_dates) if page_dates else None
    page_oldest = min(page_dates) if page_dates else None
    return rows, page_newest, page_oldest


def results_page_info(html: str, offset: int) -> tuple[list[str], int]:
    """Validate pagination independently of rows rejected by the match filters."""
    soup = BeautifulSoup(html, "lxml")
    container = get_regular_results_container(soup)
    match_ids = []
    for link in container.select(".results-sublist a.a-reset[href]"):
        match = MATCH_ID_RE.search(link.get("href", ""))
        if match:
            match_ids.append(match.group(1))
    ranges = set()
    for value in soup.stripped_strings:
        match = PAGINATION_RE.fullmatch(value)
        if match:
            ranges.add(tuple(map(int, match.groups())))
    if len(ranges) != 1:
        raise RuntimeError("Missing or inconsistent results pagination; cannot prove collection completeness.")
    first, last, total = ranges.pop()
    if not match_ids or len(set(match_ids)) != len(match_ids):
        raise RuntimeError("Empty or duplicate results page; refusing to mark collection complete.")
    if first != offset + 1 or last != min(offset + 100, total) or last - first + 1 != len(match_ids):
        raise RuntimeError("Results count does not match pagination; page may be partial or reordered.")
    return match_ids, total


def save_progress(path: Path, state: dict[str, Any]) -> None:
    """Rows and the next offset are committed together, never in separate files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_progress(path: Path, start_date: date, end_date: date) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != 1 or state.get("start_date") != str(start_date) or state.get("end_date") != str(end_date):
        raise ValueError("Checkpoint version or dates differ. Use the original dates or a new output/checkpoint.")
    offset = state.get("next_offset")
    if not isinstance(offset, int) or offset < 0 or offset % 100:
        raise ValueError("Invalid checkpoint offset")
    if not isinstance(state.get("rows"), list) or not isinstance(state.get("seen_ids"), list):
        raise ValueError("Invalid checkpoint rows")
    seen_ids = state["seen_ids"]
    seen_set = set(seen_ids)
    if len(seen_set) != len(seen_ids):
        raise ValueError("Duplicate checkpoint match IDs")
    if state.get("complete"):
        if not seen_ids or len(seen_ids) != state.get("total"):
            raise ValueError("Invalid complete checkpoint")
    elif len(seen_ids) != offset:
        raise ValueError("Checkpoint offset and collected IDs disagree")
    for row in state["rows"]:
        if not set(RAW_COLUMNS).issubset(row) or row["match_id"] not in seen_set:
            raise ValueError("Invalid match in checkpoint")
    return state


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("No valid matches collected; final output was not replaced.")
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="raise").dt.date
    return df.drop_duplicates("match_id", keep="last").sort_values(
        ["match_date", "event_name", "match_id"],
    ).reset_index(drop=True)


def build_matches_dataframe(
    start_date: date,
    end_date: date,
    max_pages: int,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    timeout: int,
    proxy: str | None,
    *,
    transport: str = "browser",
    browser_profile_dir: str = str(PROFILE_DIR),
    cdp_url: str = "",
    manual_wait_seconds: int = 180,
    checkpoint: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    if start_date < CS2_START_DATE:
        raise ValueError(
            f"start_date must be >= {CS2_START_DATE.isoformat()} for CS2-only scope"
        )
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    if max_pages <= 0 or retries <= 0 or timeout <= 0 or manual_wait_seconds < 0:
        raise ValueError("Page/retry/timeout limits must be positive; manual wait must be non-negative")
    if sleep_min < 0 or sleep_max < sleep_min:
        raise ValueError("Require 0 <= sleep_min <= sleep_max")
    if transport not in {"http", "browser"}:
        raise ValueError("Unknown transport")
    if transport == "browser" and proxy:
        raise ValueError("--proxy is supported only with --transport http; browser uses its own network settings")
    if resume and checkpoint is None:
        raise ValueError("Resume requires a checkpoint path")
    state = {
        "version": 1, "start_date": str(start_date), "end_date": str(end_date),
        "next_offset": 0, "rows": [], "seen_ids": [], "total": None,
        "first_page_ids": [], "last_page_ids": [], "complete": False,
    }
    if checkpoint and checkpoint.exists():
        if not resume:
            raise FileExistsError(f"Checkpoint already exists: {checkpoint}. Use --resume or a new output path.")
        state = load_progress(checkpoint, start_date, end_date)
    if state["complete"]:
        print("[resume] Using the already completed snapshot; no network refresh.")
        return rows_to_dataframe(state["rows"])
    # Даже сбой первой страницы оставляет состояние для продолжения сбора.
    if checkpoint:
        save_progress(checkpoint, state)

    with ExitStack() as stack:
        if transport == "browser":
            from src.hltv_browser import BrowserFetcher
            browser = stack.enter_context(BrowserFetcher(browser_profile_dir, cdp_url, manual_wait_seconds))

            def get_page(url: str) -> str:
                time.sleep(random.uniform(sleep_min, sleep_max))
                return browser.get_html(
                    url, ".results-all, div.allres", timeout_ms=timeout * 1000,
                )
        else:
            session = build_session(proxy or None)
            stack.callback(session.close)

            def get_page(url: str) -> str:
                return fetch_html(session, url, sleep_min, sleep_max, retries, timeout)

        def fetch_page(offset: int) -> tuple[str, list[str], int]:
            # Фиксируем диапазон дат, чтобы список не менялся при обходе страниц.
            query = urlencode({"startDate": str(start_date), "endDate": str(end_date), "offset": offset})
            html = get_page(f"{RESULTS_URL}?{query}")
            ids, total = results_page_info(html, offset)
            return html, ids, total

        # Проверяем начало списка и границу продолжения перед использованием смещений.
        if state["next_offset"]:
            checks = {0: state["first_page_ids"], state["next_offset"] - 100: state["last_page_ids"]}
            for offset, expected_ids in checks.items():
                _, ids, total = fetch_page(offset)
                if ids != expected_ids or total != state["total"]:
                    raise RuntimeError(
                        "HLTV listing changed since the checkpoint. Start a fresh snapshot with a new "
                        "output path, preferably ending yesterday. The old checkpoint is preserved."
                    )

        seen_ids = set(state["seen_ids"])
        for _ in range(max_pages):
            offset = state["next_offset"]
            print(f"[page {offset // 100 + 1}] fetching offset={offset}", flush=True)
            html, ids, total = fetch_page(offset)
            if state["total"] is not None and total != state["total"]:
                raise RuntimeError("HLTV result count changed during collection; start a fresh snapshot.")
            if seen_ids.intersection(ids):
                raise RuntimeError("Overlapping results pages; listing shifted. Checkpoint preserved.")
            rows, newest, oldest = extract_rows_from_page(html, start_date, end_date)
            if oldest is None or oldest < start_date or newest > end_date:
                raise RuntimeError("HLTV did not respect the requested dates; refusing incomplete coverage.")
            state["rows"].extend(rows)
            state["seen_ids"].extend(ids)
            seen_ids.update(ids)
            state["total"] = total
            state["next_offset"] = offset + 100
            if offset == 0:
                state["first_page_ids"] = ids
            state["last_page_ids"] = ids
            state["complete"] = len(seen_ids) == total
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            if checkpoint:
                save_progress(checkpoint, state)
            print(
                f"[checkpoint] scanned={len(seen_ids)}/{total}, kept={len(state['rows'])}, "
                f"page_dates={oldest}..{newest}, complete={state['complete']}", flush=True,
            )
            if state["complete"]:
                return rows_to_dataframe(state["rows"])
    raise RuntimeError(
        "Page limit reached; collection is INCOMPLETE. Final CSV was not replaced. "
        "Run the same command with --resume to continue."
    )


def main() -> None:
    args = parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    output_path = Path(args.output)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(".checkpoint.json")
    if checkpoint.resolve() == output_path.resolve():
        raise ValueError("Checkpoint and output paths must differ")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use a new path or explicitly pass --overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_matches_dataframe(
        start_date=start_date,
        end_date=end_date,
        max_pages=args.max_pages,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        retries=args.retries,
        timeout=args.timeout,
        proxy=args.proxy,
        transport=args.transport,
        browser_profile_dir=args.browser_profile_dir,
        cdp_url=args.cdp_url,
        manual_wait_seconds=args.manual_wait_seconds,
        checkpoint=checkpoint,
        resume=args.resume,
    )

    temporary = output_path.with_name(output_path.name + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(output_path)

    print(f"\nSaved {len(df)} matches to: {output_path}")
    print(df.head(10).to_string(index=False))
    print("\nDate range:")
    print(df["match_date"].min(), "->", df["match_date"].max())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Collection interrupted. Last checkpoint and previous final CSV are preserved.")
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Collection stopped: {exc}") from None
