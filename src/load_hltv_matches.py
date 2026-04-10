from __future__ import annotations

import argparse
import os
import random
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "matches_raw.csv"

CS2_START_DATE = date(2023, 9, 27)
RESULTS_URL = "https://www.hltv.org/results"

BAD_TEAM_NAMES = {"TBD", "Unknown", "", "-", "—"}

MATCH_ID_RE = re.compile(r"/matches/(\d+)")
ORDINAL_DAY_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)")


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
        help="Maximum number of HLTV results pages to scan. Default: 500",
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
        help="Max retries per page request. Default: 5",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.getenv("HLTV_PROXY", ""),
        help="Optional proxy URL, e.g. http://user:pass@host:port. "
             "By default uses HLTV_PROXY env var if set.",
    )
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

                # Sometimes Cloudflare challenge can still come back as HTML.
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
    # Current HLTV results page: first page may contain big-results duplicates.
    # The last .results-all block is the regular authoritative list.
    containers = soup.select(".results-all")
    if containers:
        return containers[-1]

    # Fallback for older markup variants.
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
            # Fallback for older markup where parser iterated anchors.
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

            if len(score_spans) < 2:
                continue

            score1 = safe_int(score_spans[0].get_text(strip=True))
            score2 = safe_int(score_spans[1].get_text(strip=True))

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


def build_matches_dataframe(
    start_date: date,
    end_date: date,
    max_pages: int,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    timeout: int,
    proxy: str | None,
) -> pd.DataFrame:
    if start_date < CS2_START_DATE:
        raise ValueError(
            f"start_date must be >= {CS2_START_DATE.isoformat()} for CS2-only scope"
        )
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    session = build_session(proxy or None)
    all_rows: list[dict[str, Any]] = []

    try:
        started_at = time.time()
        target_total_days = max((end_date - start_date).days, 1)

        for page_idx in range(max_pages):
            offset = page_idx * 100
            url = f"{RESULTS_URL}?offset={offset}"

            print(f"[page {page_idx + 1}] fetching offset={offset}")
            html = fetch_html(
                session=session,
                url=url,
                sleep_min=sleep_min,
                sleep_max=sleep_max,
                retries=retries,
                timeout=timeout,
            )

            page_rows, page_newest, page_oldest = extract_rows_from_page(
                html=html,
                start_date=start_date,
                end_date=end_date,
            )

            all_rows.extend(page_rows)

            elapsed_sec = time.time() - started_at
            elapsed_min = elapsed_sec / 60.0
            pages_done = page_idx + 1
            pages_per_min = pages_done / elapsed_min if elapsed_min > 0 else 0.0

            progress_pct = None
            eta_min = None

            if page_oldest is not None:
                covered_days = (end_date - page_oldest).days
                covered_days = max(0, min(covered_days, target_total_days))
                progress_pct = 100.0 * covered_days / target_total_days

                if progress_pct > 0:
                    total_estimated_min = elapsed_min / (progress_pct / 100.0)
                    eta_min = max(0.0, total_estimated_min - elapsed_min)

            progress_text = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
            eta_text = f"{eta_min:.1f}m" if eta_min is not None else "n/a"

            print(
                f"[page {pages_done}] rows_kept={len(page_rows)} "
                f"page_newest={page_newest} page_oldest={page_oldest} "
                f"progress={progress_text} elapsed={elapsed_min:.1f}m "
                f"eta={eta_text} speed={pages_per_min:.1f} pages/min"
            )

            if page_oldest is None:
                print("No dated results found on page. Stopping.")
                break

            if page_oldest < start_date:
                print(
                    f"Reached results older than start_date ({start_date}). Stopping."
                )
                break

    finally:
        session.close()

    if not all_rows:
        raise RuntimeError("No valid matches collected.")

    df = pd.DataFrame(all_rows)

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()
    df["match_date"] = df["match_date"].dt.date

    df = df.drop_duplicates(subset=["match_id"]).copy()
    df = df.sort_values(["match_date", "event_name", "match_id"]).reset_index(drop=True)

    column_order = [
        "match_id",
        "match_date",
        "event_id",
        "event_name",
        "team1",
        "team2",
        "team1_score",
        "team2_score",
        "winner",
        "game",
        "completed",
        "is_professional_proxy",
        "professional_filter_note",
        "source",
        "source_url",
    ]
    df = df[column_order]

    return df


def main() -> None:
    args = parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    output_path = Path(args.output)

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
    )

    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} matches to: {output_path}")
    print(df.head(10).to_string(index=False))
    print("\nDate range:")
    print(df["match_date"].min(), "->", df["match_date"].max())


if __name__ == "__main__":
    main()