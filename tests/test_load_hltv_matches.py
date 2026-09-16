from __future__ import annotations

import json
import sys
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from src import hltv_browser, load_hltv_matches as loader


DAY = date(2026, 9, 27)


def results_html(offset=0, total=101, *, first_id=None):
    last = min(offset + 100, total)
    entries = []
    for number in range(offset + 1, last + 1):
        match_id = first_id if first_id is not None and number == offset + 1 else number
        entries.append(f"""
            <div class="result-con"><a class="a-reset" href="/matches/{match_id}/alpha-bravo">
              <div class="team1"><div class="team">Alpha</div></div>
              <div class="team2"><div class="team">Bravo</div></div>
              <div class="event-name">Test cup</div>
              <div class="result-score"><span>2</span><span>1</span></div>
            </a></div>""")
    return f"""<html><span>{offset + 1} - {last} of {total}</span>
        <div class="results-all"><div class="results-sublist">
          <div class="standard-headline">Results for September 27th 2026</div>
          {''.join(entries)}
        </div></div></html>"""


def install_browser(monkeypatch, pages):
    instances = []

    class FakeBrowser:
        def __init__(self, *args):
            self.urls = []
            self.closed = False
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def get_html(self, url, *args, **kwargs):
            self.urls.append(url)
            query = parse_qs(urlsplit(url).query)
            assert query["startDate"] == [str(DAY)]
            assert query["endDate"] == [str(DAY)]
            response = pages[int(query["offset"][0])]
            if isinstance(response, BaseException):
                raise response
            return response

    monkeypatch.setattr(hltv_browser, "BrowserFetcher", FakeBrowser)
    return instances


def collect(tmp_path, **kwargs):
    return loader.build_matches_dataframe(
        DAY, DAY, sleep_min=0, sleep_max=0, retries=1, timeout=1, proxy=None,
        checkpoint=tmp_path / "progress.json", **{"max_pages": 10, **kwargs},
    )


def test_parse_results_keeps_regular_list_and_filters_invalid_matches():
    html = results_html(total=2).replace("<span>2</span>", "<span>1</span>", 1)
    # A featured duplicate must not affect counts or output.
    html = '<div class="results-all">featured duplicate</div>' + html
    rows, newest, oldest = loader.extract_rows_from_page(html, DAY, DAY)
    assert [row["match_id"] for row in rows] == ["2"]
    assert rows[0]["winner"] == "Alpha"
    assert newest == oldest == DAY
    assert loader.results_page_info(html, 0) == (["1", "2"], 2)


@pytest.mark.parametrize("change", [
    lambda html: html.replace("1 - 2 of 2", "unknown"),
    lambda html: html.replace("1 - 2 of 2", "1 - 2 of 3"),
    lambda html: html.replace("/matches/2/", "/matches/1/"),
])
def test_bad_pagination_is_not_completion(change):
    with pytest.raises(RuntimeError):
        loader.results_page_info(change(results_html(total=2)), 0)


def test_changed_markup_does_not_silently_drop_a_match():
    with pytest.raises(RuntimeError, match="Missing teams or scores"):
        loader.extract_rows_from_page(results_html(total=1).replace('class="team1"', 'class="new"'), DAY, DAY)


def test_browser_full_run_commits_complete_checkpoint(monkeypatch, tmp_path):
    instances = install_browser(monkeypatch, {0: results_html(), 100: results_html(100)})
    frame = collect(tmp_path)
    assert len(frame) == 101 and frame["match_id"].is_unique
    assert list(frame.columns) == loader.RAW_COLUMNS
    state = loader.load_progress(tmp_path / "progress.json", DAY, DAY)
    assert state["complete"] is True
    assert len(state["seen_ids"]) == state["total"] == 101
    assert instances[0].closed
    assert not list(tmp_path.glob("*.tmp"))


def test_page_limit_is_incomplete_and_resume_rechecks_boundary(monkeypatch, tmp_path):
    instances = install_browser(monkeypatch, {0: results_html(), 100: results_html(100)})
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        collect(tmp_path, max_pages=1)
    state = loader.load_progress(tmp_path / "progress.json", DAY, DAY)
    assert not state["complete"] and state["next_offset"] == 100
    assert len(state["rows"]) == 100
    assert instances[0].closed
    frame = collect(tmp_path, resume=True)
    assert len(frame) == 101
    assert [parse_qs(urlsplit(url).query)["offset"][0] for url in instances[1].urls] == ["0", "100"]


def test_failure_preserves_last_committed_page(monkeypatch, tmp_path):
    install_browser(monkeypatch, {0: results_html(), 100: hltv_browser.BrowserActionRequired("challenge")})
    with pytest.raises(hltv_browser.BrowserActionRequired):
        collect(tmp_path)
    state = loader.load_progress(tmp_path / "progress.json", DAY, DAY)
    assert state["next_offset"] == 100 and len(state["rows"]) == 100
    assert not state["complete"]


def test_changed_listing_aborts_resume_without_modifying_saved_rows(monkeypatch, tmp_path):
    pages = {0: results_html(), 100: results_html(100)}
    install_browser(monkeypatch, pages)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        collect(tmp_path, max_pages=1)
    saved = (tmp_path / "progress.json").read_bytes()
    pages[0] = results_html(first_id=999)
    with pytest.raises(RuntimeError, match="listing changed"):
        collect(tmp_path, resume=True)
    assert (tmp_path / "progress.json").read_bytes() == saved


def test_resume_rejects_other_dates(tmp_path):
    path = tmp_path / "progress.json"
    loader.save_progress(path, {"version": 1, "start_date": str(DAY), "end_date": str(DAY)})
    with pytest.raises(ValueError, match="dates differ"):
        loader.load_progress(path, date(2026, 9, 26), DAY)


def test_completed_snapshot_can_be_exported_without_browser(monkeypatch, tmp_path):
    instances = install_browser(monkeypatch, {0: results_html(total=1)})
    collect(tmp_path)
    assert len(collect(tmp_path, resume=True)) == 1
    assert len(instances) == 1
    with pytest.raises(FileExistsError, match="Checkpoint already exists"):
        collect(tmp_path)


def test_http_transport_is_still_available(monkeypatch, tmp_path):
    class Session:
        closed = False

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr(loader, "build_session", lambda proxy: session)
    monkeypatch.setattr(loader, "fetch_html", lambda *args: results_html(total=1))
    assert len(collect(tmp_path, transport="http")) == 1
    assert session.closed


def test_atomic_checkpoint_failure_preserves_previous_state(monkeypatch, tmp_path):
    path = tmp_path / "progress.json"
    loader.save_progress(path, {"value": "old"})
    saved = path.read_bytes()

    def fail_replace(*args):
        raise PermissionError("file locked")

    monkeypatch.setattr(loader.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        loader.save_progress(path, {"value": "new"})
    assert path.read_bytes() == saved


@pytest.mark.parametrize("next_page", [
    results_html(100, total=102),
    results_html(100, first_id=1),
])
def test_listing_changes_mid_run_do_not_commit_bad_page(monkeypatch, tmp_path, next_page):
    install_browser(monkeypatch, {0: results_html(), 100: next_page})
    with pytest.raises(RuntimeError, match="changed|Overlapping"):
        collect(tmp_path)
    state = loader.load_progress(tmp_path / "progress.json", DAY, DAY)
    assert state["next_offset"] == 100 and len(state["rows"]) == 100
    assert not state["complete"]


def test_failed_main_never_replaces_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "matches.csv"
    output.write_text("old,data\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "load", "--output", str(output), "--overwrite", "--max-pages", "1",
        "--start-date", str(DAY), "--end-date", str(DAY), "--sleep-min", "0", "--sleep-max", "0",
    ])
    install_browser(monkeypatch, {0: results_html()})
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        loader.main()
    assert output.read_text(encoding="utf-8") == "old,data\n"
    state = json.loads(output.with_suffix(".checkpoint.json").read_text(encoding="utf-8"))
    assert not state["complete"]


def test_main_publishes_completed_csv(monkeypatch, tmp_path):
    output = tmp_path / "matches.csv"
    monkeypatch.setattr(sys, "argv", [
        "load", "--output", str(output), "--start-date", str(DAY), "--end-date", str(DAY),
        "--sleep-min", "0", "--sleep-max", "0",
    ])
    install_browser(monkeypatch, {0: results_html(total=1)})
    loader.main()
    assert output.read_text(encoding="utf-8").startswith(",".join(loader.RAW_COLUMNS))
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("arguments", [
    {"max_pages": 0}, {"transport": "invalid"}, {"manual_wait_seconds": -1},
])
def test_invalid_options_fail_before_browser(monkeypatch, tmp_path, arguments):
    instances = install_browser(monkeypatch, {})
    with pytest.raises(ValueError):
        collect(tmp_path, **arguments)
    assert instances == []
