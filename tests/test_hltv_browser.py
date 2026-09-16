from unittest.mock import Mock

import pytest

from src.hltv_browser import BrowserActionRequired, BrowserFetcher, PlaywrightTimeoutError
from src import hltv_browser


def test_manual_wait_resumes_without_terminal_input():
    browser = BrowserFetcher("unused", manual_wait_seconds=60)
    browser.page = Mock()
    browser.page.wait_for_selector.side_effect = [PlaywrightTimeoutError("challenge"), None]
    browser.page.content.return_value = "ready"
    assert browser.get_html("https://www.hltv.org/results", ".results-all") == "ready"
    assert browser.page.wait_for_selector.call_count == 2
    assert browser.page.wait_for_selector.call_args.kwargs["timeout"] == 60000


def test_unresolved_challenge_stops_batch():
    browser = BrowserFetcher("unused", manual_wait_seconds=1)
    browser.page = Mock()
    browser.page.wait_for_selector.side_effect = PlaywrightTimeoutError("challenge")
    with pytest.raises(BrowserActionRequired, match="Collection stopped"):
        browser.get_html("https://www.hltv.org/results", ".results-all")
    browser.page.content.assert_not_called()


def test_zero_wait_is_noninteractive():
    browser = BrowserFetcher("unused", manual_wait_seconds=0)
    browser.page = Mock()
    browser.page.wait_for_selector.side_effect = PlaywrightTimeoutError("challenge")
    with pytest.raises(BrowserActionRequired, match="waiting is disabled"):
        browser.get_html("https://www.hltv.org/results", ".results-all")
    assert browser.page.wait_for_selector.call_count == 1


def test_navigation_timeout_can_leave_ready_content():
    browser = BrowserFetcher("unused")
    browser.page = Mock()
    browser.page.goto.side_effect = PlaywrightTimeoutError("timeout")
    browser.page.content.return_value = "ready"
    assert browser.get_html("https://www.hltv.org/results", ".results-all") == "ready"


def test_cdp_cleanup_does_not_close_user_browser():
    browser = BrowserFetcher("unused")
    browser.page = Mock()
    browser.context = Mock()
    browser.playwright = Mock()
    browser.__exit__(None, None, None)
    browser.page.close.assert_called_once()
    browser.context.close.assert_not_called()
    browser.playwright.stop.assert_called_once()


def test_cdp_connection_uses_own_tab_without_launching_browser(monkeypatch, tmp_path):
    runtime = Mock()
    context = Mock()
    personal_tab = Mock()
    context.pages = [personal_tab]
    runtime.chromium.connect_over_cdp.return_value.contexts = [context]
    monkeypatch.setattr(hltv_browser, "sync_playwright", lambda: Mock(start=lambda: runtime))
    profile = tmp_path / "should-not-be-created"
    with BrowserFetcher(str(profile), "http://127.0.0.1:9222") as fetcher:
        assert fetcher.page is context.new_page.return_value
        assert not fetcher.owns_context
    runtime.chromium.connect_over_cdp.assert_called_once_with("http://127.0.0.1:9222")
    runtime.chromium.launch_persistent_context.assert_not_called()
    personal_tab.goto.assert_not_called()
    personal_tab.close.assert_not_called()
    context.close.assert_not_called()
    context.new_page.return_value.close.assert_called_once()
    assert not profile.exists()


def test_cdp_connection_failure_stops_driver(monkeypatch):
    runtime = Mock()
    runtime.chromium.connect_over_cdp.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(hltv_browser, "sync_playwright", lambda: Mock(start=lambda: runtime))
    with pytest.raises(RuntimeError, match="connection refused"):
        with BrowserFetcher("unused", "http://127.0.0.1:9222"):
            pass
    runtime.stop.assert_called_once()
