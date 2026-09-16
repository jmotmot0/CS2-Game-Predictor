"""Visible browser shared by both HLTV collectors; challenges stay manual."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


class BrowserActionRequired(RuntimeError):
    """Stop a batch rather than silently skipping pages behind a challenge."""


class BrowserFetcher:
    def __init__(
        self, profile_dir: str, cdp_url: str = "", manual_wait_seconds: int = 180,
    ) -> None:
        if manual_wait_seconds < 0:
            raise ValueError("manual_wait_seconds must be non-negative")
        self.profile_dir = profile_dir
        self.cdp_url = cdp_url.strip()
        self.manual_wait_seconds = manual_wait_seconds
        self.playwright = self.browser = self.context = self.page = None
        self.owns_context = False

    def __enter__(self) -> "BrowserFetcher":
        self.playwright = sync_playwright().start()
        try:
            if self.cdp_url:
                self.browser = self.playwright.chromium.connect_over_cdp(self.cdp_url)
                if not self.browser.contexts:
                    raise RuntimeError("No browser context is available at the CDP URL")
                self.context = self.browser.contexts[0]
                # Never navigate or close a user's existing tab.
                self.page = self.context.new_page()
            else:
                Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
                self.owns_context = True
                options = dict(
                    user_data_dir=self.profile_dir, headless=False,
                    viewport={"width": 1440, "height": 1000},
                )
                try:
                    self.context = self.playwright.chromium.launch_persistent_context(
                        channel="chrome", **options,
                    )
                except Exception as chrome_error:
                    print("[browser] Chrome could not start; trying installed Chromium.")
                    try:
                        self.context = self.playwright.chromium.launch_persistent_context(**options)
                    except Exception as chromium_error:
                        raise RuntimeError(
                            "Could not start Chrome or Chromium. Close other collectors using "
                            "this profile, or install Chromium: python -m playwright install chromium. "
                            f"Chrome: {chrome_error}; Chromium: {chromium_error}"
                        ) from chromium_error
                self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.owns_context and self.context is not None:
                self.context.close()
            elif self.page is not None:
                self.page.close()
        finally:
            if self.playwright is not None:
                self.playwright.stop()

    def get_html(
        self, url: str, ready_selector: str, referer: str | None = None,
        timeout_ms: int = 45000,
    ) -> str:
        assert self.page is not None
        try:
            self.page.goto(url, wait_until="domcontentloaded", referer=referer, timeout=timeout_ms)
        except PlaywrightTimeoutError:
            # A navigation timeout may still leave a usable page loaded.
            pass
        try:
            self.page.wait_for_selector(ready_selector, timeout=12000)
        except PlaywrightTimeoutError as exc:
            if self.manual_wait_seconds:
                print(
                    f"[manual action] Expected page content is missing: {url}\n"
                    "Check the collector's browser window. If a security check appears, "
                    "complete it yourself. Do not press Enter in the terminal.\n"
                    f"Waiting up to {self.manual_wait_seconds}s; collection resumes automatically.",
                    flush=True,
                )
                try:
                    self.page.wait_for_selector(
                        ready_selector, timeout=self.manual_wait_seconds * 1000,
                    )
                except PlaywrightTimeoutError as wait_error:
                    raise BrowserActionRequired(
                        f"Page not ready: {url}. A site challenge or changed markup may be responsible. "
                        "Collection stopped; rerun with --resume after resolving access."
                    ) from wait_error
            else:
                raise BrowserActionRequired(f"Page not ready: {url}; manual waiting is disabled.") from exc
        return self.page.content()
