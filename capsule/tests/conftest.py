from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from capsule.pack import parse_capsule


ROOT = Path(__file__).resolve().parents[1]

# MR v2 builds would otherwise run SynthStrip (minutes on CPU); tests that need the brain mask
# ask for it explicitly. Subprocess builds inherit this.
os.environ.setdefault("CAPSULE_BRAIN_MASK", "none")
os.environ.setdefault("CAPSULE_ANATOMY", "none")


def _cached_chromium_executable() -> Path | None:
    cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    candidates: list[Path] = []
    candidates.extend(cache.glob("chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"))
    candidates.extend(cache.glob("chromium_headless_shell-*/chrome-mac/headless_shell"))
    candidates.extend(cache.glob("chromium_headless_shell-*/chrome-linux/headless_shell"))
    candidates.extend(cache.glob("chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell"))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        return None
    # Prefer the newest cached build without asking Playwright to download one.
    return sorted(existing, key=lambda p: tuple(int(x) for x in re.findall(r"\d+", str(p))), reverse=True)[0]


@pytest.fixture(scope="session")
def capsule_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("case-capsule") / "fixture.capsule.html"
    result = subprocess.run(
        [sys.executable, str(ROOT / "viewer" / "dev_fixture.py"), "-o", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"fixture CLI failed ({result.returncode}): {result.stderr}")
    assert output.is_file()
    return output


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def chromium():
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    executable = _cached_chromium_executable()
    if executable is None:
        playwright.stop()
        pytest.fail("No cached Chromium executable found; browser tests require the preinstalled browser")
    try:
        browser = playwright.chromium.launch(
            executable_path=str(executable),
            headless=True,
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-unsafe-swiftshader",
            ],
        )
    except Exception as exc:
        playwright.stop()
        pytest.fail(f"Cached Chromium cannot launch in this environment: {exc}")
    yield browser
    browser.close()
    playwright.stop()


@pytest.fixture
def viewer_session(chromium, capsule_file: Path, tmp_path: Path):
    context = chromium.new_context(
        accept_downloads=True,
        viewport={"width": 1440, "height": 1000},
        device_scale_factor=1,
    )
    page = context.new_page()
    allowed_file_urls = {capsule_file.as_uri()}
    evidence: dict[str, object] = {
        "requests": [],
        "console_errors": [],
        "page_errors": [],
        "allowed_file_urls": allowed_file_urls,
    }

    def record_request(request) -> None:
        scheme = urlsplit(request.url).scheme
        if scheme in {"data", "blob"}:
            return
        if scheme != "file" or request.url not in allowed_file_urls:
            evidence["requests"].append(request.url)

    page.on("request", record_request)
    page.on("console", lambda message: evidence["console_errors"].append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: evidence["page_errors"].append(str(error)))
    page.goto(capsule_file.as_uri(), wait_until="load", timeout=60000)
    assert page.evaluate("async () => { await window.__capsule.ready; return window.__capsule.state.status; }") == "ready"
    yield page, context, evidence
    context.close()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "browser: requires the preinstalled real Chromium browser")
