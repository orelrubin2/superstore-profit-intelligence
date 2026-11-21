"""Render reports/figures/project_flow.html to project_flow.png with headless Chrome.

Run from the project root:
    python -m src.render_flow

The page is laid out at 1600x881 CSS pixels and rendered at 2x (3200x1762 PNG). It needs
Google Chrome (or Chromium/Edge) and an internet connection for the Google Fonts (Inter,
JetBrains Mono). Set the CHROME environment variable if the browser is not found.
No Python packages beyond the standard library are needed.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "reports" / "figures" / "project_flow.html"
PNG = ROOT / "reports" / "figures" / "project_flow.png"
WIDTH, HEIGHT, SCALE = 1600, 881, 2

CANDIDATES = [
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "",
    shutil.which("chrome") or "",
]


def find_chrome() -> str:
    """Path of the first Chrome-like browser found."""
    for path in CANDIDATES:
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("Chrome not found; set the CHROME environment variable.")


def render(html: Path = HTML, png: Path = PNG) -> Path:
    """Screenshot `html` into `png` at WIDTH x HEIGHT CSS pixels, SCALE x resolution."""
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([
            find_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}",
            f"--force-device-scale-factor={SCALE}",
            "--virtual-time-budget=10000",          # let the web fonts load first
            f"--screenshot={png}", html.as_uri(),
        ], check=True, capture_output=True, timeout=120)
    return png


if __name__ == "__main__":
    print(f"Wrote {render().relative_to(ROOT)}")
