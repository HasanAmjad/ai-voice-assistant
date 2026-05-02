#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "docs" / "report.html"
PDF = ROOT / "docs" / "AI620_VoiceIntent_Report.pdf"

CANDIDATES = {
    "Darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ],
    "Linux": [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "/snap/bin/chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
    ],
    "Windows": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "chrome.exe",
        "msedge.exe",
    ],
}


def find_browser():
    for c in CANDIDATES.get(platform.system(), []):
        if os.path.isfile(c):
            return c
        located = shutil.which(c)
        if located:
            return located
    return None


def main():
    if not HTML.exists():
        sys.exit(f"missing {HTML}")
    browser = find_browser()
    if not browser:
        sys.exit(
            "No Chrome/Chromium/Edge found.\n"
            "macOS: download Chrome from google.com/chrome\n"
            "Ubuntu: sudo apt install chromium-browser\n"
            "Windows: install Google Chrome or use built-in Microsoft Edge"
        )
    print(f"browser: {browser}")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"renderer exited {result.returncode}")
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
