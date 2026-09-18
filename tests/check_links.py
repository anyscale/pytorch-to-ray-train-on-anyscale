"""Every http(s) URL in notebooks, README, and docs must answer. Exit 1 on failures.

    python tests/check_links.py
"""
from __future__ import annotations

import concurrent.futures
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s<>\"')\]}|`]+")
HEADERS = {"User-Agent": "Mozilla/5.0 (link checker for pytorch-to-ray-train-on-anyscale)"}
# Sites that need a login return a redirect or 4xx to anonymous clients but are still valid links.
ALLOW_ANY_STATUS = ("console.anyscale.com", "anyscale.coursifai.com")


def collect_urls() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in list(ROOT.glob("notebooks/*.ipynb")) + list(ROOT.glob("*.md")) + list(ROOT.glob("docs/*.md")):
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text())
            text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        else:
            text = path.read_text()
        for url in URL_RE.findall(text):
            url = url.rstrip(".,;:")
            if "<" in url or "your-" in url or "example.com" in url or "localhost" in url:
                continue
            found.setdefault(url, set()).add(str(path.relative_to(ROOT)))
    return found


def check(url: str) -> tuple[str, int | str]:
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                return url, resp.status
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code in (404, 410):
                return url, e.code
        except Exception as e:  # noqa: BLE001
            if method == "GET":
                return url, f"{type(e).__name__}: {e}"
    return url, "unreachable"


def main() -> int:
    urls = collect_urls()
    print(f"checking {len(urls)} unique URLs")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for url, status in pool.map(check, sorted(urls)):
            ok = isinstance(status, int) and status < 400
            if not ok and any(host in url for host in ALLOW_ANY_STATUS) and isinstance(status, int) and status < 500:
                ok = True
            mark = "ok " if ok else "BAD"
            print(f"{mark} {status!s:>6} {url}")
            if not ok:
                failures.append((url, status, sorted(urls[url])))
    if failures:
        print(f"\n{len(failures)} failing URL(s):")
        for url, status, where in failures:
            print(f"  {url} -> {status} (in {', '.join(where)})")
        return 1
    print("all links OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
