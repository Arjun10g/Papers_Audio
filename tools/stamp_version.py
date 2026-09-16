#!/usr/bin/env python3
"""Stamp the deployed app with a version string (CI passes the short commit SHA).

    python tools/stamp_version.py 1a2b3c4

Rewrites the APP_VERSION value in app/config.js and app/sw.js. The service
worker's cache names derive from it, so every deploy invalidates the old app
shell on phones and the "Update ready" toast appears.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"""(APP_VERSION\s*[:=]\s*["'])[^"']*(["'])""")


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        sys.exit("usage: stamp_version.py <version>")
    version = sys.argv[1].strip()
    for name in ("app/config.js", "app/sw.js"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        new, n = PATTERN.subn(lambda m: f"{m.group(1)}{version}{m.group(2)}", text)
        if n == 0:
            sys.exit(f"{name}: no APP_VERSION assignment found")
        path.write_text(new, encoding="utf-8")
        print(f"{name}: APP_VERSION = {version}")


if __name__ == "__main__":
    main()
