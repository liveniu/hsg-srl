#!/usr/bin/env python3
"""Fail if published artifact files contain CJK characters."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bootstrap import ARTIFACT_ROOT

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")
SKIP_DIRS = {".git", ".idea", ".vscode", ".venv", "__pycache__", "work", "external", "runs"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".pt", ".db", ".pyc", ".npz"}


def main() -> int:
    hits = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ARTIFACT_ROOT).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if CJK_RE.search(line):
                hits.append(f"{path.relative_to(ARTIFACT_ROOT).as_posix()}:{number}: {line.strip()[:160]}")
    if hits:
        print(f"CJK characters found in {len(hits)} line(s):", file=sys.stderr)
        for row in hits[:80]:
            print(row, file=sys.stderr)
        return 1
    print("English check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
