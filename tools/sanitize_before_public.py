"""Sanitize host-specific paths/hosts before opening the repository to the public.

Dry-run by default; pass --apply to rewrite files. Only tracked text files are
considered, and only the patterns listed below are replaced.

Usage:
    python tools/sanitize_before_public.py            # report occurrences
    python tools/sanitize_before_public.py --apply    # rewrite in place
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {".md", ".txt", ".py", ".sh", ".json", ".yml", ".yaml", ".tex", ".cfg", ".toml", ".bib"}
SKIP_DIRS = {".git", "__pycache__"}
SKIP_FILES = {"tools/sanitize_before_public.py"}

# ordered replacements: (compiled regex, replacement)
RULES = [
    (r"/home/ubuntu/[A-Za-z0-9_./-]*", "<PROJECT_ROOT>"),
    (r"/home/ubuntu", "<HOME>"),
    (r"\bubuntu-server-tls\b", "<SERVER_ALIAS_TLS>"),
    (r"\bubuntu-server\b", "<SERVER_ALIAS>"),
    (r"\b127\.0\.0\.1:\d{2,5}\b", "<LOCALHOST_PORT>"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<HOST_IP>"),
    (r"\b22023\b", "<PORT>"),
]


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print("warning: not a git repository; scanning the working tree instead", file=sys.stderr)
        files = [p for p in ROOT.rglob("*") if p.is_file()]
    else:
        files = [ROOT / rel for rel in out.stdout.splitlines()]
    result = []
    for p in files:
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.relative_to(ROOT).as_posix() in SKIP_FILES:
            continue
        if p.stat().st_size > 2_000_000:
            continue
        result.append(p)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="rewrite files in place")
    args = ap.parse_args()
    import re

    total = 0
    changed_files = 0
    for path in tracked_files():
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            print(f"skip (non-UTF8): {path.relative_to(ROOT)}", file=sys.stderr)
            continue
        new = text
        hits = 0
        for pattern, repl in RULES:
            new2, n = re.subn(pattern, repl, new)
            new = new2
            hits += n
        if hits:
            total += hits
            changed_files += 1
            print(f"{path.relative_to(ROOT)}: {hits} replacement(s)")
            if args.apply:
                path.write_text(new, encoding="utf-8")
    print(f"\n{'applied' if args.apply else 'dry-run'}: {total} replacement(s) in {changed_files} file(s)")
    if not args.apply:
        print("re-run with --apply to rewrite; review the diff before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
