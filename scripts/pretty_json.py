#!/usr/bin/env python3
"""
Simple JSON pretty-printer for large files.

Usage:
  python pretty_json.py <file-or-dir> [...]
Options:
  --indent N       Indentation spaces (default 2)
  --no-backup      Don't keep a .bak backup

It makes a .bak copy the first time it touches a file (unless --no-backup).
"""
import argparse
import json
import os
import shutil
import sys


def pretty_file(path, indent=2, backup=True):
    if backup:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
    # read and parse
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # write to a tmp then atomically replace
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    os.replace(tmp, path)


def find_json_files(root):
    for dirpath, dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Pretty-print JSON files (in-place)")
    parser.add_argument("paths", nargs="*", help="file or directory paths to process")
    parser.add_argument(
        "--indent", type=int, default=2, help="number of spaces for indent"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="don't create .bak backup"
    )
    args = parser.parse_args(argv)

    if not args.paths:
        parser.print_help()
        return 1

    targets = []
    for p in args.paths:
        if os.path.isdir(p):
            targets.extend(find_json_files(p))
        elif os.path.isfile(p):
            targets.append(p)
        else:
            print(f"Skipped (not found): {p}")

    if not targets:
        print("No JSON files found to process.")
        return 0

    for t in targets:
        try:
            pretty_file(t, indent=args.indent, backup=not args.no_backup)
            print(f"Formatted: {t}")
        except Exception as e:
            print(f"Failed: {t} -> {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
