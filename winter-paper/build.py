#!/usr/bin/env python3
"""Compatibility entry point for the coupled, isolated manuscript build."""
import argparse
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", metavar="TARGET")
    parser.add_argument("--build-dir", default="build-positioning")
    args = parser.parse_args()
    if set(args.targets) - {"main", "supplementary"}:
        parser.error("Targets must be main or supplementary")
    if args.targets:
        print("Both papers and highlights are built together to resolve cross-references.")
    command = [sys.executable, str(Path(__file__).resolve().parent / "audit/build_paper.py"),
               "--build-dir", args.build_dir]
    raise SystemExit(subprocess.run(command, check=False).returncode)
