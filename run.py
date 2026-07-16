#!/usr/bin/env python3
"""
Single entry point for Cog, used identically locally and on Kaggle.

Usage:
    python run.py bench            # run the deterministic benchmark suite
    python run.py broadval         # run the broad ACTIVE-validation campaign
    python run.py broadval --n 100 --seeds 1 2 3
    python run.py rollout          # controlled activation + regression check
    python run.py test             # run the test suite (requires pytest)

Both Kaggle and local development use this exact same entry point, so the
benchmark is reproducible everywhere.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _bench() -> int:
    # Deterministic benchmark suite via the package entry point.
    return subprocess.call([sys.executable, "-m", "cog.bench"], cwd=ROOT)


def _broadval(args: list[str]) -> int:
    script = ROOT / "experiments" / "exp_broad_validation.py"
    return subprocess.call([sys.executable, str(script), *args], cwd=ROOT)


def _rollout() -> int:
    script = ROOT / "experiments" / "exp_rollout.py"
    return subprocess.call([sys.executable, str(script)], cwd=ROOT)


def _test() -> int:
    # Discover tests across the repo (package subdirs + top-level tests/).
    return subprocess.call(
        [sys.executable, "-m", "pytest", "cog", "tests", "-q"], cwd=ROOT
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="run.py", description="Cog single entry point")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bench", help="run the deterministic benchmark suite")
    sub.add_parser("test", help="run the test suite (needs pytest)")
    sub.add_parser("rollout", help="controlled activation + regression check")

    p_bv = sub.add_parser("broadval", help="run the broad ACTIVE-validation campaign")
    p_bv.add_argument("--n", type=int, default=300)
    p_bv.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p_bv.add_argument("--soak", type=int, default=0)

    ns = ap.parse_args()
    if ns.cmd == "bench":
        return _bench()
    if ns.cmd == "test":
        return _test()
    if ns.cmd == "rollout":
        return _rollout()
    if ns.cmd == "broadval":
        rest = []
        rest += ["--n", str(ns.n)]
        rest += ["--seeds", *map(str, ns.seeds)]
        if ns.soak:
            rest += ["--soak", str(ns.soak)]
        return _broadval(rest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
