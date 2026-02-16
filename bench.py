#!/usr/bin/env python3
"""
Benchmark get_all_scores() for the ExpressionSignifierEngine.

Usage: python bench.py [N]
  N = number of get_all_scores() calls (default 1000).

Run from project root. Records total time and per-call time so you can
compare before/after buffer or signifier changes (see README.md §16 Performance).
"""
import os
import sys
import time

# Project root on path (script lives at project root)
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from helpers import ExpressionSignifierEngine, get_weights
from test_core import make_neutral_landmarks, warmup_engine


def main():
    n = 1000
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    shape = (480, 640, 3)
    engine = ExpressionSignifierEngine(buffer_frames=22, weights_provider=get_weights)
    lm = make_neutral_landmarks(shape)
    warmup_engine(engine, lm, shape, 15)
    # Warmup run
    engine.get_all_scores()
    start = time.perf_counter()
    for _ in range(n):
        engine.get_all_scores()
    elapsed = time.perf_counter() - start
    per_call_ms = (elapsed / n) * 1000
    print(f"get_all_scores() x{n}: {elapsed:.3f}s total, {per_call_ms:.3f} ms/call")


if __name__ == "__main__":
    main()
