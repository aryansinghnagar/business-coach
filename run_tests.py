#!/usr/bin/env python3
"""
CLI test runner for Business Meeting Copilot.

For a GUI with detailed failure analysis, use the control panel (Tests tab):
  python control_panel.py

Usage: python run_tests.py [--verbose] [pattern]
  --verbose  Show detailed output for each test
  pattern    Optional: run only tests matching this string (e.g. "api", "chat")
"""

import sys
import os
import argparse

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_tests(verbose=False, pattern=None):
    """Discover and run tests. Returns (total, failures, errors)."""
    import unittest

    loader = unittest.TestLoader()
    start_dir = os.path.join(PROJECT_ROOT, "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    if pattern:
        def gather_tests(s, acc):
            for t in s:
                if isinstance(t, unittest.TestSuite):
                    gather_tests(t, acc)
                else:
                    acc.append(t)
        all_tests = []
        gather_tests(suite, all_tests)
        filtered = unittest.TestSuite()
        pat = pattern.lower()
        for t in all_tests:
            if pat in str(t).lower():
                filtered.addTest(t)
        if list(filtered):
            suite = filtered

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    return result.testsRun, len(result.failures), len(result.errors)


def main():
    parser = argparse.ArgumentParser(
        description="Run Business Meeting Copilot test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py
  python run_tests.py --verbose
  python run_tests.py api
  python run_tests.py chat
  python run_tests.py context
        """,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Run only tests matching this string",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Business Meeting Copilot — Test Suite")
    print("=" * 60)
    if args.pattern:
        print(f"Filter: tests matching '{args.pattern}'")
    print()

    total, failures, errors = run_tests(verbose=args.verbose, pattern=args.pattern)

    print()
    print("=" * 60)
    if failures == 0 and errors == 0:
        print(f"OK — {total} test(s) passed")
        return 0
    else:
        print(f"FAILED — {failures} failure(s), {errors} error(s) out of {total} test(s)")
        print()
        print("Troubleshooting:")
        print("  • Failures often indicate logic/assertion issues in the code under test")
        print("  • Errors may indicate import problems, missing deps, or config issues")
        print("  • Run with --verbose to see full tracebacks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
