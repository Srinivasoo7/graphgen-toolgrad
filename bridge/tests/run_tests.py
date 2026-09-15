"""Minimal keyless test runner (no pytest needed on this box).

Usage (from the repo root)::

    python3 bridge/tests/run_tests.py

Runs every ``test_*`` function in ``test_*.py`` modules in this directory.
Exit code is 0 iff all tests pass. Modules whose imports are unavailable
(missing optional deps) are reported as SKIP, never as crashes.
"""

import importlib
import os
import pkgutil
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import bridge.tests as tests_pkg

    passed = 0
    skipped = 0
    failures = 0
    for _, mod_name, _ in sorted(pkgutil.iter_modules(tests_pkg.__path__)):
        if not mod_name.startswith("test_"):
            continue
        try:
            module = importlib.import_module(f"bridge.tests.{mod_name}")
        except (ModuleNotFoundError, ImportError) as e:
            skipped += 1
            print(f"SKIP {mod_name} (unimportable: {e})")
            continue
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            name = f"{mod_name}.{attr}"
            try:
                result = func()
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
                continue
            if result == "SKIP":
                skipped += 1
                print(f"SKIP {name}")
            else:
                passed += 1
                print(f"ok   {name}")
    print(f"\n{passed} passed, {skipped} skipped, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
