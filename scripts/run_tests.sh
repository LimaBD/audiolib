#!/usr/bin/env bash
# scripts/run_tests.sh — Install dependencies and run the full test suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing dependencies and building audiolib..."
"$ROOT/scripts/dev_install.sh"

PYTEST_ARGS="${*:--v --ignore=tests/benchmark.py}"

echo "==> Running tests: pytest $PYTEST_ARGS"
pytest $PYTEST_ARGS

echo ""
echo "==> Done."
