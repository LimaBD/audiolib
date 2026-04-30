#!/usr/bin/env bash
# scripts/publish.sh — Build and publish to PyPI (or TestPyPI).
#
# Usage:
#   ./scripts/publish.sh            → publish to PyPI
#   ./scripts/publish.sh --test     → publish to TestPyPI
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET=pypi
if [[ "${1:-}" == "--test" ]]; then
  TARGET=testpypi
fi

echo "==> Installing build tools..."
pip install "maturin>=1.5,<2.0" twine

echo "==> Running tests before publish..."
pytest tests/ -v --ignore=tests/benchmark.py -x

echo "==> Building release wheels..."
maturin build --release --out dist/

echo "==> Building sdist..."
maturin sdist --out dist/

if [[ "$TARGET" == "testpypi" ]]; then
  echo "==> Uploading to TestPyPI..."
  twine upload --repository testpypi dist/*
else
  echo "==> Uploading to PyPI..."
  twine upload dist/*
fi

echo ""
echo "Published to $TARGET successfully."
