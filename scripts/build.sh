#!/usr/bin/env bash
# scripts/build.sh — Build release wheels locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing maturin..."
pip install "maturin>=1.5,<2.0"

echo "==> Building release wheels..."
maturin build --release --out dist/

echo ""
echo "Wheels written to dist/:"
ls -lh dist/
