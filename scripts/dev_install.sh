#!/usr/bin/env bash
# scripts/dev_install.sh — Build and install audiolib in development (editable) mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing Rust toolchain (if not present)..."
if ! command -v cargo &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

echo "==> Installing maturin..."
pip install "maturin>=1.5,<2.0"

echo "==> Installing Python dev dependencies..."
pip install numpy soundfile pytest pytest-cov ruff

echo "==> Building and installing audiolib..."
if python -c "import sys; sys.exit(0 if (hasattr(sys, 'real_prefix') or sys.prefix != sys.base_prefix) else 1)" 2>/dev/null; then
  maturin develop --release
else
  maturin build --release --out dist
  pip install --no-index --find-links dist/ audiolib
fi

echo ""
echo "Done! audiolib is installed."
echo "Run tests with: pytest tests/ -v"
