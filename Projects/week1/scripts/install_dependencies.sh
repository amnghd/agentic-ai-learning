#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing project dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "==> Installing project in editable mode (optional)..."
if pip install -e "$PROJECT_ROOT" 2>/dev/null; then
    echo "  Project installed in editable mode."
else
    echo "  Warning: Could not install project in editable mode (this is optional)."
    echo "  Dependencies are still installed and ready to use."
fi

echo "==> Dependencies installed."
