#!/usr/bin/env bash
set -euo pipefail

# Create virtual environment if it doesn't exist
if [ ! -d "ai" ]; then
    python -m venv ai
fi

# Activate virtual environment
if [ -f "ai/bin/activate" ]; then
    . ai/bin/activate
else
    echo "[env_ai.sh] Failed to find venv activation script at ai/bin/activate" >&2
    exit 1
fi

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[env_ai.sh] Virtual environment is ready and dependencies are installed."


