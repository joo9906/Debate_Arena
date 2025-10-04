#!/usr/bin/env sh
set -e

# Always run process.py first (like the original run.ps1)
echo "[ai-server] Running data preprocessing..."
cd /app/data
python process.py
cd ..

exec "$@"


