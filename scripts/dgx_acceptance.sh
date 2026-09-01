#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
docker compose config >/dev/null
echo "Core acceptance passed. Start Neo4j with: docker compose up -d"
echo "Live K3 gate requires NVIDIA_API_KEY and LEGALBENCH_LIVE_CONFIRM=true."
