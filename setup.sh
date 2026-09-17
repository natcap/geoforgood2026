#!/usr/bin/env bash
# NatCap agent sandbox — setup.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
else
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt
fi

[ -f .env ] || { cp .env.example .env; echo "Created .env — edit it with your GCP details."; }

cat <<'MSG'

Done. Next:
  1. (optional) put a service-account key at  secrets/sa-key.json
  2. Edit .env (project id; VERTEX_API_KEY or GEMINI_API_KEY if using one)
  3. source .venv/bin/activate
  4. python -c "from natcap_agents.agents import build_crew; build_crew().run('hello')"
MSG
