#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

if [[ ! -f graphify-out/.graphify_python ]]; then
  echo "graphify-out/.graphify_python is missing; run /graphify once first." >&2
  exit 1
fi

graphify_python="$(<graphify-out/.graphify_python)"
"$graphify_python" tools/refine_graph.py --refresh-code
graphify export html \
  --input graphify-out/graph.json \
  --output graphify-out/graph.html \
  --analysis graphify-out/.graphify_analysis.json \
  --force

echo "Code graph refreshed. Run /graphify --update after document changes."
