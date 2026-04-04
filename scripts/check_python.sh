#!/usr/bin/env bash
# Syntax-check all Python sources in the repo (no .pyc writes).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - "$@" << 'PY'
import ast
import sys
from pathlib import Path

root = Path("my_implementation")
files = sorted(root.rglob("*.py"))
if not any(files):
    print("No Python files found.", file=sys.stderr)
    sys.exit(1)

for path in files:
    if ".venv" in path.parts:
        continue
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    print(f"OK {path}")
PY

if command -v docker >/dev/null 2>&1; then
  docker compose -f my_implementation/P3SL_Implementation/1st_Demo/docker-compose.yml config >/dev/null
  echo "OK P3SL docker-compose.yml"
  docker compose -f my_implementation/docker-socket-demo/docker-compose.yml config >/dev/null
  echo "OK docker-socket-demo docker-compose.yml"
fi

echo "All checks passed."
