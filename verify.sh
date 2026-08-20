#!/usr/bin/env bash
# Verify script — test + build + smoke in one command.
set -e

echo "=== ai-eyes-mcp verify ==="

echo "[1/5] Import check..."
python -c "from ai_eyes_mcp.server import mcp; from ai_eyes_mcp.engine import SigLIPEngine; print('  imports OK')"

echo "[2/5] Tool registration check..."
python -c "
from ai_eyes_mcp import EXPECTED_TOOL_NAMES
from ai_eyes_mcp.server import mcp
import asyncio
tools = asyncio.run(mcp._list_tools())
names = {t.name for t in tools}
assert names == EXPECTED_TOOL_NAMES, f'Expected {EXPECTED_TOOL_NAMES}, got {names}'
print(f'  {len(tools)} tools registered OK')
"

echo "[3/5] Engine status check..."
python -c "
from ai_eyes_mcp.engine import SigLIPEngine, assert_cold_status
s = SigLIPEngine().status()
assert_cold_status(s)
print(f'  engine OK: model={s[\"model_id\"]}, revision={s[\"revision\"]}, loaded={s[\"loaded\"]}')
"

echo "[4/5] Build check..."
if python -m build --help > /dev/null 2>&1; then
  python -m build --wheel --no-isolation
  echo "  wheel OK"
else
  echo "  SKIP: python-build not installed (pip install build)"
fi

echo "[5/5] CI-safe tests (pytest -m 'not dogfood')..."
pytest -m "not dogfood" -v

echo ""
echo "=== verify passed ==="
