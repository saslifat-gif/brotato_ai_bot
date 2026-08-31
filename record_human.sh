#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR:$PYTHONPATH"
export BROTATO_V4_POLICY_MODE="HANDCRAFTED"
export BROTATO_V4_AUTOMATE_MENUS="0"

OUTPUT="${1:-$SCRIPT_DIR/models/version_3/human_demos/manual_run.sqlite}"
shift 1 2>/dev/null || true

echo "[v4-human-demo] output=$OUTPUT"
echo "[v4-human-demo] play manually; press F9 to bookmark meaningful states"
python -u -m v4.record_human_demo --output "$OUTPUT" --run-label manual --require-capture "$@"
