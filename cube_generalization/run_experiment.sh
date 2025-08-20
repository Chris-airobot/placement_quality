#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/home/chris/Chris/placement_ws/src/placement_quality"
PY="python3"
EXP="$EXP_DIR/Experiment.py"

OUT_DIR="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments"
RESULTS="$OUT_DIR/results.jsonl"
RESUME="$OUT_DIR/resume.txt"
TOTAL=8640

mkdir -p "$OUT_DIR"

get_next_index() {
  if [[ -s "$RESULTS" ]]; then
    # read last non-empty line and extract "index"
    last_line="$(tac "$RESULTS" | grep -m1 . || true)"
    if [[ -n "$last_line" ]]; then
      # Use python for robust JSON parse
      next=$(python3 - <<'PY'
import sys, json
line = sys.stdin.read().strip()
try:
    idx = json.loads(line).get("index", -1)
    print(int(idx) + 1 if isinstance(idx, int) else 0)
except Exception:
    print(0)
PY
<<< "$last_line")
      echo "$next"
      return
    fi
  fi
  echo 0
}

while true; do
  NEXT=$(get_next_index)
  if (( NEXT >= TOTAL )); then
    echo "All $TOTAL cases are complete. Done."
    exit 0
  fi

  echo "$NEXT" > "$RESUME"
  echo "[runner] Starting Experiment.py from index $NEXT ..."
  set +e
  "$PY" "$EXP"
  code=$?
  set -e

  if (( code == 0 )); then
    # Script exited cleanly; check if we’re done
    NEXT=$(get_next_index)
    if (( NEXT >= TOTAL )); then
      echo "All $TOTAL cases are complete. Done."
      exit 0
    fi
    echo "[runner] Experiment.py exited but not complete; relaunching..."
  else
    echo "[runner] Experiment.py crashed with code $code; resuming..."
  fi
  # Loop continues: recompute NEXT and relaunch
done
