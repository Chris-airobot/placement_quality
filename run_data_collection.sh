#!/bin/bash

# Safe bash settings - don't exit on errors, we handle them manually
set -u
set -o pipefail
# Don't exit on errors, we want the loop to continue
trap 'echo "[runner] Caught error, but continuing..."' ERR

# Configuration
ISAAC_SIM_PY="/home/riot/isaacsim_4.2.0/python.sh"
PY_MODULE="cube_generalization.data_collection"

# Must match DataCollection.data_folder in cube_generalization/data_collection.py
OUTPUT_DIR="/home/riot/Chris/data/box_simulation/v5/data_collection/raw_data"

# Control knobs
MIN_FILES_PER_RUN=1
MAX_FILES_PER_RUN=3
SLEEP_BETWEEN_RUNS_SEC=10
POLL_INTERVAL_SEC=5
# Disable progress/time watchdogs by default (prevents mid-pose termination)
NO_PROGRESS_TIMEOUT_SEC=0
RUN_TIMEOUT_SEC=0

# Basic log directory for per-run diagnostics
LOG_ROOT_DIR="/home/riot/Chris/placement_quality/logs/data_collection"
mkdir -p "$LOG_ROOT_DIR"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

count_json_files() {
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.json" | wc -l | tr -d ' \t\n'
}

heartbeat_mtime() { echo 0; }

start_data_collection() {
  echo "[runner] Launching data collection..."
  # Suppress excessive logging similar to the template
  # Start Isaac in its own session so we can kill the whole process group later.
  # Lower CPU and IO priority to reduce system stutter.
  ionice -c2 -n7 nice -n 10 \
  setsid "$ISAAC_SIM_PY" -m "$PY_MODULE" \
    --no-window \
    --/app/window/enabled=false \
    --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error &
  DAEMON_PID=$!
  # Capture the process group id; negative PGID targets the whole group
  DAEMON_PGID=$(ps -o pgid= "$DAEMON_PID" | tr -d ' \t\n') || DAEMON_PGID=""
  echo "[runner] PID: $DAEMON_PID (PGID: ${DAEMON_PGID:-unknown})"
}

start_monitors() {
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  RUN_LOG_DIR="$LOG_ROOT_DIR/$RUN_ID"
  mkdir -p "$RUN_LOG_DIR"
  echo "[runner] Logs: $RUN_LOG_DIR"
  GPU_MON_PID=""; VMSTAT_PID=""; IOSTAT_PID=""; DMESG_PID=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d 1 -f "$RUN_LOG_DIR/gpu_dmon.csv" & GPU_MON_PID=$!
  fi
  if command -v vmstat >/dev/null 2>&1; then
    vmstat 1 > "$RUN_LOG_DIR/vmstat.log" & VMSTAT_PID=$!
  fi
  if command -v iostat >/dev/null 2>&1; then
    iostat -xz 1 > "$RUN_LOG_DIR/iostat.log" & IOSTAT_PID=$!
  fi
  # Kernel log tail (might require permissions; ignore if fails)
  dmesg --follow --ctime > "$RUN_LOG_DIR/dmesg_tail.log" 2>/dev/null & DMESG_PID=$!
}

stop_monitors() {
  for p in "$GPU_MON_PID" "$VMSTAT_PID" "$IOSTAT_PID" "$DMESG_PID"; do
    if [ -n "$p" ] 2>/dev/null && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
}

terminate_process() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    # Determine process group for full-tree termination
    local pgid
    pgid=$(ps -o pgid= "$pid" | tr -d ' \t\n') || pgid=""

    if [ -n "$pgid" ]; then
      echo "[runner] Stopping process group PGID $pgid (SIGTERM)..."
      # Send SIGTERM to the whole process group
      kill -TERM -"$pgid" 2>/dev/null || true
      # Wait up to 30s for graceful shutdown of the whole group
      for i in $(seq 1 30); do
        if ! kill -0 -"$pgid" 2>/dev/null; then
          echo "[runner] Process group $pgid exited gracefully."
          wait "$pid" 2>/dev/null || true
          return 0
        fi
        sleep 1
      done
      echo "[runner] Forcing process group $pgid to stop (SIGKILL)."
      kill -KILL -"$pgid" 2>/dev/null || true
      # Reap if possible
      wait "$pid" 2>/dev/null || true
    else
      echo "[runner] PGID not found for PID $pid. Stopping the PID only (SIGTERM)..."
      kill "$pid" 2>/dev/null || true
      for i in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
          echo "[runner] PID $pid exited gracefully."
          wait "$pid" 2>/dev/null || true
          return 0
        fi
        sleep 1
      done
      echo "[runner] Forcing PID $pid to stop (SIGKILL)."
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
  # Fallback: terminate any remaining module processes, then kill -9
  pgrep -f "$PY_MODULE" | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
  pgrep -f "$PY_MODULE" | xargs -r kill -KILL 2>/dev/null || true
}

choose_target_file_count() {
  local min=$1
  local max=$2
  # RANDOM yields 0..32767. Spread into [min, max]
  echo $(( (RANDOM % (max - min + 1)) + min ))
}

# Signal handler for clean shutdown
cleanup() {
  echo "[runner] Received shutdown signal. Cleaning up..."
  if [ -n "${DAEMON_PID:-}" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    terminate_process "$DAEMON_PID"
  fi
  echo "[runner] Cleanup complete. Exiting."
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "=== Data Collection Runner (throttled) ==="
echo "Isaac Python: $ISAAC_SIM_PY"
echo "Module:       $PY_MODULE"
echo "Output dir:   $OUTPUT_DIR"
echo "Files per run: $MIN_FILES_PER_RUN..$MAX_FILES_PER_RUN"
echo "Sleep between runs: ${SLEEP_BETWEEN_RUNS_SEC}s"
echo "Press Ctrl+C to stop gracefully"

run_count=0
while true; do
  run_count=$((run_count + 1))
  echo ""
  echo "=== Run #$run_count ==="
  base_count=$(count_json_files)
  target_files=$(choose_target_file_count "$MIN_FILES_PER_RUN" "$MAX_FILES_PER_RUN")
  echo "[runner] Baseline files: $base_count"
  echo "[runner] Target new files this run: $target_files"

  start_data_collection
  start_monitors

  # Monitor the process and file count
  while true; do
    now_ts=$(date +%s)
    # Init run start and last progress timestamps
    if [ -z "${run_start_ts:-}" ]; then run_start_ts=$now_ts; fi
    if [ -z "${last_progress_ts:-}" ]; then last_progress_ts=$now_ts; fi
    # Heartbeat disabled
    # Check if process is still running
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
      echo "[runner] Process $DAEMON_PID exited on its own."
      break
    fi
    
    # Check file count
    current_count=$(count_json_files)
    new_files=$(( current_count - base_count ))
    echo "[runner] New files so far: $new_files / $target_files (pid=$DAEMON_PID)"
    
    if [ "$new_files" -ge "$target_files" ]; then
      echo "[runner] Target reached. Stopping the process."
      terminate_process "$DAEMON_PID"
      break
    fi
    # Watchdogs disabled by default
    
    sleep "$POLL_INTERVAL_SEC"
  done

  # Ensure process is completely stopped
  if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[runner] Force stopping any remaining process..."
    terminate_process "$DAEMON_PID"
  fi
  stop_monitors

  echo "[runner] Cooling down for ${SLEEP_BETWEEN_RUNS_SEC}s..."
  sleep "$SLEEP_BETWEEN_RUNS_SEC"
  # Reset timestamps for next run
  unset run_start_ts
  unset last_progress_ts
done


