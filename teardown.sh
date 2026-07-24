#!/usr/bin/env bash
# Stop Bake-off: the dashboard + the SSH tunnels from run.config.sh. Portable:
# kills by listening port, so no host names are hardcoded.
set -uo pipefail
cd "$(dirname "$0")"
[ -f run.config.sh ] && source run.config.sh
DASH_PORT="${DASH_PORT:-15600}"
TUNNELS=("${TUNNELS[@]:-}")

kill_on_port() {  # kill whatever is listening on $1
  local port="$1" pid
  pid=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && { echo "  stopping :$port (pid $pid)"; kill "$pid" 2>/dev/null; }
}

echo "Stopping dashboard..."; kill_on_port "$DASH_PORT"
echo "Closing tunnels..."
for t in "${TUNNELS[@]}"; do
  [ -z "$t" ] && continue
  kill_on_port "${t%%:*}"
done
echo "Stopped. (Remote model servers are left running.)"
