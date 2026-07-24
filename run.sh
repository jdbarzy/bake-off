#!/usr/bin/env bash
# One command to bring Bake-off online: open the configured SSH tunnels and start
# the dashboard, network-accessible. Portable - all host/port settings live in
# run.config.sh (copy run.config.example.sh). No hosts or IPs are hardcoded here.
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -f run.config.sh ]; then
  echo "No run.config.sh found. Create one:"
  echo "  cp run.config.example.sh run.config.sh   # then edit for your network"
  exit 1
fi
set -a; source run.config.sh; set +a   # auto-export: every DASH_* var reaches the server, no list to maintain
export DASH_HOST="${DASH_HOST:-0.0.0.0}"
export DASH_PORT="${DASH_PORT:-15600}"

# 1. tunnels (idempotent; skips any already up)
if [ -f tunnels.sh ]; then echo "Opening tunnels..."; ./tunnels.sh; echo; fi

# 2. (re)start the dashboard on the configured bind
pid=$(ss -ltnp 2>/dev/null | grep ":$DASH_PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$pid" ] && { echo "Stopping existing dashboard (pid $pid)..."; kill "$pid" 2>/dev/null; sleep 1; }
# The server owns the http-vs-https decision (incl. the openssl fallback) - ask it.
SCHEME="$(python3 dashboard/server.py --scheme)"
echo "Starting dashboard on $DASH_HOST:$DASH_PORT ..."
nohup python3 dashboard/server.py > dashboard/dashboard.log 2>&1 &
sleep 2

# 3. report access URLs (LAN IP auto-detected, never hardcoded)
echo
if ss -ltn 2>/dev/null | grep -q ":$DASH_PORT "; then
  echo "Bake-off is live:"
  echo "  $SCHEME://localhost:$DASH_PORT/"
  if [ "$DASH_HOST" = "0.0.0.0" ]; then
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    [ -n "$ip" ] && echo "  $SCHEME://$ip:$DASH_PORT/   (share on your LAN)"
  fi
  echo
  echo "Stop everything with: ./teardown.sh"
else
  echo "Dashboard failed to start - last log lines:"; tail -5 dashboard/dashboard.log
  exit 1
fi
