#!/usr/bin/env bash
# Supervisor for temporary PUBLIC hosting: keeps the SSH tunnels, the dashboard
# (with DASH_AUTH login), and a Cloudflare quick tunnel all up. Run by the
# llmbench-online systemd user service. The public URL is written to
# cloudflared-url.txt. Stop with ./teardown-online.sh.
set -uo pipefail
cd "$(dirname "$0")"
[ -f run.config.sh ] && source run.config.sh
DASH_HOST="${DASH_HOST:-0.0.0.0}"; DASH_PORT="${DASH_PORT:-15600}"; DASH_AUTH="${DASH_AUTH:-}"
CF="$HOME/.local/bin/cloudflared"
CFLOG="$PWD/dashboard/cloudflared.log"
URLFILE="$PWD/cloudflared-url.txt"

up(){ ss -ltn 2>/dev/null | grep -q ":$1 "; }

while :; do
  ./tunnels.sh >/dev/null 2>&1 || true                       # keep model tunnels up (idempotent)

  if ! up "$DASH_PORT"; then                                 # keep the dashboard up
    DASH_HOST="$DASH_HOST" DASH_PORT="$DASH_PORT" DASH_AUTH="$DASH_AUTH" \
      nohup python3 dashboard/server.py >> dashboard/dashboard.log 2>&1 &
    sleep 3
  fi

  if ! pgrep -f 'cloudflared tunnel --url' >/dev/null 2>&1; then   # keep the public tunnel up
    : > "$CFLOG"
    nohup "$CF" tunnel --url "http://localhost:$DASH_PORT" --no-autoupdate >> "$CFLOG" 2>&1 &
    for _ in $(seq 1 25); do
      u=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CFLOG" | head -1)
      [ -n "$u" ] && { echo "$u" > "$URLFILE"; break; }
      sleep 1
    done
  fi

  sleep 30
done
