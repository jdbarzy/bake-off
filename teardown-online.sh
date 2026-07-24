#!/usr/bin/env bash
# Stop PUBLIC hosting: kill the Cloudflare tunnel + supervisor, then bring the
# dashboard back up locally (LAN only, no public URL). Run automatically at 48h
# by the llmbench-teardown timer, or manually any time.
cd "$(dirname "$0")"
[ -f run.config.sh ] && source run.config.sh
DASH_PORT="${DASH_PORT:-15600}"

systemctl --user disable --now llmbench-online.service 2>/dev/null || true   # stops supervisor + its cloudflared/dashboard
pkill -f 'cloudflared tunnel --url' 2>/dev/null || true
rm -f cloudflared-url.txt
sleep 1

# bring the dashboard back for local/LAN use (no public tunnel)
./tunnels.sh >/dev/null 2>&1 || true
if ! ss -ltn 2>/dev/null | grep -q ":$DASH_PORT "; then
  DASH_HOST="${DASH_HOST:-0.0.0.0}" DASH_PORT="$DASH_PORT" DASH_AUTH="${DASH_AUTH:-}" \
    nohup python3 dashboard/server.py >> dashboard/dashboard.log 2>&1 &
fi
echo "Public hosting stopped. Dashboard still on the LAN."
