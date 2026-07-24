#!/usr/bin/env bash
# Undo install.sh: stop + remove the Bake-off appliance services and launcher
# (including any left from when the install was named llm-bench).
# Your run.config.sh, model servers, and any hand-made tunnels are left untouched.
set -uo pipefail
cd "$(dirname "$0")"
UD="$HOME/.config/systemd/user"

echo "Stopping and removing Bake-off services..."
for svc in bake-off-dashboard.service llm-bench-dashboard.service; do
  systemctl --user disable --now "$svc" 2>/dev/null || true
  rm -f "$UD/$svc"
done
for u in "$UD"/bake-off-tunnel-*.generated.service "$UD"/llm-bench-tunnel-*.generated.service; do
  [ -e "$u" ] || continue
  systemctl --user disable --now "$(basename "$u")" 2>/dev/null || true
  rm -f "$u"
done
rm -f "$HOME/.local/share/applications/bake-off.desktop" "$HOME/.local/share/applications/llm-bench.desktop"
systemctl --user daemon-reload
echo "Removed. (Run ./run.sh to start the dashboard manually again.)"
