#!/usr/bin/env bash
# Open the SSH tunnels defined in run.config.sh. Portable: no hardcoded hosts.
# Detached + keepalive so an editor/OOM restart on this machine doesn't silently
# drop a tunnel (the dashboard depends on these being up).
set -uo pipefail
cd "$(dirname "$0")"
[ -f run.config.sh ] && source run.config.sh
TUNNELS=("${TUNNELS[@]:-}")

if [ "${#TUNNELS[@]}" -eq 0 ] || [ -z "${TUNNELS[0]:-}" ]; then
  echo "No TUNNELS configured (see run.config.example.sh). Nothing to open."
  exit 0
fi

OPTS="-fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
for t in "${TUNNELS[@]}"; do
  [ -z "$t" ] && continue
  spec="${t%@*}"; target="${t#*@}"; lport="${spec%%:*}"
  if ss -ltn 2>/dev/null | grep -q ":$lport "; then
    echo "  :$lport already up"
  elif ssh $OPTS -L "$spec" "$target"; then
    echo "  :$lport -> $target  ok"
  else
    echo "  :$lport -> $target  FAILED (check ssh access to $target)"
  fi
done
