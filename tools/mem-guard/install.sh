#!/usr/bin/env bash
# Install mem-guard as a per-user systemd service. No root required.
#   ./install.sh            # install + start + enable on boot
#   ./install.sh uninstall  # stop + remove
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin"
UNIT="$HOME/.config/systemd/user"

if [ "${1:-}" = "uninstall" ]; then
  systemctl --user disable --now mem-guard.service 2>/dev/null || true
  rm -f "$BIN/mem-guard.sh" "$UNIT/mem-guard.service"
  systemctl --user daemon-reload
  echo "mem-guard removed. (Logs left at ~/.cache/mem-guard/)"
  exit 0
fi

mkdir -p "$BIN" "$UNIT" "$HOME/.cache/mem-guard"
install -m 0755 "$SRC/mem-guard.sh" "$BIN/mem-guard.sh"
install -m 0644 "$SRC/mem-guard.service" "$UNIT/mem-guard.service"

systemctl --user daemon-reload
systemctl --user enable --now mem-guard.service
# keep it running across logout/reboot (allowed for own user via polkit on most setups)
loginctl enable-linger "$USER" 2>/dev/null || echo "note: could not enable linger; guard runs while you are logged in"

echo "mem-guard installed and running."
echo "  log:    ~/.cache/mem-guard/mem-guard.log   (tail -f to watch)"
echo "  status: systemctl --user status mem-guard"
