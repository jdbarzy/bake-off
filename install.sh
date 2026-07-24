#!/usr/bin/env bash
# Bake-off appliance installer (run once, on the control machine).
#
# Turns this machine into an always-on Bake-off appliance:
#   - the dashboard auto-starts on boot (systemd user service, auto-restart)
#   - any SSH tunnels in run.config.sh auto-start + auto-restart
#   - a desktop "Bake-off" launcher opens the dashboard
#
# After this, nobody needs a terminal: open the app (or the URL) from any device
# and click. Undo any time with ./uninstall.sh.
#
# Requires a Linux control machine with systemd (user services) + python3.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

command -v systemctl >/dev/null 2>&1 || { echo "This installer needs systemd (standard on Ubuntu, Debian, Fedora, Mint)."; echo "No systemd? Start it manually instead:  ./run.sh"; exit 1; }
command -v python3   >/dev/null 2>&1 || { echo "Python 3 is missing. On Ubuntu/Debian fix with:  sudo apt install python3"; exit 1; }

# A minimal run.config.sh so a first-timer needs zero edits: dashboard on the LAN,
# no tunnels (point the three slots at your servers inside the app: Edit endpoints).
if [ ! -f run.config.sh ]; then
  cat > run.config.sh <<'CFG'
# Bake-off runtime config (yours; gitignored). Safe defaults below.
DASH_HOST="0.0.0.0"      # 0.0.0.0 = reachable on your LAN; 127.0.0.1 = this machine only
DASH_PORT="15600"
DASH_AUTH=""             # "user:pass" to require a login
DASH_TLS=""              # "self" = serve https with an auto-created certificate (browsers warn once
                         # per device; needed if you set DASH_AUTH on a shared network). "" = plain http.
# Have real certificates? Point at them instead of DASH_TLS:
#   DASH_TLS_CERT="/path/cert.pem"; DASH_TLS_KEY="/path/key.pem"
# Only if your model servers are reachable ONLY over SSH, add tunnels here, one per line:
#   "LOCALPORT:localhost:REMOTEPORT@sshalias"
TUNNELS=()
CFG
  echo "Created run.config.sh with safe defaults."
fi
set -a; source run.config.sh; set +a   # auto-export: every DASH_* var reaches the server, no list to maintain
DASH_PORT="${DASH_PORT:-15600}"
# The server owns the http-vs-https decision (incl. the openssl fallback) - ask it.
SCHEME="$(python3 dashboard/server.py --scheme)"

UD="$HOME/.config/systemd/user"; mkdir -p "$UD"

# Clean slate: uninstall.sh owns the list of installed artifacts (both the bake-off
# and legacy llm-bench names) - retire whatever exists, then install fresh below.
bash ./uninstall.sh >/dev/null

echo "Installing dashboard service..."
cat > "$UD/bake-off-dashboard.service" <<EOF
[Unit]
Description=Bake-off dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=/bin/bash -c 'set -a; source run.config.sh; set +a; exec /usr/bin/env python3 dashboard/server.py'
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

echo "Installing tunnel services (from run.config.sh)..."
rm -f "$UD"/bake-off-tunnel-*.generated.service
for t in "${TUNNELS[@]:-}"; do
  [ -z "${t:-}" ] && continue
  spec="${t%@*}"; target="${t#*@}"; lport="${spec%%:*}"
  cat > "$UD/bake-off-tunnel-${lport}.generated.service" <<EOF
[Unit]
Description=Bake-off SSH tunnel :$lport -> $target
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L $spec $target
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
  echo "  tunnel :$lport -> $target"
done

echo "Installing desktop launcher..."
APPDIR="$HOME/.local/share/applications"; mkdir -p "$APPDIR"
cat > "$APPDIR/bake-off.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Bake-off
Comment=Open the Bake-off dashboard
Exec=xdg-open $SCHEME://localhost:$DASH_PORT/
Icon=web-browser
Terminal=false
Categories=Utility;Science;
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPDIR" >/dev/null 2>&1 || true

echo "Enabling auto-start on boot (no login required)..."
loginctl enable-linger "${USER:-$(id -un)}" >/dev/null 2>&1 || true
systemctl --user daemon-reload
systemctl --user enable --now bake-off-dashboard.service
for u in "$UD"/bake-off-tunnel-*.generated.service; do
  [ -e "$u" ] && systemctl --user enable --now "$(basename "$u")"
done

echo
echo "Done - Bake-off is installed and starts on every boot."
echo
echo "  Open it:"
echo "    this machine:      $SCHEME://localhost:$DASH_PORT/  (or click the 'Bake-off' app)"
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -n "${ip:-}" ] && echo "    any other device:  $SCHEME://$ip:$DASH_PORT/"
[ "$SCHEME" = "https" ] && [ -z "${DASH_TLS_CERT:-}" ] && echo "    (https with a self-made certificate: your browser will ask you to trust it once)"
echo
echo "  Next: get some models. On each GPU machine, run the provisioner:"
echo "    bash mwboot.sh"
echo "  It installs vLLM, serves a model per GPU with a swap agent, and prints"
echo "  the exact settings to paste into the app."
echo
echo "  Full walkthrough: INSTALL.html in this folder. Remove any time: ./uninstall.sh"
