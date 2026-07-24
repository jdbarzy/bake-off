#!/usr/bin/env bash
# Bake-off one-command installer.
#
#   curl -fsSL https://raw.githubusercontent.com/jdbarzy/bake-off/HEAD/get.sh | bash
#
# Downloads Bake-off to ~/bake-off and installs it as an always-on app
# (auto-starts on boot, opens from any device on your network).
# Safe to re-run: it updates the app and keeps your settings.
set -euo pipefail

REPO_TARBALL="https://github.com/jdbarzy/bake-off/archive/HEAD.tar.gz"
DEST="$HOME/bake-off"

say(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
die(){ printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

say "Bake-off installer"

# ---- prerequisites (with plain fixes) ----
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
  || die "Need curl (or wget). On Ubuntu/Debian:  sudo apt install curl"
command -v tar >/dev/null 2>&1 || die "Need tar. On Ubuntu/Debian:  sudo apt install tar"
command -v python3 >/dev/null 2>&1 \
  || die "Need Python 3 (preinstalled on most Linux). On Ubuntu/Debian:  sudo apt install python3"

# An install from before the rename lives at ~/llm-bench: carry it (and its
# settings) over to ~/bake-off once, so a re-run upgrades instead of forking.
if [ -d "$HOME/llm-bench" ] && [ ! -d "$DEST" ]; then
  mv "$HOME/llm-bench" "$DEST"
  echo "  moved your existing ~/llm-bench install to $DEST (settings kept)"
fi

# ---- download ----
say "Downloading Bake-off..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$REPO_TARBALL" -o "$TMP/bake-off.tar.gz"
else
  wget -qO "$TMP/bake-off.tar.gz" "$REPO_TARBALL"
fi

# unpack to a staging dir first: a broken download must never half-overwrite a working install
mkdir -p "$TMP/unpack"
tar -xzf "$TMP/bake-off.tar.gz" -C "$TMP/unpack" --strip-components=1
[ -f "$TMP/unpack/dashboard/server.py" ] || die "The download looked wrong - nothing was changed. Try again in a minute."
mkdir -p "$DEST"
cp -a "$TMP/unpack/." "$DEST/"
echo "  installed to $DEST (your settings, if any, were kept)"

# ---- install as an always-on app ----
cd "$DEST"
chmod +x install.sh uninstall.sh run.sh 2>/dev/null || true
bash install.sh
