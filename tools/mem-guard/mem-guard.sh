#!/usr/bin/env bash
# mem-guard: diagnose + mitigate the recurring OOM that kills VSCode on low-memory Linux.
#
# WHAT IT DOES
#   1. Every INTERVAL seconds, logs a memory snapshot to LOG. The decisive column
#      is GAP_MiB = (MemTotal-MemAvailable) - sum(process RSS) - Slab - PageTables.
#      If GAP climbs while process RSS stays flat, the memory is kernel/driver
#      held (prime suspect: the NVIDIA stack) rather than any app.
#   2. When MemAvailable drops below LOW_MiB (pre-OOM territory), it RAISES the
#      OOM-kill priority (oom_score_adj=1000) on the single largest-RSS process
#      that is NOT in PROTECT, so the kernel sacrifices that hog instead of `code`.
#      It only ever RAISES scores on this user's own processes (no root needed)
#      and never touches anything in PROTECT.
#
# Stop with: systemctl --user stop mem-guard
# Logs:      ~/.cache/mem-guard/mem-guard.log

set -uo pipefail

INTERVAL="${MEM_GUARD_INTERVAL:-10}"     # seconds between samples
LOW_MIB="${MEM_GUARD_LOW_MIB:-1500}"     # MemAvailable below this => protect code
LOG="${MEM_GUARD_LOG:-$HOME/.cache/mem-guard/mem-guard.log}"
MAXLOG=$((20 * 1024 * 1024))             # rotate log at 20 MiB

# Never redirect a kill onto these (substring match on comm).
PROTECT_RE='^(code|gnome-shell|Xorg|gnome-remote-de|claude|sshd|systemd|dbus-daemon|pipewire|wireplumber|gnome-session|bash|mem-guard)$'

meminfo() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }

rotate() {
  [ -f "$LOG" ] || return 0
  local sz; sz=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  [ "$sz" -gt "$MAXLOG" ] && mv -f "$LOG" "$LOG.1"
  return 0
}

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts) mem-guard started (interval=${INTERVAL}s low=${LOW_MIB}MiB pid=$$)" >> "$LOG"

while :; do
  rotate

  memtotal=$(meminfo MemTotal)
  memavail=$(meminfo MemAvailable)
  swapfree=$(meminfo SwapFree)
  slab=$(meminfo Slab)
  ptab=$(meminfo PageTables)
  sumrss=$(ps -eo rss= 2>/dev/null | awk '{s+=$1} END{print s+0}')

  used=$((memtotal - memavail))
  gap=$(( (used - sumrss - slab - ptab) / 1024 ))   # MiB unaccounted (driver/kernel)
  avail_mib=$((memavail / 1024))
  swapfree_mib=$((swapfree / 1024))
  rss_mib=$((sumrss / 1024))

  printf '%s avail=%dMiB swapfree=%dMiB procRSS=%dMiB gap=%dMiB | top:' \
    "$(ts)" "$avail_mib" "$swapfree_mib" "$rss_mib" "$gap" >> "$LOG"
  # top 6 RSS processes, compact
  ps -eo comm=,rss= --sort=-rss 2>/dev/null | head -6 \
    | awk '{printf " %s=%dM", $1, $2/1024}' >> "$LOG"
  echo >> "$LOG"

  # --- mitigation: redirect the kill away from VSCode when memory is critical ---
  if [ "$avail_mib" -lt "$LOW_MIB" ]; then
    # largest-RSS user process not in PROTECT
    victim=$(ps -o pid=,comm=,rss= -u "$(id -u)" --sort=-rss 2>/dev/null \
      | awk -v re="$PROTECT_RE" '{c=$2} c !~ re {print $1, $2; exit}')
    vpid=$(echo "$victim" | awk '{print $1}')
    vcomm=$(echo "$victim" | awk '{print $2}')
    if [ -n "${vpid:-}" ] && [ -w "/proc/$vpid/oom_score_adj" ]; then
      cur=$(cat "/proc/$vpid/oom_score_adj" 2>/dev/null || echo 0)
      if [ "${cur:-0}" -lt 1000 ]; then
        if echo 1000 > "/proc/$vpid/oom_score_adj" 2>/dev/null; then
          echo "$(ts) LOW MEM (avail=${avail_mib}MiB) -> raised oom_score_adj=1000 on pid=$vpid ($vcomm) to spare VSCode" >> "$LOG"
        fi
      fi
    fi
  fi

  sleep "$INTERVAL"
done
