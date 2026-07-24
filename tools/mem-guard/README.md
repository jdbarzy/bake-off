# mem-guard

Diagnose and mitigate recurring out-of-memory (OOM) crashes on memory-constrained
Linux machines. Built for Linux workstations that swap-thrash under memory pressure on Ubuntu, but it works on any systemd-based Linux.

## What's actually happening

VSCode "crashes" on a low-memory box are often **Linux OOM kills, not a VSCode bug**.
When RAM runs low, the kernel kills `code` first because its Electron helper
processes carry the highest OOM-kill priority (`oom_score_adj=300`). VSCode is
the victim, not the cause.

The memory is exhausted by **unaccounted kernel/driver memory** (suspected
NVIDIA stack), not local workloads. At a confirmed crash, user processes held
only ~7 GiB yet ~23 GiB of RAM was held outside any process, while 24 GiB of
swap sat unused. So **adding swap does not help**. A reboot clears it; updating
the NVIDIA driver is the durable fix.

Confirm on your own box:

```bash
journalctl -k | grep -i "Out of memory"
```

## What mem-guard does (no root required)

1. **Logs** a memory snapshot every 10s to `~/.cache/mem-guard/mem-guard.log`.
   The key column is `gap` = RAM used by neither processes, slab, nor page
   tables. When healthy it is slightly negative (process RSS double-counts
   shared pages). **If `gap` climbs large and positive while `procRSS` stays
   flat, that confirms the kernel/driver leak.** If a process and `procRSS`
   climb together, that names the guilty app.
2. **Protects VSCode**: when `MemAvailable` drops below 1500 MiB (pre-OOM
   territory), it raises the OOM-kill priority on the largest non-essential
   process (typically Chrome) so the kernel sacrifices that instead of `code`.
   It only raises scores on your own processes and never touches its
   protect-list (`code`, `gnome-shell`, `Xorg`, `claude`, etc.).

## Install

```bash
./install.sh            # install, start, enable on boot
./install.sh uninstall  # stop and remove (keeps logs)
```

## Use

```bash
tail -f ~/.cache/mem-guard/mem-guard.log     # watch live
systemctl --user status mem-guard            # check it's running
systemctl --user restart mem-guard           # apply config changes
```

Tunables: edit the top of `mem-guard.sh`, or set env in `mem-guard.service`
(`MEM_GUARD_LOW_MIB`, `MEM_GUARD_INTERVAL`, `MEM_GUARD_LOG`).

## Optional stronger fix (needs sudo)

`earlyoom` kills the biggest hog before the kernel panics and can be told to
never touch VSCode. Run mem-guard alongside it for the diagnostic log.

```bash
sudo apt install -y earlyoom
sudo sed -i 's/^EARLYOOM_ARGS=.*/EARLYOOM_ARGS="-r 60 -m 5 -s 100 --avoid \x27(^|\/)(code|gnome-shell|Xorg)$\x27 --prefer \x27(^|\/)(chrome|node|ollama|python3)$\x27"/' /etc/default/earlyoom
sudo systemctl enable --now earlyoom
```
