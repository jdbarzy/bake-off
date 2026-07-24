#!/usr/bin/env bash
# Confirm the three configured model endpoints respond (run on the control machine).
# Reads dashboard/models.config.json when present, else the built-in defaults.
set -uo pipefail
cd "$(dirname "$0")"

python3 - <<'EOF'
import json, os, urllib.request

cfg = os.path.join("dashboard", "models.config.json")
models = None
if os.path.exists(cfg):
    try:
        models = json.load(open(cfg)).get("models")
    except Exception:
        pass
if not models:
    models = [{"label": "default (vLLM on this machine)", "kind": "openai",
               "base": "http://localhost:8000/v1"}]

seen = set()
ok = True
for m in models:
    base = (m.get("base") or "").rstrip("/")
    if not base or base in seen:
        continue
    seen.add(base)
    url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    label = m.get("system") or m.get("label") or base
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            r.read(200)
        print(f"OK    {label:30} {base}")
    except Exception as e:
        print(f"DOWN  {label:30} {base}   ({e})")
        ok = False
print()
print("All endpoints answered." if ok else "Some endpoints are not reachable - start that model server (or its tunnel) and re-run.")
EOF
