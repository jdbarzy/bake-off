#!/usr/bin/env bash
# Bake-off runtime config. Copy to run.config.sh and edit for YOUR network.
#   cp run.config.example.sh run.config.sh
# run.config.sh is gitignored (it names your hosts); this example is committed.
# Nothing here is hardcoded in the app - change it freely to run on any network.

# --- Dashboard bind ---
DASH_HOST="0.0.0.0"     # 0.0.0.0 = reachable on your LAN; 127.0.0.1 = this machine only
DASH_PORT="15600"
DASH_AUTH=""          # set to "user:pass" to require a login (used when exposing publicly)

# --- HTTPS (optional) ---
DASH_TLS=""           # "self" = serve https with an auto-created certificate (browsers warn once
                      # per device; recommended if you set DASH_AUTH on a shared network). "" = http.
# Have real certificates? Point at them instead of DASH_TLS:
#   DASH_TLS_CERT="/path/cert.pem"
#   DASH_TLS_KEY="/path/key.pem"

# --- SSH tunnels to your model hosts (optional) ---
# One entry per tunnel: "LOCAL_PORT:REMOTE_HOST:REMOTE_PORT@SSH_TARGET"
#   LOCAL_PORT   - port opened on THIS machine (what the dashboard/config points at)
#   REMOTE_HOST  - host as seen FROM the ssh target (usually localhost)
#   REMOTE_PORT  - the model server's port on that box
#   SSH_TARGET   - a ~/.ssh/config alias, or user@host
# Leave TUNNELS empty () if your model endpoints are already reachable directly.
TUNNELS=(
  # "8000:localhost:8000@gpu-box-1"     # example: vLLM /v1 on ssh host 'gpu-box-1'
  # "11499:localhost:11499@gpu-box-1"   # its swap agent (vllm port + 3499)
  # "8000:localhost:8000@gpu-box-2"     # example: an OpenAI-compatible server on 'gpu-box-2'
)
