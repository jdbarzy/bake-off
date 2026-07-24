#!/usr/bin/env bash
# mwboot - make a GPU machine "eligible" for Bake-off, without Claude.
#
# Serving-layer provisioner, vLLM edition. On a candidate GPU box it:
#   1. detects the NVIDIA GPUs (type, count, VRAM),
#   2. installs vLLM into ~/vllm-env if it isn't there yet,
#   3. starts one mwboot-control swap agent PER GPU (control port = vllm port
#      + 3499) and asks each to serve a model sized to that GPU's VRAM,
#   4. prints the config to paste into Bake-off's "Warming up the models" page
#      (and writes it to mwboot-bakeoff.json).
#
# Every endpoint speaks the OpenAI /v1 API, and because the agent runs the
# server, Bake-off's model catalog can swap what each GPU serves later - no
# SSH needed.
#
# Assumes the NVIDIA driver + CUDA are already installed (check: nvidia-smi).
# It does NOT install drivers/CUDA - that's the "serving layer only" scope.
#
# Usage:
#   bash mwboot.sh                                  # detect + provision
#   MWBOOT_DRYRUN=1 bash mwboot.sh                  # show the plan, change nothing
#   MWBOOT_MODELS="Qwen/Qwen2.5-14B-Instruct-AWQ,Qwen/Qwen2.5-7B-Instruct" bash mwboot.sh
#   MWBOOT_BASE_PORT=8000 bash mwboot.sh            # first endpoint port (then +1 per GPU)
#   MWBOOT_VLLM_ARGS="--enforce-eager" bash mwboot.sh   # extra vllm serve flags
set -uo pipefail

BASE_PORT="${MWBOOT_BASE_PORT:-8000}"
DRYRUN="${MWBOOT_DRYRUN:-0}"
MAXLEN="${MWBOOT_MAX_LEN:-8192}"
OUT="mwboot-bakeoff.json"
HERE="$(cd "$(dirname "$0")" && pwd)"

c(){ printf '\033[1;36m[mwboot]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[mwboot] %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31m[mwboot] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------- 1. preflight ----------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Install the NVIDIA driver + CUDA first, then re-run."
command -v curl >/dev/null || die "curl is required."
command -v python3 >/dev/null || die "python3 is required."
[ -f "$HERE/mwboot-control.py" ] || die "mwboot-control.py must sit next to this script."

# ---------- 2. detect GPUs ----------
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -gt 0 ] || die "no NVIDIA GPUs detected."
c "Detected ${#GPU_ROWS[@]} GPU(s):"
for r in "${GPU_ROWS[@]}"; do printf '        GPU %s\n' "$r"; done

# ---------- 3. model pick by VRAM (open HF repos; override with MWBOOT_MODELS) ----------
pick_model(){   # $1 = VRAM in MiB
  local v="$1"
  if   [ "$v" -ge 90000 ]; then echo "openai/gpt-oss-120b"
  elif [ "$v" -ge 70000 ]; then echo "Qwen/Qwen3-32B"
  elif [ "$v" -ge 20000 ]; then echo "Qwen/Qwen2.5-14B-Instruct-AWQ"
  elif [ "$v" -ge 14000 ]; then echo "NousResearch/Meta-Llama-3.1-8B-Instruct"
  elif [ "$v" -ge  9000 ]; then echo "Qwen/Qwen2.5-7B-Instruct"
  else                          echo "Qwen/Qwen2.5-3B-Instruct"
  fi
}
IFS=',' read -r -a OVERRIDE <<< "${MWBOOT_MODELS:-}"

# ---------- 4. install vLLM if missing ----------
VENVPY="$HOME/vllm-env/bin/python3"
if ! "$VENVPY" -c "import vllm" >/dev/null 2>&1; then
  if [ "$DRYRUN" = 1 ]; then warn "(dry-run) would install vLLM into ~/vllm-env"
  else
    c "Installing vLLM into ~/vllm-env (first time takes a few minutes)..."
    python3 -m venv "$HOME/vllm-env" || die "python3 -m venv failed (install python3-venv)."
    "$HOME/vllm-env/bin/pip" install -q -U pip || die "pip upgrade failed."
    "$HOME/vllm-env/bin/pip" install -q vllm || die "vLLM install failed."
  fi
fi

# ---------- 5. one swap agent + vLLM endpoint per GPU (sequential: concurrent first
# starts deadlock on shared torch.compile/JIT caches) ----------
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); IP="${IP:-127.0.0.1}"
declare -a PORTS MODELS
i=0
for r in "${GPU_ROWS[@]}"; do
  idx=$(awk -F',' '{gsub(/ /,"",$1);print $1}' <<< "$r")
  name=$(awk -F',' '{sub(/^ */,"",$2);print $2}' <<< "$r")
  vram=$(awk -F',' '{gsub(/ /,"",$3);print $3}' <<< "$r")
  port=$((BASE_PORT + i))
  ctrl=$((port + 3499))
  model="${OVERRIDE[i]:-$(pick_model "$vram")}"
  c "GPU $idx ($name, ${vram}MiB) -> vLLM on :$port serving '$model' (agent on :$ctrl)"
  if [ "$DRYRUN" != 1 ]; then
    MWBOOT_VLLM_PYTHON="$VENVPY" MWBOOT_VLLM_ARGS="${MWBOOT_VLLM_ARGS:---enforce-eager}" \
      nohup python3 "$HERE/mwboot-control.py" --port "$ctrl" --vllm-port "$port" \
        --gpu "$idx" --max-len "$MAXLEN" >"/tmp/mwboot-agent-$ctrl.log" 2>&1 &
    sleep 2
    curl -s -m5 "http://127.0.0.1:$ctrl/control/load" \
      -d "{\"model\":\"$model\"}" >/dev/null || warn "load request failed on :$ctrl"
    c "  waiting for '$model' on :$port (first run downloads the weights)..."
    for _ in $(seq 1 360); do
      st=$(curl -s -m3 "http://127.0.0.1:$ctrl/control/status" 2>/dev/null || true)
      grep -q '"ready": *true' <<< "$st" && { c "  UP on :$port"; break; }
      grep -q '"error": *"' <<< "$st" && { warn "  agent reports: $st"; break; }
      sleep 5
    done
  fi
  PORTS+=("$port"); MODELS+=("$model"); i=$((i+1))
done

# ---------- 6. detect an existing hot-swap gateway (optional) ----------
GATEWAY=""
if [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)" = "200" ] \
   && [ "$BASE_PORT" != 8000 ]; then
  GATEWAY="http://$IP:8000/v1"; c "Hot-swap gateway detected on :8000."
fi

# ---------- 7. emit Bake-off config (fills the 3 slots, extras listed as spares) ----------
c "This machine is now Bake-off-eligible. Endpoints (bound on all interfaces):"
for j in "${!PORTS[@]}"; do echo "        http://$IP:${PORTS[$j]}/v1   (vLLM, ${MODELS[$j]})"; done

python3 - "$IP" "${PORTS[*]}" "${MODELS[*]}" "$OUT" <<'PY'
import sys, json
ip, ports, models, out = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4]
keys = ["slot-a", "slot-b", "slot-c"]   # Bake-off's 3 fixed slot ids
slots = [{"key": keys[i], "label": m, "system": ip, "kind": "openai",
          "base": f"http://{ip}:{p}/v1", "model": m}
         for i, (p, m) in enumerate(zip(ports, models)) if i < 3]
json.dump({"models": slots}, open(out, "w"), indent=2)
print("\n" + open(out).read())
PY

echo
c "Next: from the machine running Bake-off, either"
c "  a) paste the block above into the Warming-up page (gear -> Edit endpoints), or"
c "  b) copy $OUT to  dashboard/models.config.json  and restart the dashboard."
[ "${#PORTS[@]}" -gt 3 ] && warn "You have ${#PORTS[@]} GPUs; Bake-off compares 3 at a time - the extra endpoints are ready as spares."
[ -n "$GATEWAY" ] && c "Hot-swap: point a slot at $GATEWAY to use the gateway instead."
