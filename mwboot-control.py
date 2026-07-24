#!/usr/bin/env python3
"""mwboot-control - a tiny model-swap control plane for a vLLM box.

vLLM serves ONE model per process and has no runtime API to pull/swap a base model,
so the only way to "load a different model" is to restart it. This little service
does exactly that on request, so llm-bench (or curl) can change the served model
without SSH:

    POST /control/load   {"model": "Qwen/Qwen2.5-7B-Instruct-AWQ"}
        -> stop the current vLLM, start `vllm serve <model>` (downloads from
           HuggingFace on first use), report progress via /control/status
    GET  /control/status
        -> {"loading": bool, "ready": bool, "model": str, "error": str|null}

Run it on the GPU box (assumes vLLM installed + NVIDIA driver/CUDA present):
    python3 mwboot-control.py --port 11499 --vllm-port 8000 --gpu 0
(convention: control port = vllm port + 3499, one agent per GPU/instance)
Then point llm-bench's endpoint at http://<box>:8000/v1 ; it finds the control
plane at http://<box>:11499 automatically. --dry-run simulates a load (no vLLM)
so you can test the wiring on a machine without a GPU.
"""
import argparse
import hmac
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"loading": False, "ready": False, "model": "", "error": None, "since": 0.0}

_VLLM_PY = {"path": None}
_ARCHS = {"list": None, "started": False}   # this install's vLLM-supported architectures, probed once


def _probe_archs():
    """Ask the local vLLM which model architectures it can boot. Runs once, in the
    background (the import is heavy); until it lands, /control/status just omits
    'archs' and the dashboard falls back to its family heuristic."""
    try:
        out = subprocess.run(
            [_vllm_python(), "-c",
             "import json\n"
             "from vllm.model_executor.models.registry import ModelRegistry\n"
             "print(json.dumps(sorted(ModelRegistry.get_supported_archs())))"],
            capture_output=True, timeout=180)
        if out.returncode == 0:
            _ARCHS["list"] = json.loads(out.stdout.decode().strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - no vllm here means no archs, which is honest
        pass


def _vllm_python():
    """A python that can actually import vllm. vLLM usually lives in a venv, not the
    system python, so try: MWBOOT_VLLM_PYTHON (explicit), this interpreter, then the
    conventional ~/vllm-env venv. Cached after the first successful probe."""
    if _VLLM_PY["path"]:
        return _VLLM_PY["path"]
    for p in (os.environ.get("MWBOOT_VLLM_PYTHON"), sys.executable,
              os.path.expanduser("~/vllm-env/bin/python3")):
        if not p or not os.path.exists(p):
            continue
        try:
            subprocess.run([p, "-c", "import vllm"], check=True, capture_output=True, timeout=120)
            _VLLM_PY["path"] = p
            return p
        except Exception:  # noqa: BLE001
            continue
    return "python3"   # last resort: PATH decides (the old behavior)

_HUB_CACHE = {"at": 0.0, "models": [], "sizes": {}}


def _cached_models():
    """HF repo ids with weights already in this box's cache - swaps to these are
    minutes, not downloads. The picker prefers them; the shelf shows them as fast.
    The same scan measures each repo's real bytes on disk (symlinks counted once)."""
    now = time.time()
    if now - _HUB_CACHE["at"] < 60:
        return _HUB_CACHE["models"]
    out, sizes = [], {}
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE") or "~/.cache/huggingface/hub")
    try:
        for name in os.listdir(hub):
            if not name.startswith("models--"):
                continue
            snaps = os.path.join(hub, name, "snapshots")
            if os.path.isdir(snaps) and os.listdir(snaps):
                mid = name[len("models--"):].replace("--", "/")
                b = _repo_dir_bytes(mid)
                if b < 100_000_000:
                    continue   # config-only stub, no weights: "cached" would promise a fast seat that's really a download
                out.append(mid)
                sizes[mid] = b
    except Exception:  # noqa: BLE001
        pass
    for m in DRY_CACHED:   # dry-run: simulated prefetches count
        if m not in out:
            out.append(m)
            sizes[m] = 1_000_000_000
    _HUB_CACHE.update(at=now, models=sorted(out), sizes=sizes)
    return _HUB_CACHE["models"]


def _cached_sizes():
    _cached_models()   # refresh if stale
    return _HUB_CACHE["sizes"]


_VRAM = {"mb": -1}


def _gpu_vram_mb():
    """This agent's GPU memory in MiB (the GPU it manages), asked once. Lets the
    dashboard fit-check models on machines that never enrolled with their VRAM."""
    if _VRAM["mb"] >= 0:
        return _VRAM["mb"]
    mb = 24000 if CFG.get("dry_run") else 0
    if not CFG.get("dry_run"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
            for ln in out.stdout.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 2 and parts[0] == str(CFG.get("gpu", "0")):
                    try:
                        mb = int(float(parts[1]))
                    except ValueError:
                        mb = 0   # unified-memory GPUs report "[N/A]"
        except Exception:  # noqa: BLE001
            pass
        if not mb:
            try:   # unified memory (GB10, DGX Spark class): system RAM IS the GPU memory
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for ln in f:
                        if ln.startswith("MemTotal:"):
                            mb = int(ln.split()[1]) // 1024
                            break
            except Exception:  # noqa: BLE001
                pass
    _VRAM["mb"] = mb
    return mb


_MY_GPU = {"name": None}


def _my_gpu():
    """The device THIS agent manages (its assigned --gpu), verbatim from nvidia-smi -
    never typed by a human. Handles plain indices, and falls back to matching UUID /
    MIG identifiers against nvidia-smi -L so slices and vGPU profiles read as what
    they are. One box, many agents -> each card shows its own silicon."""
    if _MY_GPU["name"] is not None:
        return _MY_GPU["name"]
    name = "DRY GPU" if CFG.get("dry_run") else ""
    g = str(CFG.get("gpu", "0"))
    if not name:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            for ln in out.stdout.splitlines():
                parts = [x.strip() for x in ln.split(",", 1)]
                if len(parts) == 2 and parts[0] == g:
                    name = parts[1]
                    break
        except Exception:  # noqa: BLE001
            pass
    if not name and not CFG.get("dry_run"):
        try:   # MIG slice or UUID assignment: find the -L line that names it
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            for ln in out.stdout.splitlines():
                if g and g in ln:
                    name = ln.split(":", 1)[-1].split("(")[0].strip()
                    break
        except Exception:  # noqa: BLE001
            pass
    _MY_GPU["name"] = name
    return name


_GPU_CACHE = {"at": 0.0, "gpus": []}


def _gpus():
    """The hardware truth for this box: GPU names via nvidia-smi or rocm-smi, cached 60s.
    llm-bench shows these as hardware chips - agent-reported only, never hand-typed."""
    now = time.time()
    if now - _GPU_CACHE["at"] < 60:
        return _GPU_CACHE["gpus"]
    names = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        pass
    if not names:
        try:
            out = subprocess.run(["rocm-smi", "--showproductname", "--json"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                data = json.loads(out.stdout or "{}")
                for v in data.values():
                    n = v.get("Card Series") or v.get("Card series") or v.get("Card model")
                    if n:
                        names.append(str(n).strip())
        except Exception:  # noqa: BLE001
            pass
    gpus = []
    for n in sorted(set(names)):
        c = names.count(n)
        gpus.append(f"{n} x{c}" if c > 1 else n)
    _GPU_CACHE.update(at=now, gpus=gpus)
    return gpus
LOCK = threading.Lock()
PROC = {"p": None}
CFG = {}


def _vllm_up():
    """True once this box's vLLM answers /v1/models."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{CFG['vllm_port']}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _stop_current():
    p = PROC["p"]
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        for _ in range(20):
            if p.poll() is not None:
                break
            time.sleep(0.5)
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    PROC["p"] = None


def _free_port(port):
    """Kill whatever still holds the serve port - an orphan vLLM from a previous agent
    life, or a manually started server. Kills the whole process group so the renamed
    VLLM::EngineCore children release their GPU memory too (they escape name-based
    pkill). The agent owns this port by contract, so anything on it is stale."""
    try:
        out = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=10)
        pids = {int(m) for m in re.findall(r"pid=(\d+)", out.stdout)}
    except Exception:  # noqa: BLE001
        pids = set()
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    if pids:
        time.sleep(3)   # let the GPU memory actually come back before restarting


LOADGEN = {"n": 0}   # bumps every time a newer intent supersedes an in-flight load


def _load_worker(model, gen):
    def stale():
        return gen != LOADGEN["n"]   # a newer pick took over; this worker exits silently
    mine = None
    try:
        _stop_current()
        if CFG["dry_run"]:
            for _ in range(4):
                if stale():
                    return
                time.sleep(0.5)
        else:
            _free_port(CFG["vllm_port"])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(CFG["gpu"]))
            # vLLM subprocesses resolve build tools (ninja for JIT kernels) via PATH, not
            # via the python that spawned them - the venv's bin must lead PATH or models
            # that compile custom ops die with FileNotFoundError: 'ninja' (field finding 3)
            env["PATH"] = os.path.dirname(_vllm_python()) + os.pathsep + env.get("PATH", "")
            cmd = [_vllm_python(), "-m", "vllm.entrypoints.openai.api_server",
                   "--model", model, "--served-model-name", model,
                   "--host", "0.0.0.0", "--port", str(CFG["vllm_port"]),
                   "--gpu-memory-utilization", str(CFG["gpu_mem"]),
                   # many current VLMs (Kimi-VL, MiniCPM-V, InternVL, Molmo...) ship custom
                   # model code vLLM refuses to load without this. The user picks what to
                   # serve, so trust it - safe for models that don't use remote code.
                   "--trust-remote-code"]
            if CFG["max_len"]:
                cmd += ["--max-model-len", str(CFG["max_len"])]
            cmd += os.environ.get("MWBOOT_VLLM_ARGS", "").split()   # e.g. --enforce-eager --trust-remote-code
            if stale():
                return
            log = open(f"/tmp/mwboot-vllm-{CFG['vllm_port']}.log", "ab")
            mine = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
            PROC["p"] = mine
            deadline = time.time() + CFG["timeout"]
            while time.time() < deadline:
                if stale():
                    return   # superseded: the new worker's _stop_current already reaps our vLLM
                if mine.poll() is not None:
                    raise RuntimeError(f"vLLM exited early (see /tmp/mwboot-vllm-{CFG['vllm_port']}.log)")
                if _vllm_up():
                    break
                time.sleep(2)
            else:
                raise TimeoutError("vLLM did not come up in time")
        with LOCK:
            if not stale():
                STATE.update(loading=False, ready=True, model=model, error=None)
    except Exception as e:  # noqa: BLE001
        with LOCK:
            if not stale():
                STATE.update(loading=False, ready=False, error=str(e)[:200])


def start_load(model):
    with LOCK:
        if STATE["loading"]:
            if STATE["model"] == model:
                return True   # same intent already in flight: attach, don't restart
            LOADGEN["n"] += 1   # newest pick wins: cancel the in-flight load, load this instead
        elif STATE["ready"] and STATE["model"] == model:
            return True   # idempotent: re-asserting the served model is free, no restart
        gen = LOADGEN["n"]
        STATE.update(loading=True, ready=False, model=model, error=None, since=time.time())
    threading.Thread(target=_load_worker, args=(model, gen), daemon=True).start()
    return True


def unload():
    """Evacuate: stop the vLLM this agent owns (and anything else squatting on the port),
    freeing the GPU. Cancels an in-flight load too - evacuation is the newest intent."""
    with LOCK:
        if STATE["loading"]:
            LOADGEN["n"] += 1   # supersede: the loading worker exits silently
        STATE.update(loading=False, ready=False, model="", error=None, since=time.time())
    _stop_current()
    if not CFG.get("dry_run"):
        _free_port(CFG["vllm_port"])
    return True


PREFETCH = {"model": "", "running": False, "done": False, "error": None, "p": None,
            "bytes": 0, "total": 0, "rate": 0, "phase": ""}


DRY_CACHED = []   # dry-run only: models a simulated prefetch "downloaded"


def _repo_dir_bytes(model):
    """Bytes of this repo already on disk (including in-flight .incomplete blobs) -
    the honest progress signal, straight from the filesystem."""
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE") or "~/.cache/huggingface/hub")
    d = os.path.join(hub, "models--" + model.replace("/", "--"))
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):   # snapshots/ symlinks the blobs - count bytes once
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _prefetch_worker(model, total_bytes=0):
    """Pull a model's weights into this box's Hugging Face cache WITHOUT touching the
    running server, reporting live progress (bytes landed, rate, phase)."""
    if CFG.get("dry_run"):   # simulate a download with visible progress
        fake_total = total_bytes or 1_000_000_000
        for i in range(1, 5):
            time.sleep(0.5)
            with LOCK:
                PREFETCH.update(bytes=int(fake_total * i / 4), total=fake_total,
                                rate=int(fake_total / 2), phase="downloading")
        with LOCK:
            DRY_CACHED.append(model)
            PREFETCH.update(running=False, done=True, error=None, phase="done")
            _HUB_CACHE["at"] = 0.0   # the cached-models list changed; re-scan on next ask
        return
    STALL_SECS = 120   # no bytes landing for this long = a wedged resume; kill and retry
    try:
        py = _vllm_python()
        # skip duplicate weight folders (original/, metal/, onnx/...) and formats vLLM never
        # reads - a repo like gpt-oss-20b is 14 GB of usable weights inside an 88 GB repo
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                "snapshot_download(sys.argv[1], ignore_patterns=["
                "'*/*.safetensors','*/*.bin','*/*.pt','*/*.pth','*.gguf','*.onnx','*.pt','*.pth'])\n")
        p, err = None, b""
        for attempt in range(3):
            p = subprocess.Popen([py, "-c", code, model],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 start_new_session=True)
            PREFETCH["p"] = p
            last = _repo_dir_bytes(model)
            grew_at, wedged = time.time(), False
            win = [(time.time(), last)]   # ~25 s window: smoothed rate, no flicker-to-zero between bursts
            while p.poll() is None:   # sizer loop: progress = what actually landed on disk
                time.sleep(1.5)
                b, now = _repo_dir_bytes(model), time.time()
                if b > last:
                    grew_at = now
                elif now - grew_at > STALL_SECS:
                    wedged = True
                    break
                win.append((now, b))
                win[:] = [(t, x) for (t, x) in win if now - t <= 25]
                rate = int((b - win[0][1]) / max(0.5, now - win[0][0]))
                with LOCK:
                    PREFETCH.update(bytes=b, rate=max(0, rate),
                                    phase="finishing" if (total_bytes and b >= total_bytes * 0.98) else "downloading")
                last = b
            if not wedged:
                break
            try:   # stalled resume (a known HF failure mode): kill the whole group, go again
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            with LOCK:
                PREFETCH.update(phase="retrying", rate=0)
        else:
            with LOCK:
                PREFETCH.update(running=False, done=True, phase="error",
                                error="download kept stalling - check the box's connection to huggingface.co, then add the model again")
            return
        err = p.stderr.read() if p.stderr else b""
        with LOCK:
            if p.returncode == 0:
                PREFETCH.update(running=False, done=True, error=None, phase="done",
                                bytes=_repo_dir_bytes(model))
                _HUB_CACHE["at"] = 0.0   # the cached-models list changed; re-scan on next ask
            else:
                tail = (err or b"").decode("utf-8", "replace").strip().splitlines()
                PREFETCH.update(running=False, done=True, phase="error",
                                error=(tail[-1] if tail else "download failed")[:200])
    except Exception as e:  # noqa: BLE001
        with LOCK:
            PREFETCH.update(running=False, done=True, error=str(e)[:200], phase="error")


def start_prefetch(model, total_bytes=0):
    with LOCK:
        if PREFETCH["running"]:
            return PREFETCH["model"] == model   # same model already on its way: fine
        PREFETCH.update(model=model, running=True, done=False, error=None,
                        bytes=0, total=int(total_bytes or 0), rate=0, phase="starting")
    threading.Thread(target=_prefetch_worker, args=(model, int(total_bytes or 0)), daemon=True).start()
    return True


def _adopt_running():
    """On agent (re)start: if something already answers on our vLLM port, adopt it instead
    of reporting it missing. An agent restart must never make a healthy server look dead."""
    try:
        data = _http_get_json(f"http://127.0.0.1:{CFG['vllm_port']}/v1/models")
        model = (data.get("data") or [{}])[0].get("id") or ""
        if model:
            with LOCK:
                STATE.update(loading=False, ready=True, model=model, error=None, since=time.time())
            return model
    except Exception:  # noqa: BLE001
        pass
    return ""


def _http_get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _keyed(self):
        """Mutating endpoints require the control key when one is configured (MWBOOT_KEY).
        A keyless agent stays open - legacy lab mode. /control/status is always open."""
        want = CFG.get("key") or ""
        if not want:
            return True
        got = self.headers.get("X-MWBoot-Key", "")
        if got and hmac.compare_digest(got, want):
            return True
        self._send({"ok": False, "error": "control key missing or wrong - this box only "
                    "accepts changes from the bake-off that enrolled it"}, 401)
        return False

    def do_GET(self):
        if self.path.split("?")[0] == "/control/status":
            if not _ARCHS["started"]:
                _ARCHS["started"] = True
                threading.Thread(target=_probe_archs, daemon=True).start()
            with LOCK:
                snap = dict(STATE)
                pf = {k: PREFETCH[k] for k in ("model", "running", "done", "error", "bytes", "total", "rate", "phase")}
            extra = {"archs": _ARCHS["list"]} if _ARCHS["list"] else {}
            return self._send({**snap, **extra, "gpus": _gpus(), "gpu": _my_gpu(), "vram_mb": _gpu_vram_mb(), "cached": _cached_models(), "cached_sizes": _cached_sizes(), "prefetch": pf})
        return self._send({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/control/load":
            if not self._keyed():
                return
            ln = int(self.headers.get("Content-Length", "0"))
            opts = json.loads(self.rfile.read(ln) or b"{}") if ln else {}
            model = str(opts.get("model") or "").strip()
            if not model:
                return self._send({"ok": False, "error": "model is required"}, 400)
            ok = start_load(model)
            return self._send({"ok": ok, "error": None if ok else "a model is already loading"})
        if self.path == "/control/unload":
            if not self._keyed():
                return
            ok = unload()
            return self._send({"ok": ok, "error": None if ok else "a model is loading right now - try again when it settles"})
        if self.path == "/control/delete":
            if not self._keyed():
                return
            ln = int(self.headers.get("Content-Length", "0"))
            opts = json.loads(self.rfile.read(ln) or b"{}") if ln else {}
            model = str(opts.get("model") or "").strip()
            if not model:
                return self._send({"ok": False, "error": "model is required"}, 400)
            with LOCK:
                if (STATE["ready"] or STATE["loading"]) and STATE["model"] == model:
                    return self._send({"ok": False, "error": "that model is serving on this machine - swap or evacuate first"}, 409)
                if PREFETCH["running"] and PREFETCH["model"] == model:
                    return self._send({"ok": False, "error": "that model is downloading right now - let it finish first"}, 409)
            freed = _repo_dir_bytes(model)
            hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE") or "~/.cache/huggingface/hub")
            shutil.rmtree(os.path.join(hub, "models--" + model.replace("/", "--")), ignore_errors=True)
            with LOCK:
                if model in DRY_CACHED:
                    DRY_CACHED.remove(model)
                _HUB_CACHE["at"] = 0.0   # the cached list just changed
            return self._send({"ok": True, "freed": freed})
        if self.path == "/control/prefetch":
            if not self._keyed():
                return
            ln = int(self.headers.get("Content-Length", "0"))
            opts = json.loads(self.rfile.read(ln) or b"{}") if ln else {}
            model = str(opts.get("model") or "").strip()
            if not model:
                return self._send({"ok": False, "error": "model is required"}, 400)
            ok = start_prefetch(model, opts.get("total_bytes") or 0)
            return self._send({"ok": ok, "error": None if ok else f"already downloading {PREFETCH['model']} - one at a time"})
        return self._send({"error": "not found"}, 404)

    def log_message(self, *a):   # quiet
        pass


def main():
    ap = argparse.ArgumentParser(description="model-swap control plane for a vLLM box")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MWBOOT_CONTROL_PORT", 11499)))
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="simulate loads (no vLLM) to test the wiring")
    a = ap.parse_args()
    CFG.update(vllm_port=a.vllm_port, gpu=a.gpu, gpu_mem=a.gpu_mem, max_len=a.max_len,
               timeout=a.timeout, dry_run=a.dry_run,
               key=os.environ.get("MWBOOT_KEY", "").strip())   # set -> mutating calls need X-MWBoot-Key
    adopted = "" if a.dry_run else _adopt_running()
    print(f"mwboot-control on :{a.port} -> manages vLLM on :{a.vllm_port} (gpu {a.gpu})"
          f"{' [dry-run]' if a.dry_run else ''}{' [keyed]' if CFG['key'] else ''}"
          f"{f' [adopted {adopted}]' if adopted else ''}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
