#!/usr/bin/env python3
"""llm-bench live dashboard (port 15600).

Serves a wall-display status board and drives a continuous load loop: it cycles
an uploaded CSV of questions through all three models, grading each answer and
accumulating per-model pass rate / tokens / throughput so the board updates live.

Endpoints:
  GET  /                -> index.html
  GET  /run/stats       -> live per-model counters (JSON)
  GET  /run/rows        -> current CSV rows + columns
  POST /run/upload      -> body = raw CSV text; replaces the question set
  POST /run/start       -> body = {"max_tokens": int}; (re)start the loop
  POST /run/stop        -> stop the loop
"""
import base64
import csv
import difflib
import hmac
import io
import json
import os
import re
import shlex
import shutil
import sys
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import http.server
import socketserver

HOST = os.environ.get("DASH_HOST", "127.0.0.1")  # localhost by default; set DASH_HOST=0.0.0.0 to expose on the LAN
# Optional HTTP Basic Auth. Set DASH_AUTH="user:pass" to require a browser login on
# every request (use this when exposing the dashboard publicly, e.g. a Cloudflare tunnel).
_AUTH = os.environ.get("DASH_AUTH", "").strip()
_AUTH_EXPECT = ("Basic " + base64.b64encode(_AUTH.encode()).decode()) if _AUTH else None
PORT = int(os.environ.get("DASH_PORT", "15600"))
HERE = os.path.dirname(os.path.abspath(__file__))

csv.field_size_limit(10_000_000)  # some prompt datasets have huge multi-paragraph fields


# ---- shared JSON persistence shape: best-effort save, tolerant keyed load ----
def _jdump(path, obj, indent=2):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent)
    except Exception:  # noqa: BLE001
        pass


def _jload(path, key, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(key) or default
    except Exception:  # noqa: BLE001
        return default

# The three benchmark slots. llm-bench speaks the OpenAI /v1 API only (vLLM,
# TensorRT-LLM, SGLang, NIM, llama-server, LM Studio, a gateway). Out of the box the
# slots are empty - add your serving machines in Settings (or the warm-up editor) and
# each slot fills from a machine's live model list. Persisted to models.config.json,
# which overrides these defaults.
DEFAULT_MODELS = [
    {"key": "slot-a", "label": "", "system": "", "kind": "openai",
     "base": "", "model": "", "color": "#e5944b", "params_b": 0, "empty": True},
    {"key": "slot-b", "label": "", "system": "", "kind": "openai",
     "base": "", "model": "", "color": "#2ec27e", "params_b": 0, "empty": True},
    {"key": "slot-c", "label": "", "system": "", "kind": "openai",
     "base": "", "model": "", "color": "#6ea8fe", "params_b": 0, "empty": True},
]

# Optional hot-swap gateway (an OpenAI-compatible router that can swap the model behind
# a slot). Configured via gitignored gateway.json {base, bearer}; absent = feature off.
GATEWAY_FILE = os.path.join(HERE, "gateway.json")
GATEWAY_SLOTS = {m["key"]: f"slot-{i}" for i, m in enumerate(DEFAULT_MODELS)}   # dashboard slot key -> gateway slot


def _load_gateway():
    try:
        with open(GATEWAY_FILE) as f:
            g = json.load(f)
        if g.get("base") and g.get("bearer"):
            return g
    except Exception:  # noqa: BLE001
        pass
    return None


GATEWAY = _load_gateway()


def _is_gateway(m):
    return bool(GATEWAY) and m.get("base") == GATEWAY["base"] and m.get("key") in GATEWAY_SLOTS


def _gw_request(path, method="GET", admin_base=False, params=None, timeout=8):
    """Call the hot-swap gateway. admin routes use X-Admin-Key; /v1 uses Bearer."""
    if not GATEWAY:
        return None
    base = GATEWAY["admin_base"] if admin_base else GATEWAY["base"]
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"X-Admin-Key": GATEWAY["admin_key"]} if admin_base else {"Authorization": "Bearer " + GATEWAY["bearer"]}
    req = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_gw_cache = {"at": 0.0, "specs": {}}   # {variant_name: {base, quant, adapters}}, refreshed live from the gateway


def gateway_specs():
    """Live per-variant details (base model, quantization, LoRA adapters) from the
    gateway admin API, cached ~30s. Source of truth for what's really on the GPU."""
    now = time.time()
    if GATEWAY and now - _gw_cache["at"] > 30:
        try:
            data = _gw_request("/variants", admin_base=True)
            specs = {v["name"]: {"base": v.get("model", ""), "quant": v.get("quantization", ""),
                                 "adapters": v.get("adapters") or []}
                     for v in (data or {}).get("variants", [])}
            if specs:
                _gw_cache.update(at=now, specs=specs)
        except Exception:  # noqa: BLE001
            pass
    return _gw_cache["specs"]


def gateway_variants():
    """Live catalog of swappable variant names (order preserved)."""
    return list(gateway_specs().keys())


def gateway_ensure(model_name, mw_slot):
    """Make `model_name` the AWAKE variant on `mw_slot`. Skips the swap if it's
    already armed there; otherwise swaps (synchronous, can take ~60s cold)."""
    try:
        data = _gw_request("/variants", admin_base=True)
        for v in (data or {}).get("variants", []):
            if v.get("name") == model_name and v.get("state") == "AWAKE" and v.get("slot") == mw_slot:
                return True   # already armed - don't churn the slot
        _gw_request(f"/slots/{mw_slot}/swap", method="POST", admin_base=True,
                    params={"to": model_name}, timeout=180)
        return True
    except Exception:  # noqa: BLE001
        return False

def _http_json(url, timeout=6, key=None):
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def probe_endpoint(kind, base, key=None):
    """Ask an endpoint what models it serves, so a business analyst can adopt their
    own machine without knowing model names or VRAM. Returns the model list and
    whether it is the hot-swap gateway. Read-only - lists models, changes nothing.
    `key` is the machine's API key for hosted endpoints (a NIM, a cloud /v1)."""
    base = (base or "").strip().rstrip("/")
    # configured hot-swap gateway: its /v1 needs a bearer, so list the swappable
    # variants through the authenticated admin API instead of an anonymous GET.
    if GATEWAY and base == GATEWAY["base"].rstrip("/"):
        try:
            variants = gateway_variants()
            if variants:
                return {"ok": True, "kind": "openai", "models": variants, "gateway": True}
        except Exception:  # noqa: BLE001
            pass
    # Everything llm-bench talks to serves the OpenAI /v1 API (vLLM, TensorRT-LLM,
    # SGLang, NIM, llama-server, LM Studio, a gateway). Accept a base with or
    # without the /v1 suffix.
    try:
        url = base + ("/models" if base.endswith("/v1") else "/v1/models")
        data = _http_json(url, key=key)
        models = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        gateway = bool(GATEWAY and base == GATEWAY["base"].rstrip("/"))
        return {"ok": True, "kind": "openai", "models": models, "gateway": gateway}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:140] or "no response"}


_think_cache = {}   # (kind, base, model) -> bool: can this model reason step by step?
REASONING_HEADROOM = 2048   # extra output tokens a thinking model gets so it can finish thinking AND answer


def model_can_think(m):
    """True if the model can produce step-by-step reasoning. OpenAI-compatible
    endpoints expose no capability API, so this is a name heuristic. Cached per model."""
    ck = (m.get("kind"), m.get("base"), m.get("model"))
    if ck in _think_cache:
        return _think_cache[ck]
    name = (m.get("model") or "").lower()
    can = any(t in name for t in ("qwen3", "-r1", "deepseek-r1", "o1", "o3", "magistral",
                                  "qwq", "reasoner", "gpt-oss"))
    _think_cache[ck] = can
    return can


_see_cache = {}   # (kind, base, model) -> bool: can this model read images?
# name fragments that mark a vision/VLM model on OpenAI-compatible endpoints (no capability API there)
_VLM_NAMES = ("vl", "vision", "llava", "pixtral", "minicpm-v", "gemma3", "gemma-3", "internvl",
              "phi-3.5-v", "phi-3-v", "phi3.5-v", "nemotron-nano-vl", "molmo", "idefics", "smolvlm", "paligemma")


def model_can_see(m):
    """True if the model accepts images (a vision / VLM model). The library and the
    trending catalog know from HF metadata; only models unknown to both fall back to
    a name check - OpenAI-compatible endpoints expose no capability API. Cached per model."""
    mid = m.get("model")
    lib = next((x for x in LIBRARY if x.get("id") == mid), None)
    if lib is not None:
        return bool(lib.get("vision"))   # the library knows (from HF metadata) - beats name guessing
    if any(e.get("id") == mid and e.get("vision") for e in TRENDING.get("entries") or []):
        return True   # the trending refresh knows (HF pipeline tag). A trending entry
        # without the flag still falls through: text-pipeline listings can miss VLMs.
    ck = (m.get("kind"), m.get("base"), m.get("model"))
    if ck in _see_cache:
        return _see_cache[ck]
    can = False
    try:
        name = (m.get("model") or "").lower()
        parts = re.split(r"[^a-z0-9.]+", name)
        can = any(t in name for t in _VLM_NAMES if len(t) > 2) or "vl" in parts
    except Exception:  # noqa: BLE001
        can = False
    _see_cache[ck] = can
    return can


# Fields a user may override per slot via the warm-up "Edit endpoints" form
# (persisted to models.config.json). key / color / desc stay fixed.
EDITABLE_FIELDS = ("label", "system", "kind", "base", "model", "machine")
MODELS_CONFIG_FILE = os.path.join(HERE, "models.config.json")


def _active_models():
    """Slots that actually have a machine + model. Empty slots are legal: a lab with
    one or two machines still races - the run and results just use what exists."""
    return [m for m in MODELS if m.get("model") and m.get("base")]


# ---- lineup protection: vision models race each other on document tasks; text models
# race each other on Q&A. A lineup never mixes the two, and a task never runs on the
# wrong kind - enforced here so the UI checks can't be bypassed. ----

def _lineup_split():
    """Names of the filled slots, split by capability: (vision models, text models)."""
    vis, txt = [], []
    for m in _active_models():
        (vis if model_can_see(m) else txt).append(m.get("model") or m.get("label") or m["key"])
    return vis, txt


def _task_lineup_error(doc):
    """User-facing error if the task kind doesn't fit the current lineup, else None.
    doc=True for document ingestion (page images), False for Q&A / text sets."""
    vis, txt = _lineup_split()
    if not (vis or txt):
        return None
    if doc and txt:
        return (", ".join(txt) + " can't read images. Document tasks need a vision model in "
                "every slot - swap those slots to vision models in Settings.")
    if not doc and vis:
        return (", ".join(vis) + (" is a vision model" if len(vis) == 1 else " are vision models")
                + ". Vision models only play other vision models, on document tasks - "
                "to run Q&A, swap the lineup to text models.")
    return None


def _slot_mix_error(slot_key, kind, base, model):
    """User-facing error if putting `model` into slot_key would mix vision and text
    models in one lineup, else None."""
    new_see = model_can_see({"kind": kind, "base": base, "model": model})
    clash = [m.get("model") or m.get("label") or m["key"] for m in _active_models()
             if m["key"] != slot_key and model_can_see(m) != new_see]
    if not clash:
        return None
    if new_see:
        return (f"{model} reads images, but {', '.join(clash)} can't. Vision models only play "
                "other vision models - empty or swap those slots first.")
    return (f"{model} is a text model, but {', '.join(clash)} read{'s' if len(clash) == 1 else ''} "
            "images. Vision models only play other vision models - empty or swap those slots first.")


def _load_models():
    """Built-in defaults, with any per-slot overrides from models.config.json laid
    on top (matched by key). A missing or malformed file is ignored so the
    dashboard always starts."""
    models = [dict(m) for m in DEFAULT_MODELS]
    by_key = {m["key"]: m for m in models}
    try:
        with open(MODELS_CONFIG_FILE) as f:
            for entry in (json.load(f).get("models") or []):
                slot = by_key.get(entry.get("key"))
                if slot:
                    for fld in EDITABLE_FIELDS:
                        if entry.get(fld):
                            slot[fld] = entry[fld]
                    if entry.get("empty"):   # an explicitly emptied slot stays empty
                        slot.update(model="", label="", base="", system="", machine=None)
    except Exception:  # noqa: BLE001
        pass
    return models


# Fallback per-slot model menu for the warm-up dropdown, used only until a Check
# discovers what an endpoint really serves. Open (ungated) Hugging Face repos that
# vLLM serves by name.
_STARTER_CATALOG = ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct",
                    "NousResearch/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen3-8B", "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.3"]
CATALOGS = {m["key"]: list(_STARTER_CATALOG) for m in DEFAULT_MODELS}

# Per-MODEL display metadata, keyed by the ACTUAL model/variant name (not the slot).
# This is the source of truth so the grid, comparison, and report always describe
# what is really loaded on the GPU - critical for an evaluation tool.
MODEL_META = {
    # text models
    "Qwen/Qwen2.5-3B-Instruct": {"params_b": 3, "base": "Qwen2.5-3B", "quant": "BF16", "desc": "Alibaba Qwen2.5 3B Instruct. Small and fast with strong multilingual, math, and coding ability for its size."},
    "Qwen/Qwen3-4B": {"params_b": 4, "base": "Qwen3-4B", "quant": "BF16", "desc": "Alibaba Qwen3 4B. A small hybrid model that can answer directly or think step by step before answering."},
    "microsoft/Phi-3.5-mini-instruct": {"params_b": 4, "base": "Phi-3.5-mini", "quant": "BF16", "desc": "Microsoft Phi-3.5 mini. Compact 3.8B generalist with strong reasoning for its size."},
    "NousResearch/Meta-Llama-3.1-8B-Instruct": {"params_b": 8, "base": "Llama-3.1-8B", "quant": "BF16", "desc": "Meta Llama 3.1 8B Instruct (open mirror). Fast and extremely popular for low-cost chat, Q&A, and summarization."},
    "Qwen/Qwen2.5-7B-Instruct": {"params_b": 7, "base": "Qwen2.5-7B", "quant": "BF16", "desc": "Alibaba Qwen2.5 7B Instruct. Strong multilingual, math, and coding ability alongside solid Q&A."},
    "mistralai/Mistral-7B-Instruct-v0.3": {"params_b": 7, "base": "Mistral-7B-v0.3", "quant": "BF16", "desc": "Mistral 7B Instruct v0.3. Efficient general-purpose model for chat, Q&A, and summarization."},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"params_b": 7, "base": "Qwen2.5-Coder-7B", "quant": "BF16", "desc": "Alibaba Qwen2.5 Coder 7B. Tuned for writing and reviewing code across many languages."},
    "Qwen/Qwen3-8B": {"params_b": 8, "base": "Qwen3-8B", "quant": "BF16", "desc": "Alibaba Qwen3 8B. A hybrid model that can answer directly or think step by step before answering."},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"params_b": 8, "base": "R1-Distill-Llama-8B", "quant": "BF16", "desc": "DeepSeek-R1 distilled onto Llama 8B. A reasoning model that thinks step by step before answering."},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"params_b": 7, "base": "R1-Distill-Qwen-7B", "quant": "BF16", "desc": "DeepSeek-R1 distilled onto Qwen 7B. A compact reasoner that thinks before answering."},
    "Qwen/Qwen2.5-14B-Instruct-AWQ": {"params_b": 14, "base": "Qwen2.5-14B", "quant": "AWQ", "desc": "Alibaba Qwen2.5 14B Instruct, AWQ-quantized to fit 24GB cards."},
    "Qwen/Qwen3-32B": {"params_b": 32, "base": "Qwen3-32B", "quant": "BF16", "desc": "Alibaba Qwen3 32B. A hybrid model that can answer directly or think step by step before answering."},
    "openai/gpt-oss-120b": {"params_b": 120, "base": "gpt-oss 120B", "quant": "MXFP4", "desc": "OpenAI's open-weight 120B mixture-of-experts model (~5B active params per token). Strong reasoning and instruction following at high throughput; needs a large-memory GPU box."},
    # vision models
    "Qwen/Qwen2.5-VL-7B-Instruct": {"params_b": 7, "base": "Qwen2.5-VL-7B", "quant": "BF16", "desc": "Alibaba Qwen2.5-VL 7B Instruct. Compact vision model with strong document and chart reading."},
    "Qwen/Qwen2.5-VL-32B-Instruct": {"params_b": 32, "base": "Qwen2.5-VL-32B", "quant": "BF16", "desc": "Alibaba Qwen2.5-VL 32B Instruct. A top open vision model for documents, tables, charts, and handwriting."},
    "microsoft/Phi-3.5-vision-instruct": {"params_b": 4, "base": "Phi-3.5-Vision", "quant": "BF16", "desc": "Microsoft Phi-3.5 Vision. A small vision model tuned for charts, tables, and document understanding."},
    "openbmb/MiniCPM-V-4_5": {"params_b": 9, "base": "MiniCPM-V-4.5", "quant": "BF16", "desc": "OpenBMB MiniCPM-V 4.5. Efficient 8B vision model with strong OCR-style document, table, and handwriting reading."},
}


def _fallback_desc(m):
    """Neutral description for a model that isn't in MODEL_META (e.g. a user-loaded
    one), so a swapped slot never shows another model's blurb."""
    sysn = (m.get("system") or "").strip()
    return f"{m.get('model','This model')} served on {sysn}." if sysn else f"{m.get('model','This model')} served via your own endpoint."


def model_view(m):
    """Resolve a slot's DISPLAY identity (name/desc/size) to the model actually on it,
    so nothing downstream mislabels what was evaluated. Unknown models fall back to the
    slot's own fields (and the name is always the real model string)."""
    meta = MODEL_META.get(m.get("model"), {})
    return {"label": m.get("model") or m.get("label", ""),
            "desc": meta.get("desc") or _fallback_desc(m),
            "params_b": meta.get("params_b", m.get("params_b"))}


def _pretty_base(s):
    """'llama-3.1-8b-instruct' -> 'Llama-3.1-8B', 'qwen25-7b' -> 'Qwen2.5-7B'."""
    s = (s or "").replace("-instruct", "").strip("-")
    out = []
    for p in s.split("-"):
        fam = re.fullmatch(r"([a-z]+)(\d)(\d)", p)   # qwen25 -> Qwen2.5
        if re.fullmatch(r"\d+\.?\d*b", p):
            out.append(p[:-1] + "B")                 # 8b -> 8B
        elif fam:
            out.append(fam.group(1).capitalize() + fam.group(2) + "." + fam.group(3))
        elif p and p[0].isalpha():
            out.append(p[0].upper() + p[1:])         # llama -> Llama
        else:
            out.append(p)
    return "-".join(out)


def model_specs(m):
    """base model + quantization + LoRA adapters for the model on this slot. Gateway
    slots pull it LIVE from the gateway; others fall back to MODEL_META."""
    if _is_gateway(m):
        gv = gateway_specs().get(m.get("model"))
        if gv:
            return {"base": _pretty_base(gv["base"]), "quant": (gv["quant"] or "").upper(),
                    "adapters": list(gv["adapters"])}
    meta = MODEL_META.get(m.get("model"), {})
    return {"base": meta.get("base", ""), "quant": meta.get("quant", ""), "adapters": []}


def _catalog_for(m):
    """Gateway slots get the live variant list; others use the static catalog."""
    if _is_gateway(m):
        return gateway_variants() or CATALOGS.get(m["key"], [])
    return CATALOGS.get(m["key"], [])


def _models_editable():
    """The per-slot view the warm-up editor reads/writes (+ the dropdown catalog)."""
    return [{**{k: m.get(k) for k in ("key", "color") + EDITABLE_FIELDS},
             "catalog": _catalog_for(m), "can_think": model_can_think(m),
             "can_see": model_can_see(m)} for m in MODELS]


def _save_models_config():
    try:
        data = {"models": [{**{k: m.get(k) for k in ("key",) + EDITABLE_FIELDS},
                            "empty": not (m.get("model") and m.get("base"))} for m in MODELS]}
        with open(MODELS_CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


MODELS = _load_models()
MODEL_BY_KEY = {m["key"]: m for m in MODELS}

MAX_ROUNDS = 5             # full passes per model before it stops (override per run via /run/start {rounds})
MAX_RUN_SECONDS = 15 * 60  # hard backstop: a walked-away run can't loop longer than this (running time, excl. pauses)

# ---- user preferences (set in the dashboard's Settings hub; persisted, gitignored) ----
SETTINGS_FILE = os.path.join(HERE, "settings.config.json")


def _load_settings():
    out = {"rounds": MAX_ROUNDS,   # default matches the historical behavior (5 passes per question)
           "text_baseline": "ocr"}  # document compare's text lane: "ocr" | "parse" | "both"
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        out["rounds"] = max(1, min(MAX_ROUNDS, int(data.get("rounds") or out["rounds"])))
        if data.get("text_baseline") in ("ocr", "parse", "both"):
            out["text_baseline"] = data["text_baseline"]
    except Exception:  # noqa: BLE001
        pass
    return out


SETTINGS = _load_settings()


def _save_settings():
    _jdump(SETTINGS_FILE, SETTINGS)


# ---- Machines: the one place addresses live. Slots and cards reference a machine by id
#      and re-resolve against it, so a moved server never strands a stale URL in a card. ----
MACHINES_FILE = os.path.join(HERE, "machines.config.json")


def _load_machines():
    try:
        with open(MACHINES_FILE, encoding="utf-8") as f:
            ms = json.load(f).get("machines") or []
            if ms:
                return ms
    except Exception:  # noqa: BLE001
        pass
    # first run: derive machines from the current slots (one per unique address)
    ms, seen = [], {}
    for m in MODELS:
        base = (m.get("base") or "").rstrip("/")
        if not base or base in seen:
            if base:
                m["machine"] = seen[base]
            continue
        mid = f"mach-{len(ms) + 1}"
        seen[base] = mid
        ms.append({"id": mid, "name": m.get("system") or f"Machine {len(ms) + 1}",
                   "base": m.get("base"), "kind": "openai"})
        m["machine"] = mid
    return ms


def _save_machines():
    _jdump(MACHINES_FILE, {"machines": MACHINES})


MACHINES = _load_machines()
_save_machines()


# ---- mwboot bootloader enrollment: tokens minted with each generated bootloader; a GPU
# box presents one to /enroll (the only endpoint outside Basic Auth) to register itself ----
ENROLL_FILE = os.path.join(HERE, "enrollments.json")
ENROLL_TTL = 24 * 3600     # a bootloader lying in Downloads next month must be inert
ENROLL_MAX_USES = 16       # same file may enroll a few boxes / many GPUs, not a fleet


def _load_enrollments():
    return _jload(ENROLL_FILE, "tokens", {})


ENROLLMENTS = _load_enrollments()


def _save_enrollments():
    _jdump(ENROLL_FILE, {"tokens": ENROLLMENTS})


def mint_enrollment():
    """A fresh (token, control key) pair for one generated bootloader."""
    tok, ckey = os.urandom(16).hex(), os.urandom(16).hex()
    now = time.time()
    for t in [t for t, e in ENROLLMENTS.items() if now - e.get("created", 0) > ENROLL_TTL]:
        del ENROLLMENTS[t]   # expired tokens don't accumulate
    ENROLLMENTS[tok] = {"ckey": ckey, "created": now, "uses": 0}
    _save_enrollments()
    return tok, ckey


def take_enrollment(tok):
    """Validate + consume one use of an enrollment token. Returns the entry or an error string."""
    e = ENROLLMENTS.get(tok or "")
    if not e:
        return None, "unknown or revoked token - generate a fresh bootloader in Configure > Machines"
    if time.time() - e.get("created", 0) > ENROLL_TTL:
        return None, "this bootloader has expired - generate a fresh one in Configure > Machines"
    if e.get("uses", 0) >= ENROLL_MAX_USES:
        return None, "this bootloader has been used too many times - generate a fresh one"
    e["uses"] = e.get("uses", 0) + 1
    _save_enrollments()
    return e, None


def build_bootloader(mp_url):
    """The one-file machine onboarder: the bootstrap template with this dashboard's URL,
    a fresh token + control key, and the swap agent embedded (base64). Returns (script, token)."""
    tok, ckey = mint_enrollment()
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "mwboot-bootstrap.sh"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(root, "mwboot-control.py"), "rb") as f:
        agent_b64 = base64.b64encode(f.read()).decode()
    agent_wrapped = "\n".join(agent_b64[i:i + 76] for i in range(0, len(agent_b64), 76))
    script = (tpl.replace("__MP_URL__", mp_url.rstrip("/"))
                 .replace("__TOKEN__", tok)
                 .replace("__AGENT_B64__", agent_wrapped)
                 .replace("__CKEY__", ckey))
    return script, tok


def _mach_for_base(base):
    b = (base or "").rstrip("/")
    return next((m for m in MACHINES if (m.get("base") or "").rstrip("/") == b), None)


# ---- the model library: models a user added by link. Persistent, shared by everyone on
# this board, and first-class: auto-lineup, machine taps, and the catalog all use it. ----
LIBRARY_FILE = os.path.join(HERE, "library.json")


def _load_library():
    return _jload(LIBRARY_FILE, "models", [])


LIBRARY = _load_library()


def _save_library():
    _jdump(LIBRARY_FILE, {"models": LIBRARY})


def lib_entry(model_id):
    return next((x for x in LIBRARY if x.get("id") == model_id), None)


def hf_model_meta(mid):
    """Best-effort single call to the public HF API at add time: real on-disk weight size,
    vision capability from the pipeline tag, params, gated flag. Offline or failing, name
    heuristics fill in - the add always succeeds."""
    meta = {"id": mid, "params_b": _params_of(mid), "weight_gb": 0.0,
            "vision": bool(model_can_see({"kind": "openai", "base": "", "model": mid})),
            "gated": False, "vendor": (mid.split("/")[0] if "/" in mid else ""),
            "blurb": "Added from Hugging Face."}
    try:
        d = _http_json("https://huggingface.co/api/models/" + urllib.parse.quote(mid) + "?blobs=true", timeout=8)
        sibs = d.get("siblings") or []
        # count root-level weight files only: repos often carry duplicate weights in
        # subfolders (original/, metal/) that vLLM never downloads
        weights = [s for s in sibs if "/" not in str(s.get("rfilename", ""))
                   and str(s.get("rfilename", "")).endswith((".safetensors", ".bin", ".gguf"))]
        if not weights:
            weights = [s for s in sibs if str(s.get("rfilename", "")).endswith((".safetensors", ".bin", ".gguf"))]
        size = sum((s.get("size") or 0) for s in weights)
        if size:
            meta["weight_gb"] = round(size / 1e9, 1)
        total = (d.get("safetensors") or {}).get("total")
        if total:
            meta["params_b"] = round(total / 1e9, 1)
        tags = [str(t).lower() for t in (d.get("tags") or [])]
        pipe = str(d.get("pipeline_tag") or "").lower()
        if pipe in ("image-text-to-text", "visual-question-answering") or "image-text-to-text" in tags:
            meta["vision"] = True
        meta["gated"] = bool(d.get("gated"))
        meta["hf_ok"] = True   # HF really answered - callers may trust (and persist) this meta
        dls = d.get("downloads")
        if dls:
            meta["blurb"] = f"Added from Hugging Face · {dls:,} downloads last month."
    except Exception:  # noqa: BLE001
        pass
    return meta


def _model_need_gb(meta):
    """GB of GPU memory a model realistically needs to serve."""
    need = float(meta.get("weight_gb") or 0)
    if not need:
        pb = float(meta.get("params_b") or 4)
        quant = bool(re.search(r"awq|gptq|int4|4bit|mxfp4|q4", (meta.get("id") or "").lower()))
        need = pb * (0.7 if quant else 2.1)
    return round(need + 2, 1)   # KV cache + runtime overhead floor


def _machine_fit(meta, mm):
    """True / False / None(unknown): can this machine hold the model? Enrolled VRAM when
    known; a machine that never told us its memory neither blocks nor promises."""
    need = _model_need_gb(meta)
    vram_gb = (mm.get("vram_mb") or 0) / 1024.0
    if vram_gb:
        return need <= vram_gb, need
    return None, need


def machine_by_id(mid):
    return next((x for x in MACHINES if x.get("id") == mid), None)


def apply_machines(new_list):
    """Replace the registry, then re-point every slot that references an edited machine."""
    prior = {m.get("id"): m for m in MACHINES}
    clean = []
    for i, m in enumerate(new_list or []):
        base = str(m.get("base") or "").strip().rstrip("/")
        if not base:
            continue
        # optional bearer for hosted endpoints (NIM, cloud /v1). Setup files travel without
        # keys, so an import that OMITS the field keeps the key already saved for that machine;
        # an explicit empty string (the user cleared the field) really clears it.
        key = str(m["key"] or "").strip() if "key" in m else ((prior.get(m.get("id")) or {}).get("key") or "")
        old = prior.get(m.get("id")) or {}
        clean.append({"id": m.get("id") or f"mach-{int(time.time())}-{i}",
                      "name": (str(m.get("name") or "").strip() or f"Machine {i + 1}")[:60],
                      "base": base,
                      "key": key,
                      # enrollment/agent-owned fields ride along untouched by UI saves and imports
                      "ssh": str(m.get("ssh") or old.get("ssh") or "") or None,
                      "ckey": str(m.get("ckey") or old.get("ckey") or ""),
                      "cport": int(m.get("cport") or old.get("cport") or 0) or None,
                      "vram_mb": int(m.get("vram_mb") or old.get("vram_mb") or 0) or None,
                      "gpu": str(m.get("gpu") or old.get("gpu") or "") or None,
                      "agent": bool(m.get("agent") or old.get("agent")),
                      "kind": "openai"})   # every machine speaks the OpenAI /v1 API
    MACHINES[:] = clean
    _save_machines()
    for s in MODELS:
        mm = machine_by_id(s.get("machine"))
        if mm:
            s["base"], s["kind"], s["system"] = mm["base"], mm["kind"], mm["name"]
    _save_models_config()
    return MACHINES


# ---- the auto-picker: field the best race lineup for a task, no model knowledge needed ----
# Ranked per task: strongest first. params_b gates capacity; family drives diversity
# (a race wants different horses). Future task types (coding, multimodal) add rows here.
TASK_PICKS = {
    "qa": [("openai/gpt-oss-120b", 120), ("Qwen/Qwen3-32B", 32),
           ("Qwen/Qwen2.5-14B-Instruct-AWQ", 14),
           ("NousResearch/Meta-Llama-3.1-8B-Instruct", 8), ("Qwen/Qwen3-8B", 8),
           ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 8),
           ("Qwen/Qwen2.5-7B-Instruct", 7), ("mistralai/Mistral-7B-Instruct-v0.3", 7),
           ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7), ("Qwen/Qwen2.5-Coder-7B-Instruct", 7),
           ("microsoft/Phi-3.5-mini-instruct", 4), ("Qwen/Qwen3-4B", 4),
           ("Qwen/Qwen2.5-3B-Instruct", 3)],
    # Document ingestion needs vision models - ones that read the page image itself.
    # Phi-3.5-vision is retired from picks: weakest doc reader AND it needs
    # trust-remote-code, which not every agent passes (a picker swap onto a box
    # without it crash-loops vLLM). Still in MODEL_META for display of old runs.
    "doc": [("Qwen/Qwen2.5-VL-32B-Instruct", 32), ("openbmb/MiniCPM-V-4_5", 9),
            ("Qwen/Qwen2.5-VL-7B-Instruct", 7)],
}

# Curated vLLM roster: known-OPEN Hugging Face repos (no gated licenses), so a tap
# on an agent-equipped box "just works". Trending refresh adds more at the user's request.
VLLM_PICKS = [
    ("openai/gpt-oss-120b", 120),
    ("Qwen/Qwen3-32B", 32), ("Qwen/Qwen2.5-14B-Instruct-AWQ", 14),
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", 8), ("Qwen/Qwen3-8B", 8),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 8), ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7),
    ("Qwen/Qwen2.5-7B-Instruct", 7), ("Qwen/Qwen2.5-Coder-7B-Instruct", 7),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7),
    ("Qwen/Qwen2.5-VL-32B-Instruct", 32),
    ("Qwen/Qwen2.5-VL-7B-Instruct", 7), ("openbmb/MiniCPM-V-4_5", 9),
    ("microsoft/Phi-3.5-mini-instruct", 4), ("Qwen/Qwen3-4B", 4),
    ("Qwen/Qwen2.5-3B-Instruct", 3),
]

# The trending shelf: human names + one-liners so picking never needs model knowledge.
CATALOG_INFO = {
    "openai/gpt-oss-120b": {"title": "gpt-oss 120B", "vendor": "OpenAI", "blurb": "OpenAI's open-weight flagship. Huge and smart; needs a big-memory box."},
    "mistralai/Mistral-7B-Instruct-v0.3": {"title": "Mistral 7B v0.3", "vendor": "Mistral", "blurb": "Efficient all-rounder for chat and summaries."},
    "microsoft/Phi-3.5-mini-instruct": {"title": "Phi-3.5 mini", "vendor": "Microsoft", "blurb": "Compact generalist with strong reasoning for its size."},
    "Qwen/Qwen3-4B": {"title": "Qwen3 4B", "vendor": "Alibaba", "blurb": "Small hybrid thinker - answers fast or reasons step by step."},
    "Qwen/Qwen3-32B": {"title": "Qwen3 32B", "vendor": "Alibaba", "blurb": "Hybrid thinker - answers fast or reasons step by step."},
    "Qwen/Qwen2.5-14B-Instruct-AWQ": {"title": "Qwen2.5 14B AWQ", "vendor": "Alibaba", "blurb": "Bigger Qwen squeezed to fit 24GB cards."},
    "NousResearch/Meta-Llama-3.1-8B-Instruct": {"title": "Llama 3.1 8B", "vendor": "Meta", "blurb": "The crowd favorite - fast, solid, everyday model."},
    "Qwen/Qwen3-8B": {"title": "Qwen3 8B", "vendor": "Alibaba", "blurb": "Small hybrid thinker, quick on modest GPUs."},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"title": "DeepSeek-R1 Llama 8B", "vendor": "DeepSeek", "blurb": "Compact reasoner built on Llama."},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"title": "DeepSeek-R1 Qwen 7B", "vendor": "DeepSeek", "blurb": "Compact reasoner built on Qwen."},
    "Qwen/Qwen2.5-7B-Instruct": {"title": "Qwen2.5 7B", "vendor": "Alibaba", "blurb": "Strong at math and languages for its size."},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"title": "Qwen2.5 Coder 7B", "vendor": "Alibaba", "blurb": "Tuned for writing and fixing code."},
    "Qwen/Qwen2.5-3B-Instruct": {"title": "Qwen2.5 3B", "vendor": "Alibaba", "blurb": "Tiny but sharp."},
    # vision models (read images and documents, not just text)
    "Qwen/Qwen2.5-VL-7B-Instruct": {"title": "Qwen2.5-VL 7B", "vendor": "Alibaba", "blurb": "Compact vision model, good at documents."},
    "Qwen/Qwen2.5-VL-32B-Instruct": {"title": "Qwen2.5-VL 32B", "vendor": "Alibaba", "blurb": "Top open document reader - tables, charts, handwriting."},
    "microsoft/Phi-3.5-vision-instruct": {"title": "Phi-3.5 Vision 4B", "vendor": "Microsoft", "blurb": "Small vision model tuned for charts and documents."},
    "openbmb/MiniCPM-V-4_5": {"title": "MiniCPM-V 4.5", "vendor": "OpenBMB", "blurb": "Strong compact document reader - handwriting, tables, receipts."},
}


# ---- trending refresh (manual only): the user presses the button, we ask Hugging Face.
#      Nothing is fetched automatically - offline-first stays the default. ----
CATALOG_FILE = os.path.join(HERE, "catalog.config.json")


def _load_trending():
    try:
        with open(CATALOG_FILE, encoding="utf-8") as f:
            d = json.load(f)
            return {"fetched_at": d.get("fetched_at") or "", "entries": d.get("entries") or []}
    except Exception:  # noqa: BLE001
        return {"fetched_at": "", "entries": []}


TRENDING = _load_trending()

_VENDOR_ORGS = {"meta-llama": "Meta", "qwen": "Alibaba", "google": "Google", "deepseek-ai": "DeepSeek",
                "mistralai": "Mistral", "openai": "OpenAI", "microsoft": "Microsoft", "ibm-granite": "IBM",
                "nvidia": "NVIDIA", "moonshotai": "Moonshot", "zai-org": "Z.ai", "unsloth": "Unsloth"}


# Multimodal architectures this build knows vLLM can serve. Fail-safe by design:
# an architecture NOT listed here is never one-tap offered (new archs appear on HF
# before serving engines support them) - it can still join via the library flow.
VLLM_VLM_ARCHS = {
    "Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
    "InternVLChatModel", "InternVLForConditionalGeneration",
    "LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration", "MiniCPMV",
    "Gemma3ForConditionalGeneration", "PaliGemmaForConditionalGeneration",
    "Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration",
    "Phi3VForCausalLM", "Phi4MMForCausalLM",
    "PixtralForConditionalGeneration", "Mistral3ForConditionalGeneration",
    "MllamaForConditionalGeneration", "MolmoForCausalLM",
    "DeepseekVLV2ForCausalLM", "Glm4vForConditionalGeneration", "GLM4VForCausalLM",
    "KimiVLForConditionalGeneration", "AyaVisionForConditionalGeneration", "Ovis",
}

# Known multimodal lineages: a NEW arch in a family vLLM already serves (Qwen3_5...,
# Gemma4...) is almost always supported by a current vLLM before this list updates.
_VLM_FAMILY = re.compile(r"qwen|gemma|internvl|intern_vl|llava|minicpm|ovis|glm|phi\d|pixtral"
                         r"|mistral|idefics|smolvlm|molmo|deepseek|kimi|paligemma|mllama|aya", re.I)


def _vlm_arch_ok(arch):
    """Can vLLM (probably) boot this vision architecture? Exact list, else family
    heuristic. Agents report their install's real list and override this per machine."""
    if not arch:
        return False
    if arch in VLLM_VLM_ARCHS:
        return True
    return bool(_VLM_FAMILY.search(arch)) and arch.endswith(
        ("ForConditionalGeneration", "ForCausalLM", "ChatModel"))


def refresh_trending():
    """Ask Hugging Face for today's trending models - text generators AND vision
    readers (image-text-to-text), so the shelf stays current for document tasks too.
    Vision entries carry architecture + true weight size + gating from HF metadata,
    so the shelf only one-taps what vLLM can boot and the card can hold. Persisted
    locally; runs only on user demand. Returns (total, vision_count)."""
    entries, seen, fetched = [], set(), 0
    for pipe, limit, vision in (("text-generation", 40, False), ("image-text-to-text", 25, True)):
        url = (f"https://huggingface.co/api/models?pipeline_tag={pipe}"
               f"&sort=trendingScore&direction=-1&limit={limit}")
        if vision:   # one list call also brings arch, true param count, and gating
            url += "&expand%5B%5D=config&expand%5B%5D=safetensors&expand%5B%5D=gated" \
                   "&expand%5B%5D=downloads&expand%5B%5D=tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-bench"})
            with urllib.request.urlopen(req, timeout=12) as r:
                items = json.loads(r.read())
            fetched += 1
        except Exception:  # noqa: BLE001 - one pipeline down shouldn't empty the other's refresh
            continue
        for it in items:
            mid = it.get("id") or ""
            if "/" not in mid or mid in seen:
                continue
            seen.add(mid)
            org, repo = mid.split("/", 1)
            gguf = "gguf" in (it.get("tags") or [])
            pm = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", repo)
            params = float(pm.group(1)) if pm else None   # unknown size -> never offered as a one-tap get
            title = re.sub(r"[-_]", " ", re.sub(r"-?(Instruct|Chat|GGUF|HF)$", "", repo, flags=re.I)).strip()
            vendor = _VENDOR_ORGS.get(org.lower(), org)
            dls = it.get("downloads") or 0
            blurb = ("Trending vision reader on Hugging Face" if vision else "Trending on Hugging Face") \
                + (f" - {dls:,} downloads" if dls else "")
            e = {"id": mid, "tag": mid, "gguf": gguf,
                 "params": params, "title": title[:48], "vendor": vendor, "blurb": blurb}
            if vision:
                e["vision"] = True
                total = ((it.get("safetensors") or {}).get("total") or 0)
                if total:
                    e["params"] = round(total / 1e9, 1)   # true size beats the name guess
                arch = ((it.get("config") or {}).get("architectures") or [""])[0]
                e["arch"] = arch
                e["vllm_ok"] = _vlm_arch_ok(arch)
                e["gated"] = bool(it.get("gated"))
            entries.append(e)
    if not fetched:
        raise OSError("Hugging Face unreachable")
    TRENDING["entries"] = entries
    TRENDING["fetched_at"] = time.strftime("%Y-%m-%d %H:%M")
    _see_cache.clear()   # fresh HF metadata may correct earlier name-guessed answers
    try:
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(TRENDING, f, indent=2)
    except Exception:  # noqa: BLE001
        pass
    return len(entries), sum(1 for e in entries if e.get("vision"))


def merged_catalog():
    """Built-in curated entries + the last trending refresh + the user's own library."""
    out = dict(CATALOG_INFO)
    for e in TRENDING.get("entries") or []:
        if e["tag"] not in out:
            out[e["tag"]] = {"title": e["title"], "vendor": e["vendor"], "blurb": e["blurb"]}
    for x in LIBRARY:
        short = x["id"].split("/")[-1].replace("-Instruct", "").replace("-instruct", "").replace("-", " ").replace("_", " ")
        out[x["id"]] = {"title": short, "vendor": x.get("vendor") or "", "blurb": x.get("blurb") or "Added from Hugging Face."}
    return out


def _params_of(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name or "")
    return float(m.group(1)) if m else 4.0


def _family_of(name):
    n = (name or "").lower()
    for fam, keys in (("meta", ("llama",)), ("qwen", ("qwen",)), ("google", ("gemma", "codegemma")),
                      ("mistral", ("mistral", "mixtral")), ("deepseek", ("deepseek",)),
                      ("openai", ("gpt-oss",)), ("microsoft", ("phi",))):
        if any(k in n for k in keys):
            return fam
    return n.split("/")[0] or "other"


def auto_lineup(task="qa"):
    """Pick the strongest three-way match-up from the live shelf: every online machine
    fields its best eligible model; distinct families preferred; served models beat
    downloads; at most one pick needs a download. Applies the result to the slots."""
    ranked = TASK_PICKS.get(task) or TASK_PICKS["qa"]
    rank = {name: i for i, (name, _) in enumerate(ranked)}
    # document ingestion is a vision task: only models that can read images may race.
    # every other task is text: vision models sit it out (they only race each other, on docs)
    eligible = (lambda mc, n: n in rank or model_can_see({"kind": mc["kind"], "base": mc["base"], "model": n})) \
        if task == "doc" else \
        (lambda mc, n: not model_can_see({"kind": mc["kind"], "base": mc["base"], "model": n}))
    cands = []
    for mc in shelf_view():
        # a machine counts if its server answers OR its agent does - the agent can start
        # any cached model even when nothing is being served right now
        if not (mc.get("online") or mc.get("can_swap")):
            continue
        served = [m["name"] for m in mc.get("models") or []]
        capacity = max([_params_of(n) for n in served] or [4.0])   # evidence: it already holds this much
        for n in served:
            if not eligible(mc, n):
                continue
            cands.append({"machine": mc, "model": n, "served": True,
                          "score": rank.get(n, 40 + max(0, 40 - _params_of(n))), "family": _family_of(n)})
        if mc.get("can_pull"):
            for (n, pb) in ranked:
                if n not in served and pb <= capacity:
                    cands.append({"machine": mc, "model": n, "served": False,
                                  "score": rank.get(n, 99) + 0.5, "family": _family_of(n)})
        elif mc.get("can_swap"):
            # a control-plane vLLM box can restart onto another model. CACHED weights make
            # that a ~1 min swap - the picker's favorite after served, and no capacity
            # doubt (someone downloaded them to that box on purpose). Anything else means
            # taking the server down for a full download, so it ranks far behind and keeps
            # the capacity-evidence gate (small slack: a box that held 7B also holds 8B).
            cached = set(mc.get("cached") or [])
            pool = list(VLLM_PICKS)
            for x in LIBRARY:   # user-added models are first-class picks, fit-checked by VRAM
                if _machine_fit(x, mc)[0] is not False:
                    pool.append((x["id"], float(x.get("params_b") or 4)))
            for (n, pb) in pool:
                if n in served or not eligible(mc, n):
                    continue
                if n in cached:
                    cands.append({"machine": mc, "model": n, "served": False, "swap": True,
                                  "score": rank.get(n, 60 + max(0, 40 - pb)), "family": _family_of(n)})
                elif pb <= capacity * 1.3 or lib_entry(n):
                    cands.append({"machine": mc, "model": n, "served": False, "swap": True,
                                  "score": 200 + rank.get(n, 60 + max(0, 40 - pb)), "family": _family_of(n)})
    cands.sort(key=lambda c: (not c["served"], c["score"]))
    picks, used_pairs, used_machines, used_families, pulls = [], set(), set(), set(), [0]

    def take(c):
        picks.append(c)
        used_pairs.add((c["machine"]["id"], c["model"]))
        used_machines.add(c["machine"]["id"])
        used_families.add(c["family"])
        if not c["served"] and not c.get("swap"):
            pulls[0] += 1

    # Served models beat swaps and downloads OUTRIGHT: relax machine/family spread over
    # what is already on the GPUs before reaching for a swap or a pull. Otherwise the
    # family-diversity pass would restart a server just to avoid two same-family picks.
    for served_only in (True, False):
        for spread_machines, spread_families in ((True, True), (True, False), (False, True), (False, False)):
            for c in cands:
                if len(picks) == 3:
                    break
                if served_only and not c["served"]:
                    continue
                if (c["machine"]["id"], c["model"]) in used_pairs:
                    continue
                if any(p["model"] == c["model"] for p in picks):
                    continue   # a race needs different horses, even across machines
                if not c["served"] and not c.get("swap") and pulls[0] >= 1:
                    continue   # at most one download per auto-pick (server swaps are cheap, downloads aren't)
                if spread_machines and c["machine"]["id"] in used_machines:
                    continue
                if spread_families and c["family"] in used_families:
                    continue
                take(c)
            if len(picks) == 3:
                break
        if len(picks) == 3:
            break
    if not picks:
        return None
    slots = []
    for i, c in enumerate(picks[:3]):
        key = MODELS[i]["key"] if i < len(MODELS) else "slot-" + str(i)
        slots.append({"key": key, "label": c["model"], "system": c["machine"]["name"],
                      "kind": c["machine"]["kind"], "base": c["machine"]["base"],
                      "model": c["model"], "machine": c["machine"]["id"]})
    apply_models_config(slots)
    return [{"key": sl["key"], "model": sl["model"], "machine": sl["system"],
             "served": c["served"]} for sl, c in zip(slots, picks)]


def shelf_view():
    """The Model Shelf: every machine with what it serves now, what it could get, and
    agent-reported hardware. This is the whole truth the picker needs - no URLs in the UI."""
    out = []
    for mm in MACHINES:
        pr = probe_endpoint(mm.get("kind"), mm.get("base"), mm.get("key"))
        online = bool(pr.get("ok"))
        kind = pr.get("kind") or mm.get("kind")
        served = (pr.get("models") or []) if online else []
        capacity = max([_params_of(n) for n in served] or [4.0]) if online else 0
        entry = {"id": mm["id"], "name": mm["name"], "base": mm["base"], "kind": kind, "online": online,
                 "capacity": capacity, "key": mm.get("key") or "", "vram_mb": mm.get("vram_mb") or 0,
                 "gateway": bool(pr.get("gateway")), "models": [], "gettable": [], "gpus": [],
                 "can_pull": False, "can_swap": False}
        for name in served[:60]:
            info = {"name": name,
                    "think": model_can_think({"kind": kind, "base": mm["base"], "model": name}),
                    "see": model_can_see({"kind": kind, "base": mm["base"], "model": name})}
            entry["models"].append(info)
        if not entry["gateway"]:   # even with the server down, an answering agent means swappable
            st = _control_status(mm["base"])
            if st is not None:
                entry["can_swap"] = True
                entry["gpus"] = st.get("gpus") or []
                entry["cached"] = st.get("cached") or []   # weights on disk: swaps are minutes, not downloads
                entry["cached_sizes"] = st.get("cached_sizes") or {}
                entry["prefetch"] = st.get("prefetch") or {}   # background download in flight, if any
                # the registry learns hardware truth from the agent - hand-added machines get
                # fit-checked, targeted, and labeled correctly without re-enrolling
                learned = False
                if st.get("vram_mb") and not mm.get("vram_mb"):
                    mm["vram_mb"] = int(st["vram_mb"])
                    entry["vram_mb"] = mm["vram_mb"]
                    learned = True
                if st.get("gpu") and st.get("gpu") != mm.get("gpu"):
                    mm["gpu"] = st["gpu"]   # the managed device, verbatim (incl. MIG/vGPU profiles)
                    learned = True
                if not mm.get("agent"):
                    mm["agent"] = True
                    learned = True
                if learned:
                    _save_machines()
                # this box's vLLM told us its real supported architectures (newer agents)
                agent_archs = set(st.get("archs") or [])
                # tap a card, the agent swaps the server to that model
                capacity = max([_params_of(n) for n in served] or [8.0])
                get = [i for (i, pb) in VLLM_PICKS if i not in served and pb <= capacity][:8]
                # user-added library models join this machine's options when they (can) fit
                for x in LIBRARY:
                    fit, _need = _machine_fit(x, mm)
                    if fit is not False and x["id"] not in served and x["id"] not in get:
                        get.append(x["id"])
                # vision first: text already has curated picks above, and the 16-slot cap
                # would otherwise fill with text entries before any vision reader is seen
                for e in sorted(TRENDING.get("entries") or [], key=lambda x: not x.get("vision")):
                    if len(get) >= 16:
                        break
                    if e.get("gguf") or e["id"] in served or e["id"] in get:
                        continue
                    if e.get("vision"):
                        # a one-tap vision get must be bootable and holdable: known
                        # vLLM-served architecture, no license gate, and real fit
                        # (weights + KV + image buffers) against enrolled VRAM.
                        # A box whose agent reports its real arch list is gated by
                        # THAT list alone - the global heuristic is only for boxes
                        # that haven't told us what their vLLM can actually serve.
                        if agent_archs:
                            bootable = (e.get("arch") or "") in agent_archs
                        else:
                            bootable = e.get("vllm_ok")
                        if not bootable or e.get("gated"):
                            continue
                        need = _model_need_gb({"id": e["id"], "params_b": e.get("params")}) + 1.5
                        vram_gb = (mm.get("vram_mb") or 0) / 1024.0
                        if vram_gb:
                            if need > vram_gb:
                                continue
                        elif (e.get("params") or 99) > capacity * 0.8:   # unknown VRAM: haircut the proxy
                            continue
                    elif (e.get("params") or 99) > capacity:
                        continue
                    get.append(e["id"])
                entry["gettable"] = get
                # cached weights in no list (curated, library, trending) are one restart away
                # but endorsed by NOBODY - offered separately so leftovers never pose as picks
                entry["disk_only"] = [n for n in entry["cached"] if n not in served and n not in get]
        out.append(entry)
    return out

LOCK = threading.Lock()


def _blank_stats():
    return {"requests": 0, "passes": 0, "fails": 0, "ungraded": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_latency_ms": 0.0,
            "loops": 0, "done": False, "last_input": "", "last_output": "", "last_in_tokens": 0, "last_out_tokens": 0, "last_error": "", "last_ms": 0,
            "ttft_ms_total": 0.0, "ttft_count": 0,
            "fields_ok": 0, "fields_total": 0}   # document rows: ground-truth fields read correctly


STATE = {
    "running": False,
    "owner": None,            # client token that started the active run (one-at-a-time lock)
    "paused": False,
    "pause_started": None,
    "pause_total": 0.0,
    "started_at": None,
    "ended_at": None,
    "gen": 0,                 # run generation: bumped each start so stale workers (blocked in a slow call) can't touch a new run
    "max_tokens": 256,
    "rounds": MAX_ROUNDS,
    "think": False,           # "Let models think": allow reasoning models to think before answering
    "rows": [],
    "columns": [],
    "csv_name": "(none)",
    "models": {m["key"]: _blank_stats() for m in MODELS},
    "saved": {m["key"]: 0 for m in MODELS},   # cumulative tokens saved per model (session)
    "lifetime": 0,   # grand total tokens evaluated in-house across all jobs, persisted to disk
    "messages": [],  # notes from watchers to the current driver: [{id, name, text, ts}]
    "msg_seq": 0,    # monotonic id so each client shows a message once
}

# POST paths that mutate/drive the shared run - blocked for non-owners while a run is active
_GATED_WHILE_RUNNING = {
    "/run/start", "/run/stop", "/run/pause", "/run/resume", "/run/reset", "/run/clear",
    "/run/upload", "/run/warmup", "/config/models",
    "/run/dataset/save", "/run/dataset/delete", "/run/dataset/restore", "/run/dataset/purge",
    "/config/import", "/config/machines", "/slot/assign", "/lineup/auto", "/race/enter", "/slot/clear", "/machine/evacuate",
    "/update/apply", "/uninstall",
}

# ---------- lifetime token tally (persisted, never resets) ----------
LIFETIME_FILE = os.path.join(HERE, "lifetime.json")
_last_save = [0.0]


def _load_lifetime():
    try:
        with open(LIFETIME_FILE) as f:
            return int(json.load(f).get("tokens", 0))
    except Exception:  # noqa: BLE001
        return 0


def _save_lifetime():
    _jdump(LIFETIME_FILE, {"tokens": STATE["lifetime"]}, indent=None)


def add_lifetime(n):
    if not n:
        return
    with LOCK:
        STATE["lifetime"] += n
    t = time.time()
    if t - _last_save[0] > 5:   # debounce disk writes
        _last_save[0] = t
        _save_lifetime()


# ---------- input parsing + grading ----------
_Q_COLS = ("question", "prompt", "input", "query")
_A_COLS = ("answer", "expected", "expected_answer", "correct", "contains")


_JSON_Q = ("prompt", "question", "input", "text", "content", "instruction", "query", "act")
_JSON_A = ("answer", "expected", "output", "response", "completion", "target")


def _from_csv(text):
    """CSV/TSV with a recognized question column (question/prompt/...). Else None."""
    first = text.splitlines()[0]
    delim = next((d for d in (",", "\t", ";") if d in first), None)
    if not delim:
        return None
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        cols = reader.fieldnames or []
    except Exception:  # noqa: BLE001
        return None
    low = {(c or "").lower().strip(): c for c in cols if c}
    qcol = next((low[c] for c in _Q_COLS if c in low), None)
    if not qcol:
        return None  # no recognized question column -> not a test CSV
    acol = next((low[c] for c in _A_COLS if c in low), None)
    rows = []
    for d in reader:
        q = (d.get(qcol) or "").strip()
        if not q:
            continue
        row = {"question": q}
        for k, v in d.items():
            if k and k.startswith("__expected") and v:
                row[k] = v
        if acol and (d.get(acol) or "").strip():
            row["__expected_simple"] = "icontains:" + d[acol].strip()
        rows.append(row)
    return rows or None


def _from_json(text):
    t = text.lstrip()
    data = None
    if t[:1] in "[{":
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            data = None
    if data is None:  # JSONL
        objs = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                objs.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                return None
        data = objs or None
    if data is None:
        return None
    items = data if isinstance(data, list) else (data.get("prompts") or data.get("data") or data.get("examples"))
    if not isinstance(items, list):
        return None
    rows = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s:
                rows.append({"question": s})
        elif isinstance(it, dict):
            q = next((it[k] for k in _JSON_Q if it.get(k)), None)
            if not q:
                continue
            row = {"question": str(q).strip()}
            a = next((it[k] for k in _JSON_A if it.get(k)), None)
            if a:
                row["__expected_simple"] = "icontains:" + str(a).strip()
            rows.append(row)
    return rows or None


def _from_md_table(text):
    """Markdown table like | Act | Prompt | (used by awesome-chatgpt-prompts README)."""
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        h, sep = lines[i], lines[i + 1]
        if "|" in h and "-" in sep and re.match(r'^\s*\|?[\s:\-|]+\|[\s:\-|]*$', sep):
            headers = [c.strip().lower() for c in h.strip().strip("|").split("|")]
            if len(headers) < 2:
                continue
            qi = next((idx for idx, hd in enumerate(headers) if hd in _Q_COLS), None)
            ai = next((idx for idx, hd in enumerate(headers) if hd in _A_COLS), None)
            if qi is None:
                qi = len(headers) - 1  # default to last column (e.g. Act | Prompt)
            rows = []
            for ln in lines[i + 2:]:
                if "|" not in ln:
                    if ln.strip() == "":
                        continue
                    break
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if qi >= len(cells):
                    continue
                q = cells[qi].strip().strip('"').replace("<br>", " ").replace("<br/>", " ").strip()
                if not q or set(q) <= set("-: "):
                    continue
                row = {"question": q}
                if ai is not None and ai < len(cells) and cells[ai].strip():
                    row["__expected_simple"] = "icontains:" + cells[ai].strip().strip('"')
                rows.append(row)
            if rows:
                return rows
    return None


def _from_plain(text):
    """One question per line, skipping markdown/HTML noise."""
    rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or len(s) < 3:
            continue
        if s[0] in "#>|`" or s.startswith(("```", "---", "===", "<", "![", "[!")):
            continue
        if '="' in s or "='" in s:  # HTML attributes
            continue
        s = re.sub(r'^[-*+]\s+', '', s)        # bullet
        s = re.sub(r'^\d+[.)]\s+', '', s)      # numbered list
        if len(s) < 3:
            continue
        rows.append({"question": s})
    return rows


_QA_Q = re.compile(r'^(?:Question|Q)\s*[:.)]\s*(.+)$', re.I)
_QA_A = re.compile(r'^(?:Answer|A)\s*[:.)]\s*(.+)$', re.I)


def _from_qa_pairs(text):
    """Pasted 'Q: ... / A: ...' pairs (also Question:/Answer:). Pairs each question with
    its answer into a graded row. Only fires when both Q and A markers are present, so it
    never hijacks a plain question list. A question with no answer stays ungraded."""
    lines = [ln.strip() for ln in text.splitlines()]
    if not (any(_QA_Q.match(ln) for ln in lines) and any(_QA_A.match(ln) for ln in lines)):
        return None
    rows, cur_q = [], None
    for ln in lines:
        if not ln:
            continue
        mq, ma = _QA_Q.match(ln), _QA_A.match(ln)
        if mq:
            if cur_q is not None:               # a question with no answer before this one
                rows.append({"question": cur_q})
            cur_q = mq.group(1).strip()
        elif ma:
            if cur_q is not None:
                row = {"question": cur_q}
                ans = ma.group(1).strip()
                if ans:
                    row["__expected_simple"] = "icontains:" + ans
                rows.append(row)
                cur_q = None
        elif cur_q is not None:                 # continuation line -> append to the open question
            cur_q += " " + ln
    if cur_q is not None:
        rows.append({"question": cur_q})
    return rows or None


def parse_questions(text):
    """Smart, forgiving parse: rejects web pages; understands CSV/TSV, JSON/JSONL,
    markdown tables, Q:/A: pairs, or plain one-question-per-line text."""
    text = (text or "").strip()
    if not text:
        return [], []
    low = text[:1500].lower().lstrip()
    if low.startswith(("<!doctype", "<html", "<?xml")) or "<head" in low[:400] or "<body" in low[:900]:
        raise ValueError("that link returned a web page, not a data file. On GitHub/GitLab open the "
                         "file and use the 'Raw' button - the link should end in .csv, .txt or .json")
    for fn in (_from_json, _from_csv, _from_md_table, _from_qa_pairs):
        try:
            r = fn(text)
        except Exception:  # noqa: BLE001
            r = None
        if r:
            return r, ["question"]
    return _from_plain(text), ["question"]


def _http_get(url, token=None, accept=None):
    headers = {"User-Agent": "llm-bench"}
    if accept:
        headers["Accept"] = accept
    if token:
        if "gitlab" in url.lower():
            headers["PRIVATE-TOKEN"] = token
        else:
            headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


# file-name desirability for repo auto-pick (lower = better); README/license/etc. score 9 = rejected
def _score_name(name):
    n = (name or "").lower()
    if n in ("prompts.csv", "tests.csv", "test.csv", "questions.csv", "prompts.txt",
             "tests.txt", "questions.txt", "prompts.jsonl", "tests.jsonl", "prompts.json", "tests.json"):
        return 0
    if n.endswith((".csv", ".tsv")):
        return 1
    if n.endswith(".jsonl"):
        return 2
    if n.endswith(".json"):
        return 3
    if n.endswith(".txt") and n not in ("requirements.txt", "license.txt", "changelog.txt"):
        return 4
    return 9


# folders to probe (in order) when a bare repo URL gives no folder to look in
_COMMON_DIRS = ("", "data", "prompts", "tests", "test", "evals", "eval",
                "questions", "datasets", "dataset", "benchmarks", "fixtures")


def _gh_contents(owner, repo, path, token, branch=None):
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}".rstrip("/")
    if branch:
        api += f"?ref={branch}"
    return json.loads(_http_get(api, token, accept="application/vnd.github+json"))


def _github_pick(owner, repo, token=None, branch=None, path=""):
    """Find the most likely test file in a GitHub repo. Searches the given folder
    (from a /tree/ link) or, for a bare repo, the root plus common data subfolders.
    Fetches via the contents API (Accept: raw) so private repos work with a token."""
    search = [path] if path else list(_COMMON_DIRS)
    best_sc, best = 9, None
    for d in search:
        try:
            listing = _gh_contents(owner, repo, d, token, branch)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(listing, dict):            # the path pointed at a file, not a folder
            listing = [listing]
        for it in listing:
            if isinstance(it, dict) and it.get("type") == "file":
                sc = _score_name(it.get("name"))
                if sc < best_sc:
                    best_sc, best = sc, it
        if best_sc == 0:
            break                                 # ideal name found, stop early
    if not best or best_sc >= 9:
        raise ValueError("couldn't find a test file (.csv/.txt/.json) in that repo - link the file directly")
    return _http_get(best["url"], token, accept="application/vnd.github.raw"), best.get("html_url") or best["url"]


def _gist_pick(gist_id, token=None):
    """Pull the best (or only) file out of a GitHub gist."""
    data = json.loads(_http_get(f"https://api.github.com/gists/{gist_id}", token,
                                accept="application/vnd.github+json"))
    files = list((data.get("files") or {}).values())
    if not files:
        raise ValueError("that gist has no files")
    files.sort(key=lambda f: _score_name(f.get("filename")))   # gist names are arbitrary; still take the first
    f = files[0]
    content = f.get("content")
    if content is None and f.get("raw_url"):
        content = _http_get(f["raw_url"], token)
    return content or "", f.get("raw_url") or f"https://gist.github.com/{gist_id}"


def _gitlab_pick(host, proj_path, token=None, branch=None, path=""):
    """Find a test file in a GitLab repo via the v4 API (project path is URL-encoded)."""
    enc = urllib.parse.quote(proj_path, safe="")
    base = f"https://{host}/api/v4/projects/{enc}"
    if not branch:
        branch = json.loads(_http_get(base, token)).get("default_branch") or "main"
    tree = f"{base}/repository/tree?per_page=100&ref={urllib.parse.quote(branch)}&path={urllib.parse.quote(path)}"
    listing = json.loads(_http_get(tree, token))
    files = [it for it in listing if isinstance(it, dict) and it.get("type") == "blob"]
    best = min(files, key=lambda it: _score_name(it.get("name")), default=None)
    if not best or _score_name(best.get("name")) >= 9:
        raise ValueError("couldn't find a test file (.csv/.txt/.json) in that GitLab repo - link the file directly")
    raw = f"{base}/repository/files/{urllib.parse.quote(best['path'], safe='')}/raw?ref={urllib.parse.quote(branch)}"
    return _http_get(raw, token), raw


def _hf_dataset_ref(url):
    """Pull (dataset_id, config, split) out of any huggingface.co/datasets URL a user
    might copy - the plain page, a /viewer/<config>/<split> deep link, /tree or /blob
    paths, even the catalog listing page (which gets a helpful refusal)."""
    pu = urllib.parse.urlparse(url if "://" in url else "https://" + url)
    segs = [s for s in pu.path.split("/") if s]
    if not segs or segs[0] != "datasets" or len(segs) == 1:
        raise ValueError("that's the Hugging Face datasets catalog page - open one dataset and paste its page link (it looks like huggingface.co/datasets/org/name)")
    SPECIAL = {"viewer", "tree", "blob", "resolve", "discussions", "commits", "settings"}
    if len(segs) >= 3 and segs[2] not in SPECIAL:
        did, rest = segs[1] + "/" + segs[2], segs[3:]
    else:
        did, rest = segs[1], segs[2:]
    config = split = None
    if rest and rest[0] == "viewer":
        config = urllib.parse.unquote(rest[1]) if len(rest) > 1 else None
        split = urllib.parse.unquote(rest[2]) if len(rest) > 2 else None
    return did, config, split


def fetch_hf_dataset(url, token=None):
    """Import a Hugging Face DATASET as a question set, deterministically - no libraries,
    just the public datasets-server API, and only the FIRST rows: a million-row set costs
    the same few seconds as a small one, nothing is downloaded to disk. An eval-suited
    split is preferred (or the one in a /viewer deep link), columns are mapped by
    conventional names, multiple-choice sets (mmlu/arc/hellaswag style) arrive with
    lettered options in the question and the correct option's text as the answer, and
    chat-transcript sets (a messages/conversations column) map the last user turn to the
    question and the last assistant turn to the answer. Gated sets work when the user has
    stored an hf_ token (+ menu > Add a private token) AND their account has access.
    Returns (csv_text, name, total_rows_in_split)."""
    did, want_config, want_split = _hf_dataset_ref(url)
    api = "https://datasets-server.huggingface.co"
    hf_key = token if token and token.startswith("hf_") else None
    try:
        sp = _http_json(f"{api}/splits?dataset={urllib.parse.quote(did)}", timeout=20, key=hf_key)
    except urllib.error.HTTPError:
        if hf_key:
            raise ValueError(f"'{did}' is private or gated and your Hugging Face token doesn't unlock it - open the dataset page while signed in and accept its terms first, then try again")
        raise ValueError(f"'{did}' is gated, private, or misspelled. If your Hugging Face account has access, press + then Add a private token, paste an hf_ token, and paste the link again")
    entries = sp.get("splits") or []
    if not entries:
        raise ValueError("that dataset has no browsable splits (it may be gated or script-based)")
    if want_config:
        entries = [e for e in entries if e.get("config") == want_config] or entries
    LETTERS = "ABCDEFGHIJ"
    def cell(v):
        v = str(v if v is not None else "").replace("\r", " ").replace("\n", " ").strip()
        return '"' + v.replace('"', '""') + '"' if re.search(r'[",\n]', v) else v
    def build(pick):
        qs = (f"dataset={urllib.parse.quote(did)}&config={urllib.parse.quote(pick['config'])}"
              f"&split={urllib.parse.quote(pick['split'])}")
        try:
            rd = _http_json(f"{api}/rows?{qs}&offset=0&length=100", timeout=30, key=hf_key)
        except Exception:   # noqa: BLE001 - partial viewers ("preview only") still serve first-rows
            rd = _http_json(f"{api}/first-rows?{qs}", timeout=30, key=hf_key)
        total = rd.get("num_rows_total") or 0
        orig, types = {}, {}          # lowercased name -> exact column name / declared type
        for f in rd.get("features") or []:
            n = str(f.get("name", ""))
            orig.setdefault(n.lower(), n)
            types[n.lower()] = f.get("type") or {}
        def find(cands):
            return next((orig[c] for c in cands if c in orig), None)
        qcol = find(["question", "prompt", "instruction", "input", "query", "problem", "text", "goal", "ctx"])
        acol = find(["answer", "answers", "best_answer", "answerkey", "correct_answer", "response",
                     "output", "solution", "target", "label"])
        ccol = find(["choices", "options", "endings", "candidates"])
        # a transcript column beats a bare question column WITHOUT an answer column -
        # the transcript's assistant turns give us graded pairs, the bare column doesn't
        chatcol = None if (qcol and acol) else find(["messages", "conversations", "conversation", "dialogue", "chat"])
        if not qcol and not chatcol:
            raise ValueError("couldn't find a question column - this set has: " + ", ".join(orig.values()))
        labels = None      # ClassLabel answers arrive as ints; names[] turns them back into words
        if acol:
            t = types.get(acol.lower())
            if isinstance(t, dict) and t.get("_type") == "ClassLabel":
                labels = t.get("names") or None
        def chat_qa(msgs):
            # a transcript row: the last human turn is the question, the last model turn the answer
            if not isinstance(msgs, list):
                return None, ""
            turns = []
            for mm in msgs:
                if not isinstance(mm, dict):
                    continue
                role = str(mm.get("role") or mm.get("from") or "").lower()
                c = mm.get("content") or mm.get("value") or ""
                if isinstance(c, list):
                    c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
                turns.append((role, str(c)))
            q = next((c for role, c in reversed(turns) if role in ("user", "human")), None)
            a = next((c for role, c in reversed(turns) if role in ("assistant", "gpt", "bot", "model")), "")
            return q, a
        out, answered = [], 0
        for r in (rd.get("rows") or [])[:60]:
            row = r.get("row") or {}
            if chatcol:
                q, a = chat_qa(row.get(chatcol))
                if q:
                    out.append(f"{cell(q[:4000])},{cell(a[:2000])}")
                    if a.strip():
                        answered += 1
                continue
            q = row.get(qcol)
            a = row.get(acol) if acol else ""
            texts, labs = None, []
            ch = row.get(ccol) if ccol else None
            if isinstance(ch, dict) and isinstance(ch.get("text"), list):   # ARC style {text:[], label:[]}
                texts, labs = [str(x) for x in ch["text"]], [str(x) for x in (ch.get("label") or [])]
            elif isinstance(ch, list) and ch:
                texts = [str(x) for x in ch]
            if texts:      # multiple choice: options go into the question, the right option's text is the answer
                if isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(texts):
                    a = texts[a]
                elif isinstance(a, str):
                    s = a.strip()
                    if labs and s in labs:
                        a = texts[labs.index(s)]
                    elif s.isdigit() and int(s) < len(texts):
                        a = texts[int(s)]
                    elif len(s) == 1 and s.upper() in LETTERS[:len(texts)]:
                        a = texts[LETTERS.index(s.upper())]
                opts = "  ".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts[:len(LETTERS)]))
                q = f"{q}  Options: {opts}  Answer with the text of the correct option."
            elif isinstance(a, bool):    # boolq style: yes/no reads better than True/False
                q = f"{q}{'' if str(q).rstrip().endswith('?') else '?'} Answer yes or no."
                a = "yes" if a else "no"
            elif isinstance(a, dict):    # SQuAD style {"text": [...], "answer_start": [...]}
                a = (a.get("text") or [""])[0] if isinstance(a.get("text"), list) else str(a)
            elif isinstance(a, list):
                a = a[0] if a else ""
            elif labels is not None and isinstance(a, int) and 0 <= a < len(labels):
                a = labels[a]
            if q:
                a = str(a if a is not None else "")
                a = re.sub(r"<<[^>]*>>", "", a)      # gsm8k-style calculator annotations
                if "####" in a:                       # gsm8k marks the final answer after ####
                    a = a.split("####")[-1].strip()
                if a.strip():
                    answered += 1
                out.append(f"{cell(q)},{cell(a)}")
        return out, answered, total, bool(acol or chatcol)
    # split order: a /viewer deep link wins; otherwise eval splits first. If the preferred
    # split turns out unlabeled (hellaswag's hidden test answers), fall through to one that
    # actually has answers rather than importing an answerless quiz.
    order = ([want_split] if want_split else []) + ["test", "validation", "train"]
    picks, seen = [], set()
    for s in order:
        for e in entries:
            if e.get("split") == s and (e["config"], e["split"]) not in seen:
                picks.append(e); seen.add((e["config"], e["split"])); break
    if entries and (entries[0]["config"], entries[0]["split"]) not in seen:
        picks.append(entries[0])
    best = None
    for pick in picks:
        out, answered, total, has_answers = build(pick)
        if best is None:
            best = (out, total, pick)
        if not has_answers or answered >= max(1, len(out) // 2):
            best = (out, total, pick)
            break
    out, total, pick = best
    if not out:
        raise ValueError("no usable rows in that dataset's first 100 entries")
    name = did.split("/")[-1] + (f" ({pick['config']})" if len({e.get('config') for e in sp.get('splits') or []}) > 1 else "")
    return "\n".join(["question,answer"] + out), name, total


def fetch_remote(url, token=None):
    """Resolve a GitHub/GitLab/Gist link to raw test content. Handles direct file
    links (blob->raw, incl. #line and ?plain= junk), repo subfolders (/tree/<ref>/<dir>),
    bare repos (auto-find), and gists. Returns (text, resolved_url)."""
    u = url.strip().split("#", 1)[0].rstrip("/")          # drop #L10 fragment + trailing slash
    low = u.lower()

    # ---- GitHub ----
    if "gist.github.com/" in low:
        return _gist_pick(u.split("/")[-1], token)
    if "raw.githubusercontent.com/" in low:
        return _http_get(u.split("?", 1)[0], token), u
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$', u)        # file
    if m:
        path = m.group(4).split("?", 1)[0]
        raw = f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}/{path}"
        return _http_get(raw, token), raw
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.*))?$', u)   # folder / branch
    if m:
        return _github_pick(m.group(1), m.group(2), token, branch=m.group(3), path=(m.group(4) or ""))
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$', u)               # bare repo
    if m:
        return _github_pick(m.group(1), m.group(2), token)

    # ---- GitLab ----
    if "/-/blob/" in u:
        return _http_get(u.replace("/-/blob/", "/-/raw/").split("?", 1)[0], token), u
    if "/-/raw/" in u:
        return _http_get(u.split("?", 1)[0], token), u
    m = re.match(r'https?://([^/]+)/(.+?)/-/tree/([^/]+)(?:/(.*))?$', u)               # gitlab folder
    if m:
        return _gitlab_pick(m.group(1), m.group(2), token, branch=m.group(3), path=(m.group(4) or ""))
    m = re.match(r'https?://([^/]+)/(.+?)(?:\.git)?$', low)                            # gitlab bare repo
    if m and "gitlab" in m.group(1):
        gm = re.match(r'https?://([^/]+)/(.+?)(?:\.git)?$', u)
        return _gitlab_pick(gm.group(1), gm.group(2), token)

    return _http_get(u, token), u                                                     # assume already raw


# ---------- datasets (built-in + user-saved "cards") ----------
DATASETS_FILE = os.path.join(HERE, "datasets.json")

BUILTIN_DATASETS = [
    {"id": "general", "name": "General knowledge",
     "desc": "Everyday facts: capitals, science, history.", "csv":
"""question,answer
What is the capital of Australia?,Canberra
Who wrote the novel 1984?,Orwell
What is the chemical symbol for gold?,Au
What is the largest planet in our solar system?,Jupiter
How many continents are there on Earth?,seven
In what year did World War II end?,1945
What gas do plants take in from the air?,carbon dioxide
What is the tallest mountain on Earth?,Everest"""},
    {"id": "logic", "name": "Logic & reasoning",
     "desc": "Short puzzles with one definite answer.", "csv":
"""question,answer
A bat and a ball cost $1.10 and the bat costs $1.00 more than the ball. How many cents is the ball?,5
A farmer has 17 sheep and all but 9 run away. How many are left?,9
If every rose is a flower must every flower be a rose? Answer yes or no,no
What number comes next in the sequence 2 4 8 16 ?,32
Tom is taller than Sam and Sam is taller than Bill. Who is shortest?,Bill
Rearrange the letters CIFAIPC into the name of an ocean,Pacific
If it takes 5 machines 5 minutes to make 5 widgets how many minutes for 100 machines to make 100 widgets?,5"""},
    {"id": "finance", "name": "Banking & finance (math)",
     "desc": "Interest, returns and ratios. Answers are numbers.", "csv":
"""question,answer
What is 5% simple interest on $2000 for one year in dollars?,100
If an investment grows from $1000 to $1200 what is the percent return?,20
A loan of $5000 at 10% annual simple interest accrues how much interest in one year in dollars?,500
What is 15% of $80 in dollars?,12
If revenue is $500 and profit is $150 what is the profit margin as a percent?,30
An asset worth $2000 depreciates 10%. What is its new value in dollars?,1800
Using the rule of 72 about how many years to double money at 8% per year?,9"""},
    {"id": "reasoning", "name": "Reasoning challenge", "suggest": 512,
     "desc": "Multi-step word problems and traps - where a thinking model can pull ahead.", "csv":
"""question,answer
A shirt costs $40 and is discounted 25%. What is the sale price in dollars?,30
A bat and a ball cost $1.10 and the bat costs $1.00 more than the ball. How many cents is the ball?,5
Sara has 3 boxes of 12 apples and eats 5. How many apples are left?,31
A train travels 60 miles in 1.5 hours. What is its speed in miles per hour?,40
Tom is twice as old as Jerry and in 5 years their ages add to 40. How old is Jerry now?,10
If 8 workers build a wall in 6 hours how many hours do 12 workers need at the same rate?,4
A number doubled and then increased by 4 equals 20. What is the number?,8
There are 5 red and 3 blue balls in a bag. How many must you draw to be sure of two the same color?,3
What is the next number in the sequence 1 1 2 3 5 8?,13
A rectangle measures 8 by 5. What is its perimeter?,26"""},
    {"id": "stress", "name": "Stress test (hard mix)", "suggest": 512,
     "desc": "40 demanding questions across math logic science and trivia - built to push every model and rack up tokens.", "csv":
"""question,answer
A jacket costs $80 then is marked up 25% and then discounted 20%. What is the final price in dollars?,80
A car uses 6 liters of fuel per 100 km. How many liters does it need for 250 km?,15
You invest $1000 at 10% interest compounded annually. What is the total after 2 years in dollars?,1210
Three painters finish a house in 4 days. How many days would 6 painters need at the same rate?,2
The average of four numbers is 15 and three of them are 10 and 20 and 12. What is the fourth number?,18
A square has an area of 144. What is its perimeter?,48
What is the sum of all whole numbers from 1 to 100?,5050
What is 7 factorial?,5040
What is 2 raised to the power of 10?,1024
What is the next number in the sequence 2 4 8 16 32?,64
What is the next number in the sequence 1 4 9 16 25?,36
A pen and a notebook cost $1.50 together and the notebook costs $1.00 more than the pen. How many cents does the pen cost?,25
A farmer has 17 sheep and all but 9 die. How many sheep are left?,9
How many months of the year have at least 28 days?,12
A doctor gives you 3 pills to take one every 30 minutes. How many minutes pass until the last pill is taken?,60
If all Bloops are Razzies and all Razzies are Lazzies are all Bloops definitely Lazzies? Answer yes or no.,yes
What is the chemical symbol for sodium?,Na
How many bones are in the adult human body?,206
What is the pH of pure water?,7
What is the powerhouse of the cell?,mitochondria
What is the hardest natural material on Earth?,diamond
What is the largest organ of the human body?,skin
Which planet is known as the Red Planet?,Mars
What gas do plants absorb from the air for photosynthesis?,carbon dioxide
What is the capital city of Canada?,Ottawa
What is the capital city of Switzerland?,Bern
What is the capital city of Australia?,Canberra
In what year did World War II end?,1945
How many continents are there on Earth?,seven
What is the currency of Japan?,yen
How many sides does a hexagon have?,6
What is the Roman numeral for 49?,XLIX
What is the binary number 1011 in decimal?,11
Rearrange the letters of LISTEN to spell a word meaning quiet.,silent
In Python what does len of the string hello return?,5
A clock loses 5 minutes every hour. How many minutes behind is it after 12 real hours?,60
What is 15% of 15% of 2000?,45
What is the smallest prime number greater than 20?,23
How many degrees do the interior angles of a triangle add up to?,180
What word means the opposite of expand and starts with the letter c?,contract"""},
    {"id": "stump", "name": "Stump the 7-8B", "suggest": 512,
     "desc": "Multi-step math counting and base-conversion that small models trip on but a larger model can work through.", "csv":
"""question,answer
What is 17 times 23?,391
What is 23 times 19?,437
What is 13 times 17?,221
What is 47 times 12?,564
What is 111 times 11?,1221
What is 99 squared?,9801
What is 2 to the power of 9?,512
What is 2 to the power of 12?,4096
What is 7 factorial?,5040
What is 12 cubed?,1728
What is the square root of 729?,27
What is the sum of the first 10 prime numbers?,129
What is the sum of all whole numbers from 1 to 50?,1275
What is 1000 minus 7 then minus 7 then minus 7?,979
What is 365 times 24?,8760
How many seconds are in one day?,86400
Convert the binary number 101010 to decimal.,42
Convert the decimal number 255 to binary.,11111111
What is the smallest three digit number divisible by both 6 and 8?,120
What is the greatest common divisor of 48 and 36?,12
A rectangle has an area of 60 and a length of 12. What is its perimeter?,34
How many times does the letter r appear in the word strawberry?,3
How many times does the letter s appear in the word Mississippi?,4
How many vowels are in the word education?,5
How many days are in January February and March combined in a common year?,90
If today is Wednesday what day of the week will it be 100 days from now?,Friday
If today is Monday what day of the week will it be in 30 days?,Wednesday
Spell the word PYTHON backwards.,nohtyp
Spell the word GARDEN backwards.,nedrag
What is the next term in the look and say sequence 1 11 21 1211 111221?,312211"""},
    {"id": "typed-drills", "name": "Extraction & format drills", "suggest": 128,
     "desc": "Reply with ONLY the value, in the exact format asked. Graded by strict rules - a right answer wrapped in chatter still fails.", "csv":
'''question,__expected_1
"Read this line: ""Invoice INV-2041 was issued March 3 for $1180 to Novatech Ltd."" Reply with only the invoice number.",regex:(?i)^\\W*INV-2041\\W*$
"Read: ""Order 7 units of part BRK-88 at $12.50 each."" Reply with only the part code.",regex:(?i)^\\W*BRK-88\\W*$
"From ""Contact Maria Duarte at ext. 4417 before Friday"" reply with only the extension number.",regex:^\\W*4417\\W*$
"From ""Shipment weight: 480 kg; volume: 3.2 cubic meters"" reply with only the weight in kg as a bare number.",regex:^\\W*480\\W*$
"Classify the sentiment: ""The new dashboard is fast and the team loves it."" Reply with exactly one word: positive or negative or mixed.",regex:(?i)^\\W*positive\\W*$
"Classify the sentiment: ""The rollout broke logins and support tickets doubled."" Reply with exactly one word: positive or negative or mixed.",regex:(?i)^\\W*negative\\W*$
"Classify the sentiment: ""Great hardware but the setup wizard kept crashing."" Reply with exactly one word: positive or negative or mixed.",regex:(?i)^\\W*mixed\\W*$
"Is this message spam? ""WIN a FREE cruise - click now!!!"" Reply with exactly one word: yes or no.",regex:(?i)^\\W*yes\\W*$
"Convert the date March 3 2026 to YYYY-MM-DD format. Reply with only the date.",regex:^\\W*2026-03-03\\W*$
Write two hundred forty-seven as digits only.,regex:^\\W*247\\W*$
"Lowercase this exactly and reply with only the result: URGENT-Review-NOW",regex:^\\W*urgent-review-now\\W*$
Round 3.14159 to two decimal places and reply with only the number.,regex:^\\W*3\\.14\\W*$'''},
    {"id": "sum-memos", "name": "Brief the memo", "suggest": 256,
     "desc": "Compress a workplace memo to two or three sentences. Graded on whether every key decision, number and name survived.", "csv":
'''question,__expected_1,__expected_2,__expected_3
"Summarize this memo in two or three sentences, keeping every decision, number and name: The facilities team confirmed the Byron Street office move for Saturday June 14. Movers arrive at 7 am and the network cutover happens at noon, so staff must clear their desks by Friday 5 pm. The budget rose from $40,000 to $52,500 after elevator repairs were added to the scope. Anyone needing weekend access must email Priya Raman by Wednesday.",icontains:June 14,icontains:52,icontains:Priya
"Summarize this memo in two or three sentences, keeping every decision, number and name: Hiring stays frozen through the third quarter, with one exception approved on Monday: the platform team may fill two backend roles because the payments migration slipped. Recruiting for those roles is owned by Tomas Herrera. All other requisitions stay parked until the October review, and referral bonuses are paused as well.",icontains:two backend,icontains:Tomas,icontains:October
"Summarize this memo in two or three sentences, keeping every decision, number and name: The database migration window is set for Sunday March 9 from 2 am to 6 am. Customer traffic fails over to the Fairview replica, and order processing runs in read-only mode during the window. If checks fail, rollback is automatic at 5:30 am. Escalations go to Dana Okafor, not the general on-call channel.",icontains:March 9,icontains:Fairview,icontains:Dana
"Summarize this memo in two or three sentences, keeping every decision, number and name: Starting May 1, all client travel needs director approval when the trip exceeds $1,500. Bookings move from TravelPoint to the Concord portal, which averages 12 percent cheaper fares. Trips already booked keep their arrangements. Questions go to Lena Fischer in finance operations.",icontains:May 1,icontains:Concord,icontains:Lena
"Summarize this memo in two or three sentences, keeping every decision, number and name: Quality has recalled batch 4471 of the desk chargers after 9 units overheated in testing. Retailers were notified Tuesday and replacement stock ships within 15 business days. Customers get a prepaid return label plus a $25 credit. Refunds are handled by the support team, not the resellers.",icontains:4471,icontains:15 business,icontains:25
"Summarize this memo in two or three sentences, keeping every decision, number and name: The revenue target for the fourth quarter moves from $8.2 million to $9.1 million after the Meridian contract closed early. Marketing gets an extra $150,000 to support the push, and weekly pipeline reviews move from Fridays to Tuesdays. Team leads owe updated forecasts to Marcus Bell by end of month.",icontains:9.1,icontains:Tuesdays,icontains:Marcus
"Summarize this memo in two or three sentences, keeping every decision, number and name: We are switching packaging suppliers from Corrugate Plus to Bluepine Materials effective August 18. Unit cost drops 7 percent and defect rates in the trial fell by half. The old contract runs out its 60-day notice period, and open purchase orders with Corrugate Plus will still be honored. Supplier onboarding is led by Aisha Karim.",icontains:Bluepine,icontains:August 18,icontains:Aisha
"Summarize this memo in two or three sentences, keeping every decision, number and name: Annual security training is due by Friday September 26 and takes about 45 minutes. Anyone not finished by the deadline loses VPN access on Monday morning until it is complete. Contractors are included this year for the first time. Completion is tracked automatically, so no certificates need to be sent to Robert Ngata.",icontains:September 26,icontains:VPN,icontains:Contractors'''},
]


# ---- Document ingestion docsets: folders of page images + exact ground-truth fields ----
# Each dashboard/docsets/<id>/ holds page-N.png files and a manifest.json written by
# docsets/generate.py (with OCR text baked in by docsets/ocr_precompute.py). These are
# built-in cards of type "doc"; intake stays simple - no user upload path for pages.
DOCSETS_DIR = os.path.join(HERE, "docsets")


def _load_docsets():
    out = []
    try:
        for name in sorted(os.listdir(DOCSETS_DIR)):
            mp = os.path.join(DOCSETS_DIR, name, "manifest.json")
            if not os.path.isfile(mp):
                continue
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            if (m.get("type") or "doc") == "ingest":
                continue   # parser-bench sets belong to the tiering project, not Bake-off
            m["_dir"] = name   # folder under docsets/, also the public URL segment
            m["type"] = "doc"
            out.append(m)
    except Exception:  # noqa: BLE001
        pass
    out.sort(key=lambda m: (m.get("order") or 99, m.get("name") or ""))
    return out


BUILTIN_DOCSETS = _load_docsets()

# ---- Task-type registry ----
# The one place a kind of task is defined. Both task lists (the wizard picker and
# Configure > Tasks) build their sections, badges, and lineup requirements from this,
# so adding a type here - and serving datasets that carry its id - grows the UI a
# new section with no client changes. "models" names the lineup that qualifies.
TASK_TYPES = [
    {"id": "qa", "label": "Q&A", "section": "Q&A tasks",
     "tip": "Questions with graded answers - a right answer anywhere in the reply counts", "models": "text",
     "noun": "graded questions"},
    {"id": "doc", "label": "Docs", "section": "Document ingestion",
     "tip": "Page images graded on known-correct fields", "models": "vision"},
]

# First-open curation: these built-ins lead their sections; the rest wait behind
# "Show all". Cards the user saves themselves are always shown.
FEATURED_DATASETS = {"general", "logic", "finance", "reasoning", "typed-drills", "sum-memos",
                     "docs-invoices", "docs-forms", "docs-tables"}

# ---------- self-update + uninstall (Configure > General) ----------
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
VERSION_FILE = os.path.join(REPO_ROOT, "VERSION")   # stamped into release tarballs, absent in source checkouts
_PUBLIC_RAW = "https://raw.githubusercontent.com/jdbarzy/bake-off/HEAD/"
_UPDATE_CACHE = {"ts": 0.0, "latest": "", "notes": ""}


def _installed_version():
    try:
        with open(VERSION_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:  # noqa: BLE001
        return ""


# ---- one-click push enrollment: paste an ssh address, the dashboard drives the whole
#      onboarding over ssh - the product answer to labs where the box cannot phone home ----
PUSHENROLL = {"running": False, "addr": "", "steps": [], "error": None, "keyauth": False,
              "added": [], "transport": ""}
_PE_STEPS = [("connect", "Connect over ssh"), ("inspect", "Inspect GPUs"),
             ("runtime", "Install serving runtime"), ("agents", "Start control agents"),
             ("wire", "Wire the connection"), ("register", "Register machines")]


def _pe_set(sid, state, detail=""):
    for s in PUSHENROLL["steps"]:
        if s["id"] == sid:
            s["state"], s["detail"] = state, detail[:220]


def _pe_ssh(addr, cmd, timeout=30):
    """One remote command, key-auth only (BatchMode never prompts). -> (rc, out+err)."""
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        "-o", "StrictHostKeyChecking=accept-new", addr, cmd],
                       capture_output=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).decode(errors="replace")


def _pe_used_local_ports():
    used = set()
    for m in MACHINES:
        try:
            used.add(int(urllib.parse.urlparse(m["base"]).port or 0))
        except Exception:  # noqa: BLE001
            pass
    try:
        out = subprocess.run(["ss", "-ltn"], capture_output=True, timeout=5).stdout.decode()
        used |= {int(x.rsplit(":", 1)[-1]) for x in re.findall(r"[\d.\[\]:*]+:\d+", out)}
    except Exception:  # noqa: BLE001
        pass
    return used


def _push_enroll(addr, name):
    try:
        # connect: the only manual prerequisite is key auth (the modal shows ssh-copy-id)
        _pe_set("connect", "active")
        rc, out = _pe_ssh(addr, "true")
        if rc != 0:
            if "Permission denied" in out:
                PUSHENROLL["keyauth"] = True
                raise RuntimeError("key auth is not set up yet")
            raise RuntimeError(f"cannot reach {addr}: {out.strip().splitlines()[-1] if out.strip() else 'no route'}")
        _pe_set("connect", "ok")

        # inspect: GPUs + disk; no GPU is a hard stop with the real output shown
        _pe_set("inspect", "active")
        rc, out = _pe_ssh(addr, "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits")
        gpus = []
        for ln in out.strip().splitlines():
            try:
                i, gname, mem = [x.strip() for x in ln.split(",", 2)]
                gpus.append({"i": int(i), "name": gname, "vram_mb": int(float(mem))})
            except Exception:  # noqa: BLE001
                pass
        if rc != 0 or not gpus:
            raise RuntimeError(f"no usable GPU: {out.strip()[:160] or 'nvidia-smi not found'}")
        _pe_set("inspect", "ok", " + ".join(f"{g['name']} {g['vram_mb'] // 1024}GB" for g in gpus))

        # runtime: reuse any venv that already imports vllm; else uv-install one.
        # ninja/cmake ALWAYS ensured - JIT-kernel models die without them (field finding 2)
        _pe_set("runtime", "active", "checking for an existing vLLM…")
        rc, out = _pe_ssh(addr, 'for p in ~/vllm-env ~/mwboot-vllm; do "$p/bin/python3" -c "import vllm" 2>/dev/null && { echo "$p"; break; }; done')
        venv = out.strip().splitlines()[-1].strip() if rc == 0 and out.strip() else ""
        if not venv or not venv.startswith("/"):
            _pe_set("runtime", "active", "installing vLLM (this takes a few minutes)…")
            rc, out = _pe_ssh(addr, 'command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; '
                                    '~/.local/bin/uv venv ~/mwboot-vllm --python 3.12 >/dev/null 2>&1; '
                                    '~/.local/bin/uv pip install -p ~/mwboot-vllm/bin/python3 -q vllm ninja cmake && echo VENV-OK', timeout=1500)
            if "VENV-OK" not in out:
                raise RuntimeError(f"vLLM install failed: {out.strip()[-200:]}")
            venv = "~/mwboot-vllm"
        else:
            _pe_ssh(addr, f'~/.local/bin/uv pip install -p {venv}/bin/python3 -q ninja cmake 2>/dev/null || {venv}/bin/python3 -m pip install -q ninja cmake 2>/dev/null || true', timeout=300)
        _pe_set("runtime", "ok", venv)

        # transport probe + port allocation: direct if the box can call us, tunneled otherwise
        _pe_set("wire", "active", "probing reachability…")
        my_ip = ""
        try:
            my_ip = subprocess.run(["hostname", "-I"], capture_output=True, timeout=5).stdout.decode().split()[0]
        except Exception:  # noqa: BLE001
            pass
        rc, _out = _pe_ssh(addr, f"curl -s -o /dev/null --max-time 5 http://{my_ip}:{PORT}/ && echo REACH") if my_ip else (1, "")
        direct = rc == 0 and "REACH" in _out
        PUSHENROLL["transport"] = "direct" if direct else "tunneled"
        # sticky ports: a box that already runs our agents keeps its ports, so re-enrolling
        # upserts instead of duplicating (and never orphans models currently serving)
        rc, out = _pe_ssh(addr, 'pgrep -af "mwboot-control.py --port" | grep -oE -- "--vllm-port [0-9]+ --gpu [0-9]+" || true')
        sticky = {}   # gpu index -> vllm port
        for vp, gi in re.findall(r"--vllm-port (\d+) --gpu (\d+)", out or ""):
            sticky[int(gi)] = int(vp)
        rc, out = _pe_ssh(addr, "ss -ltn 2>/dev/null | grep -oE ':[0-9]+ ' | tr -d ': '")
        remote_used = {int(x) for x in out.split()} if rc == 0 else set()
        local_used = _pe_used_local_ports()
        ports, p = [], 8005
        for g in gpus:
            if g["i"] in sticky:
                ports.append(sticky[g["i"]])
                continue
            while True:
                if p not in remote_used and (p + 3499) not in remote_used and p not in ports and \
                   (direct or (p not in local_used and (p + 3499) not in local_used)):
                    ports.append(p)
                    p += 1
                    break
                p += 1

        # agents: one per GPU in tmux mwctl, venv bin leading PATH (field finding 3)
        _pe_set("agents", "active")
        subprocess.run(["scp", "-q", os.path.join(REPO_ROOT, "mwboot-control.py"), f"{addr}:~/mwboot-control.py"],
                       capture_output=True, timeout=30)
        parts = " & ".join(f"python3 ~/mwboot-control.py --port {pp + 3499} --vllm-port {pp} --gpu {g['i']} --max-len 16384"
                           for g, pp in zip(gpus, ports))
        launch = (f'export PATH={venv}/bin:$PATH MWBOOT_VLLM_PYTHON={venv}/bin/python3 '
                  f'MWBOOT_VLLM_ARGS="--enforce-eager --trust-remote-code"; {parts} & wait')
        rc, out = _pe_ssh(addr, f"tmux kill-session -t mwctl 2>/dev/null; tmux new-session -d -s mwctl '{launch}' && echo AGENTS-OK")
        if "AGENTS-OK" not in out:
            raise RuntimeError(f"agent launch failed: {out.strip()[-160:]}")
        _pe_set("agents", "ok", f"{len(gpus)} agent(s)")

        # tunneled boxes get a persistent per-machine ssh tunnel unit with localhost bases
        host = addr.split("@")[-1]
        if direct:
            bases = [f"http://{host}:{pp}/v1" for pp in ports]
            _pe_set("wire", "ok", "box reaches the dashboard - direct connection")
        else:
            # slug from the HOST, never the display name: renaming a machine must not
            # spawn a second tunnel unit fighting over the same forwarded ports
            slug = re.sub(r"[^a-z0-9-]+", "-", host.lower()).strip("-") or "box"
            fwd = " ".join(f"-L {pp}:localhost:{pp} -L {pp + 3499}:localhost:{pp + 3499}" for pp in ports)
            unit = os.path.expanduser(f"~/.config/systemd/user/llm-bench-tunnel-{slug}.service")
            os.makedirs(os.path.dirname(unit), exist_ok=True)
            with open(unit, "w", encoding="utf-8") as f:
                f.write("[Unit]\nDescription=llm-bench SSH tunnel -> " + addr + "\n"
                        "After=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\n"
                        "ExecStart=/usr/bin/ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "
                        "-o ServerAliveCountMax=3 " + fwd + " " + addr + "\nRestart=always\nRestartSec=5\n\n"
                        "[Install]\nWantedBy=default.target\n")
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=15)
            subprocess.run(["systemctl", "--user", "enable", "--now", os.path.basename(unit)], capture_output=True, timeout=15)
            time.sleep(2)
            subprocess.run(["systemctl", "--user", "restart", os.path.basename(unit)], capture_output=True, timeout=15)
            time.sleep(2)
            bases = [f"http://localhost:{pp}/v1" for pp in ports]
            _pe_set("wire", "ok", "box cannot call the dashboard - private tunnel created")

        # verify an agent answers through the chosen transport before registering anything
        deadline = time.time() + 30
        alive = False
        while time.time() < deadline and not alive:
            try:
                with urllib.request.urlopen(_control_url(bases[0]) + "/control/status", timeout=3) as r:
                    alive = r.status == 200
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if not alive:
            raise RuntimeError("agents started but are not answering through the connection")

        # register: upsert by base - re-enrolling the same box never duplicates
        _pe_set("register", "active")
        label = (name or host)[:48]
        by_base = {m["base"].rstrip("/"): m for m in MACHINES}
        added = []
        for g, base in zip(gpus, bases):
            mname = f"{label} (GPU {g['i']})" if len(gpus) > 1 else label
            row = by_base.get(base.rstrip("/"))
            if row:
                row.update(name=mname, vram_mb=g["vram_mb"], gpu=g["name"], agent=True, ssh=addr)
            else:
                MACHINES.append({"id": f"mach-{int(time.time())}-{g['i']}", "name": mname, "base": base,
                                 "key": "", "ckey": "", "cport": None, "vram_mb": g["vram_mb"],
                                 "gpu": g["name"], "agent": True, "kind": "openai", "ssh": addr})
            added.append(mname)
        _save_machines()
        PUSHENROLL["added"] = added
        _pe_set("register", "ok", ", ".join(added))
    except Exception as e:  # noqa: BLE001
        PUSHENROLL["error"] = str(e)[:300]
        for s in PUSHENROLL["steps"]:
            if s["state"] == "active":
                s["state"] = "fail"
                s["detail"] = str(e)[:220]
    finally:
        PUSHENROLL["running"] = False


# ---- unenroll: the reverse of push enrollment. Frees the GPUs, removes what we put on
#      the box, removes the tunnel, forgets the machines. Weights cache stays (harmless,
#      and re-enrolling later reuses it) - mirrors the RAG-teardown keep-the-cache call. ----
UNENROLL = {"running": False, "addr": "", "steps": [], "error": None, "removed": []}
_UE_STEPS = [("unload", "Free the GPUs"), ("cleanbox", "Remove Bake-off files from the box"),
             ("tunnel", "Remove the private tunnel"), ("deregister", "Forget the machines")]


def _ue_set(sid, state, detail=""):
    for s in UNENROLL["steps"]:
        if s["id"] == sid:
            s["state"], s["detail"] = state, detail[:220]


def _unenroll(addr, targets):
    try:
        _ue_set("unload", "active")
        freed = 0
        for m in targets:
            try:
                req = urllib.request.Request(_control_url(m["base"]) + "/control/unload",
                                             data=b"{}", headers=_control_headers(m["base"]), method="POST")
                with urllib.request.urlopen(req, timeout=20) as r:
                    freed += 1 if r.status == 200 else 0
            except Exception:  # noqa: BLE001
                pass
        _ue_set("unload", "ok", f"{freed}/{len(targets)} agents unloaded")

        _ue_set("cleanbox", "active")
        rc, out = _pe_ssh(addr, "tmux kill-session -t mwctl 2>/dev/null; rm -f ~/mwboot-control.py; "
                                "rm -rf ~/mwboot-vllm; echo CLEAN-OK", timeout=60)
        if "CLEAN-OK" not in out:
            raise RuntimeError(f"could not clean the box: {out.strip()[-160:] or 'ssh failed'}")
        _ue_set("cleanbox", "ok", "agents stopped, Bake-off files removed (model weights cache kept)")

        _ue_set("tunnel", "active")
        host = addr.split("@")[-1]
        slug = re.sub(r"[^a-z0-9-]+", "-", host.lower()).strip("-") or "box"
        unit = os.path.expanduser(f"~/.config/systemd/user/llm-bench-tunnel-{slug}.service")
        if os.path.exists(unit):
            subprocess.run(["systemctl", "--user", "disable", "--now", os.path.basename(unit)],
                           capture_output=True, timeout=15)
            os.remove(unit)
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=15)
            _ue_set("tunnel", "ok", "tunnel stopped and removed")
        else:
            _ue_set("tunnel", "ok", "no tunnel (direct connection)")

        _ue_set("deregister", "active")
        keep = [m for m in MACHINES if m not in targets]
        apply_machines(keep)
        UNENROLL["removed"] = [m["name"] for m in targets]
        _ue_set("deregister", "ok", ", ".join(UNENROLL["removed"]))
    except Exception as e:  # noqa: BLE001
        UNENROLL["error"] = str(e)[:300]
        for s in UNENROLL["steps"]:
            if s["state"] == "active":
                s["state"] = "fail"
                s["detail"] = str(e)[:220]
    finally:
        UNENROLL["running"] = False


def start_unenroll(mid):
    if UNENROLL["running"]:
        return False, "an unenroll is already running"
    m = machine_by_id(mid)
    if not m:
        return False, "unknown machine"
    addr = m.get("ssh")
    if not addr:
        return False, "this machine was not enrolled over ssh - use Delete to just forget it"
    targets = [x for x in MACHINES if x.get("ssh") == addr]
    UNENROLL.update(running=True, addr=addr, error=None, removed=[],
                    steps=[{"id": i, "label": lb, "state": "pending", "detail": ""} for i, lb in _UE_STEPS])
    threading.Thread(target=_unenroll, args=(addr, targets), daemon=True).start()
    return True, None


def start_push_enroll(addr, name):
    if PUSHENROLL["running"]:
        return False
    PUSHENROLL.update(running=True, addr=addr, error=None, keyauth=False, added=[], transport="",
                      steps=[{"id": i, "label": lb, "state": "pending", "detail": ""} for i, lb in _PE_STEPS])
    threading.Thread(target=_push_enroll, args=(addr, name), daemon=True).start()
    return True


def _run_detached(cmd, tag):
    """Run a maintenance command that will stop this very server (update / uninstall).
    On Linux it must live OUTSIDE the service's cgroup, or systemd kills it mid-flight
    the moment the unit stops - so prefer a transient systemd-run unit, plain detach
    otherwise. On Windows (jump-host installs) detach a powershell instead."""
    log = os.path.join(REPO_ROOT, tag + ".log")
    if os.name == "nt":
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                          f"Start-Sleep 2; {cmd} *>> '{log}'"],
                         creationflags=0x00000208, cwd=REPO_ROOT)   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        return
    shell = f"sleep 2; {{ {cmd} ; }} >> {shlex.quote(log)} 2>&1"
    if shutil.which("systemd-run"):
        subprocess.Popen(["systemd-run", "--user", "--collect",
                          f"--unit=bake-off-{tag}-{int(time.time())}", "bash", "-c", shell])
    else:
        subprocess.Popen(["bash", "-c", shell], start_new_session=True, cwd=REPO_ROOT)


def _docset_rows(d):
    """Turn a docset manifest into run rows. __image is the file to send to the model;
    __image_url is what the browser shows; __ocr is the precomputed OCR text baseline."""
    rows = []
    for r in d.get("rows") or []:
        rows.append({"question": r["question"],
                     "__label": r.get("label") or r["file"],
                     "__image": os.path.join(DOCSETS_DIR, d["_dir"], r["file"]),
                     "__image_url": f"/docsets/{d['_dir']}/{r['file']}",
                     "__expected_fields": r.get("fields") or {},
                     "__ocr": r.get("ocr") or "",
                     "__parse": r.get("parse") or ""})
    return rows


_img_b64_cache = {}


def _img_b64(path):
    if path not in _img_b64_cache:
        with open(path, "rb") as f:
            _img_b64_cache[path] = base64.b64encode(f.read()).decode()
    return _img_b64_cache[path]


def _load_datasets_store():
    """Backward-compatible: old file is a bare list of saved datasets; new file is
    {saved:[...], hidden:[...], deleted:[...], used:{id: ts}} so built-in cards can be
    deleted (hidden) and restored, saved cards keep a trash you can empty, and every
    card remembers when it was last loaded (for the recently-used sort)."""
    try:
        with open(DATASETS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, [], [], {}, [], []
        return (data.get("saved") or [], data.get("hidden") or [],
                data.get("deleted") or [], data.get("used") or {},
                data.get("purged") or [], data.get("pinned") or [])
    except Exception:  # noqa: BLE001
        return [], [], [], {}, [], []


(SAVED_DATASETS, HIDDEN_DATASETS, DELETED_DATASETS, DATASET_USED,
 PURGED_DATASETS, PINNED_DATASETS) = _load_datasets_store()


def _save_datasets():
    _jdump(DATASETS_FILE, {"saved": SAVED_DATASETS, "hidden": HIDDEN_DATASETS,
                           "deleted": DELETED_DATASETS, "used": DATASET_USED,
                           "purged": PURGED_DATASETS, "pinned": PINNED_DATASETS})


def _all_datasets():
    hidden = set(HIDDEN_DATASETS)
    return ([{**d, "builtin": True} for d in BUILTIN_DATASETS if d["id"] not in hidden]
            + [{**d, "builtin": True} for d in BUILTIN_DOCSETS if d["id"] not in hidden]
            + SAVED_DATASETS)


def _deleted_datasets():
    """What the trash holds: deleted saved cards plus hidden built-ins (restorable).
    Purged built-ins stay hidden but leave the trash - emptied means emptied."""
    hidden = set(HIDDEN_DATASETS) - set(PURGED_DATASETS)
    return (list(DELETED_DATASETS)
            + [{**d, "builtin": True} for d in BUILTIN_DATASETS + BUILTIN_DOCSETS
               if d["id"] in hidden])


def _find_dataset(did):
    return next((d for d in _all_datasets() if d.get("id") == did), None)


def _rule_text(v):
    """One expected value as a human line for previews. Our anchored strict patterns
    read as 'exactly: X'; anything else regex shows as the rule it is."""
    typ, _, val = v.partition(":") if ":" in v else ("icontains", "", v)
    typ = typ.strip().lower()
    if typ == "regex":
        m = re.match(r"^\(\?i\)?\^\\W\*(.+?)\\W\*\$$", val.strip())
        if m:
            return "exactly: " + re.sub(r"\\(.)", r"\1", m.group(1))
        return "matches: " + val
    if typ == "equals":
        return "exactly: " + val
    return val


def _row_answer(r):
    parts = [_rule_text(v) for k, v in r.items() if k.startswith("__expected") and v]
    if len(parts) > 1:   # several expecteds = ALL must appear (key-point coverage)
        return "must include: " + " + ".join(p for p in parts)
    return parts[0] if parts else ""


def _suggest_tokens(rows, graded):
    """Pick a sensible answer-length default from the dataset's own content:
    open-ended sets and longer prompts need more room; short graded Q&A need less."""
    if not rows:
        return 256
    avg_q = sum(len(r.get("question", "")) for r in rows) / len(rows)
    if not graded:                       # open-ended: answers tend to run long
        if avg_q > 400:
            return 1024
        if avg_q > 150:
            return 512
        return 256
    return 512 if avg_q > 300 else 256   # graded: usually short exact answers, bump up for long prompts


def _preview_rows(rows, n=500):
    graded = any(any(k.startswith("__expected") for k in r) for r in rows)
    sample = [{"q": r["question"], "a": _row_answer(r)} for r in rows[:n]]
    return {"count": len(rows), "graded": graded, "sample": sample,
            "suggest": _suggest_tokens(rows, graded)}


def _dataset_meta(d):
    stamps = {"created": d.get("created") or 0, "used": DATASET_USED.get(d["id"], 0),
              # curation: user-saved cards always lead; built-ins lead only when featured
              "featured": (not d.get("builtin", d.get("type") == "doc")) or d["id"] in FEATURED_DATASETS}
    if d.get("type") == "doc":
        rows = d.get("rows") or []
        nf = sum(len(r.get("fields") or {}) for r in rows)
        return {"id": d["id"], "name": d["name"], "desc": d.get("desc", ""), "type": "doc",
                "count": len(rows), "fields": nf, "graded": True, "builtin": True, **stamps}
    rows, _ = parse_questions(d["csv"])
    rows = rows or []
    graded = any(any(k.startswith("__expected") for k in r) for r in rows)
    return {"id": d["id"], "name": d["name"], "desc": d.get("desc", ""),
            "type": d.get("dtype") or "qa",   # text-kind subtypes (typed, sum) section by the registry
            "count": len(rows), "graded": graded, "builtin": d.get("builtin", False), **stamps}


def grade(row, out):
    exps = [v for k, v in row.items() if k.startswith("__expected") and v]
    if not exps:
        return None
    low = out.lower()
    for e in exps:
        typ, _, val = e.partition(":") if ":" in e else ("icontains", "", e)
        typ = typ.strip().lower()
        val = val.strip()
        if typ in ("icontains", "contains"):
            if val.lower() not in low:
                return False
        elif typ == "equals":
            if out.strip() != val:
                return False
        elif typ == "regex":
            if not re.search(val, out):
                return False
        else:
            if val.lower() not in low:
                return False
    return True


def _squash(s):
    """Normalize for value matching: lowercase, spell & as and, drop spaces, commas,
    currency signs, parens, hyphens, and periods. Applied to BOTH sides, so
    $1,240,000 matches 1240000 and (503) 555-0187 matches 503-555-0187."""
    return re.sub(r"[\s,$()\-.]+", "", (s or "").lower().replace("&", "and"))


def field_hit(val, out):
    """Is this exact ground-truth field value present in the model's answer?
    Case-insensitive contains, with a squashed fallback so $1,284.50 also
    matches 1284.50 or 1,284.50. Deterministic - no judge model."""
    v = (val or "").strip().lower()
    o = (out or "").lower()
    if not v:
        return True
    if v in o:
        return True
    sv = _squash(val)
    return len(sv) >= 2 and sv in _squash(out)


def grade_fields(fields, out):
    """Per-field correctness for a document row: {field name: True/False}."""
    return {k: field_hit(v, out) for k, v in (fields or {}).items()}


# ---------- model calls ----------
def count_tokens(m, text):
    """Count tokens with the *model's own* tokenizer, used only when the serving
    engine didn't report usage. vLLM exposes /tokenize beside /v1 (each model loads
    its own tokenizer); engines without it fall back to an estimate."""
    if not text or m is None:
        return None
    root = m["base"].rsplit("/v1", 1)[0]   # vLLM /tokenize sits beside /v1, not under it
    try:
        body = json.dumps({"model": m["model"], "prompt": text}).encode()
        req = urllib.request.Request(root + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(json.loads(r.read()).get("count", 0)) or None
    except Exception:  # noqa: BLE001
        return None


def approx_tokens(text):
    """Model-agnostic last resort (only if a model exposes neither usage nor a
    tokenizer): ~4 chars/token, floored at the word count."""
    return max(len(text.split()), round(len(text) / 4)) if text else 0


def call_model(m, question, max_tokens, think=False, image=None):
    """Streams the response so we can measure TTFT (time to first token = prompt
    processing / prefill latency). Returns (output, prompt_tokens, completion_tokens, ttft_ms).

    Reasoning models: by default we ask for the answer only (think off) so they grade
    fairly against non-reasoning models. When `think` is on we let them reason and add
    token headroom so they finish thinking AND still produce an answer.

    Document rows pass `image` (a PNG path), sent as an image_url content part
    (data URI)."""
    start = time.time()
    thinks = model_can_think(m)
    want_think = bool(think and thinks)
    if image:
        content = [{"type": "text", "text": question},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _img_b64(image)}}]
    else:
        content = question
    # Reasoning models on /v1 can't universally be told not to think, so any thinker
    # gets headroom - reasoning tokens must never starve the graded answer.
    budget = max_tokens + REASONING_HEADROOM if thinks else max_tokens
    payload = {"model": m["model"], "max_tokens": budget, "stream": True, "temperature": 0,
               "stream_options": {"include_usage": True},
               "messages": [{"role": "user", "content": content}]}
    if thinks and "qwen3" in (m.get("model") or "").lower():
        # Qwen3's chat template honors this; templates without the knob ignore it.
        payload["chat_template_kwargs"] = {"enable_thinking": want_think}
    auth = GATEWAY["bearer"] if _is_gateway(m) else ((machine_by_id(m.get("machine")) or {}).get("key") or "dummy")
    return _stream(m["base"] + "/chat/completions", payload, start, "openai", m=m, prompt_text=question, auth=auth)


def _stream(url, payload, start, kind, m=None, prompt_text="", auth=None, timeout=180):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = "Bearer " + auth
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    ttft = None
    parts = []
    ptok = ctok = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            ch = obj.get("choices") or []
            if ch:
                # reasoning deltas (servers with a reasoning parser) are not the answer:
                # they don't count toward TTFT and never reach grading.
                c = (ch[0].get("delta") or {}).get("content") or ""
                if c:
                    if ttft is None:
                        ttft = time.time() - start
                    parts.append(c)
            u = obj.get("usage")
            if u:
                ptok = int(u.get("prompt_tokens", 0))
                ctok = int(u.get("completion_tokens", 0))
    out = "".join(parts)
    # Engines without a reasoning parser stream <think>...</think> inside content;
    # grade only what comes after the reasoning block (TTFT already measured the
    # first visible token, which is the honest prefill signal either way).
    out = re.sub(r"^\s*<think>.*?</think>\s*", "", out, flags=re.S)
    if ttft is None:
        ttft = time.time() - start
    if ptok == 0:   # engine reported no usage: count with the model's own tokenizer
        ptok = count_tokens(m, prompt_text) or approx_tokens(prompt_text)
    if ctok == 0:
        ctok = count_tokens(m, out) or approx_tokens(out)
    return out, ptok, ctok, round(ttft * 1000)


# ---------- worker loop ----------
def _elapsed_running():
    """Seconds the run has actually been running (excludes paused time)."""
    if not STATE["started_at"]:
        return 0.0
    now = time.time()
    paused_now = (now - STATE["pause_started"]) if (STATE["paused"] and STATE["pause_started"]) else 0
    return max(0.0, now - STATE["started_at"] - STATE["pause_total"] - paused_now)


def _finish_if_complete():
    """End the run once every model has finished its rounds."""
    with LOCK:
        if STATE["running"] and all(s["done"] for s in STATE["models"].values()):
            STATE["running"] = False
            STATE["owner"] = None            # run finished -> release the lock for the next tester
            STATE["ended_at"] = time.time()
            _save_lifetime()


def worker(m, gen):
    key = m["key"]
    live = lambda: STATE["running"] and STATE["gen"] == gen   # noqa: E731  this run, still going
    while live():
        if _elapsed_running() >= MAX_RUN_SECONDS:    # walk-away backstop: hard-stop the whole run
            with LOCK:
                STATE["running"] = False
                STATE["owner"] = None
                STATE["ended_at"] = time.time()
            break
        rows = list(STATE["rows"])
        if not rows:
            time.sleep(0.4)
            continue
        for row in rows:
            if not live():
                break
            while live() and STATE["paused"]:   # hold here while paused
                time.sleep(0.2)
            if not live():
                break
            t0 = time.time()
            try:
                out, ptok, ctok, ttft_ms = call_model(m, row["question"], STATE["max_tokens"], STATE["think"],
                                                      image=row.get("__image"))
                if STATE["gen"] != gen:          # a newer run started while we were blocked: discard, don't pollute it
                    return
                ms = (time.time() - t0) * 1000
                fields = row.get("__expected_fields")
                if fields:                       # document row: per-field ground truth, all-correct = a pass
                    hits = grade_fields(fields, out)
                    fok = sum(1 for v in hits.values() if v)
                    ok = fok == len(fields)
                else:
                    fok, ok = 0, grade(row, out)
                with LOCK:
                    s = STATE["models"][key]
                    s["requests"] += 1
                    s["prompt_tokens"] += ptok
                    s["completion_tokens"] += ctok
                    STATE["saved"][key] += ptok + ctok   # per-model session savings
                    s["total_latency_ms"] += ms
                    s["ttft_ms_total"] += ttft_ms
                    s["ttft_count"] += 1
                    if fields:
                        s["fields_ok"] += fok
                        s["fields_total"] += len(fields)
                    s["last_ms"] = int(ms)
                    s["last_input"] = ((row.get("__label") + " - ") if row.get("__label") else "") + row["question"][:2000]
                    s["last_output"] = out[:4000]
                    s["last_in_tokens"] = ptok
                    s["last_out_tokens"] = ctok
                    if ok is True:
                        s["passes"] += 1
                    elif ok is False:
                        s["fails"] += 1
                    else:
                        s["ungraded"] += 1
                add_lifetime(ptok + ctok)   # grand lifetime tally (persisted)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["models"][key]["errors"] += 1
                    STATE["models"][key]["last_error"] = str(e)[:200]
                time.sleep(0.5)
        if not live():                   # stopped/superseded mid-pass: don't count a partial round
            break
        with LOCK:
            STATE["models"][key]["loops"] += 1
            reached = STATE["models"][key]["loops"] >= STATE["rounds"]
            if reached:
                STATE["models"][key]["done"] = True
        if reached:                      # this model has done all its rounds
            _finish_if_complete()
            break


def start_run(max_tokens, rounds=None, think=False):
    stop_run()
    with LOCK:
        STATE["max_tokens"] = int(max_tokens or 256)
        STATE["rounds"] = max(1, int(rounds or SETTINGS.get("rounds") or MAX_ROUNDS))
        STATE["think"] = bool(think)
        STATE["gen"] += 1
        gen = STATE["gen"]
        STATE["models"] = {m["key"]: _blank_stats() for m in _active_models()}
        STATE["messages"] = []   # a fresh drive starts with a clean inbox (msg_seq keeps climbing so ids stay unique)
        STATE["started_at"] = time.time()
        STATE["ended_at"] = None
        STATE["paused"] = False
        STATE["pause_started"] = None
        STATE["pause_total"] = 0.0
        STATE["running"] = True
    for m in _active_models():
        threading.Thread(target=worker, args=(m, gen), daemon=True).start()


def pause_run():
    if STATE["running"] and not STATE["paused"]:
        STATE["pause_started"] = time.time()
        STATE["paused"] = True


def resume_run():
    if STATE["paused"]:
        if STATE["pause_started"]:
            STATE["pause_total"] += time.time() - STATE["pause_started"]
        STATE["pause_started"] = None
        STATE["paused"] = False


def stop_run():
    if STATE["running"]:
        resume_run()   # fold any open pause into pause_total before freezing
        STATE["ended_at"] = time.time()   # freeze the run clock at stop
    STATE["running"] = False
    STATE["owner"] = None
    STATE["paused"] = False
    _save_lifetime()   # flush the persisted tally
    time.sleep(0.1)


def reset_all():
    """Fresh start: stop, clear per-run stats + session savings + comparison,
    reload the default questions. The persisted lifetime tally is kept."""
    stop_run()
    with LOCK:
        STATE["models"] = {m["key"]: _blank_stats() for m in MODELS}
        STATE["saved"] = {m["key"]: 0 for m in MODELS}
        STATE["started_at"] = None
        STATE["ended_at"] = None
        STATE["paused"] = False
        STATE["pause_started"] = None
        STATE["pause_total"] = 0.0
        COMPARE.update(running=False, rows=[], agg=None, done=0, total=0, models=[], task="qa")


def _clear_results():
    """Wipe the previous run's per-model stats + timers so a stale report/download
    doesn't linger when a new dataset is loaded."""
    with LOCK:
        STATE["models"] = {m["key"]: _blank_stats() for m in MODELS}
        STATE["saved"] = {m["key"]: 0 for m in MODELS}
        STATE["started_at"] = None
        STATE["ended_at"] = None
        STATE["paused"] = False
        STATE["pause_started"] = None
        STATE["pause_total"] = 0.0


def apply_models_config(new_models):
    """Lay user-edited endpoint fields onto the three slots, persist them to
    models.config.json, and clear stale per-model results. Returns the new view.
    Guardrail: a no-op (nothing actually changed) leaves a playing run and its
    results untouched - e.g. saving the current lineup as a card mid-run."""
    by_key = {m["key"]: m for m in MODELS}
    changed = False
    for entry in (new_models or []):
        slot = by_key.get(entry.get("key"))
        if not slot:
            continue
        if slot.get("kind") != "openai":   # legacy configs: every slot is /v1 now
            slot["kind"] = "openai"
            changed = True
        for fld in ("label", "system", "base", "model", "machine"):
            v = entry.get(fld)
            if v is not None and str(v).strip() and str(v).strip() != (slot.get(fld) or ""):
                slot[fld] = str(v).strip()
                changed = True
    if not changed:
        return _models_editable()
    _save_models_config()
    if STATE["running"]:
        stop_run()
    _clear_results()
    return _models_editable()


# ---------- load a model onto a vLLM endpoint (restart via its mwboot-control agent) ----------
PULL = {"active": False, "model": "", "base": "", "status": "idle", "completed": 0, "total": 0, "done": False, "error": None}


def _control_url(base):
    """The mwboot-control agent sits next to its vLLM endpoint at port+3499
    (8000 -> 11499, 8002 -> 11501 ...), so one host can run an agent per GPU.
    MWBOOT_CONTROL_PORT overrides the offset convention for single-agent setups."""
    u = urllib.parse.urlparse(base if "://" in base else "http://" + base)
    env = os.environ.get("MWBOOT_CONTROL_PORT")
    mm = _mach_for_base(base)
    port = int(env) if env else ((mm or {}).get("cport") or (u.port or 8000) + 3499)   # enrolled boxes told us their agent port
    return f"{u.scheme or 'http'}://{u.hostname or 'localhost'}:{port}"


def _control_status(base):
    """The agent's status dict (served model, hub-cached models, prefetch progress), or
    None when the box has no control plane. One call answers 'can swap?' and 'what's warm?'."""
    try:
        with urllib.request.urlopen(_control_url(base) + "/control/status", timeout=2) as r:
            return json.loads(r.read()) if r.status == 200 else None
    except Exception:  # noqa: BLE001
        return None


def _has_control(base):
    """True if this vLLM box runs an mwboot-control plane, i.e. self-service model
    swap is allowed here. Absent = read-only: the served catalog is fixed by MLOps."""
    return _control_status(base) is not None


def _control_headers(base):
    """Enrolled machines require their control key on mutating agent calls."""
    h = {"Content-Type": "application/json"}
    mm = _mach_for_base(base)
    if mm and mm.get("ckey"):
        h["X-MWBoot-Key"] = mm["ckey"]
    return h


def _agent_post(base, path, payload, timeout):
    """POST a JSON payload to a box's mwboot-control agent and decode the JSON reply.
    Exceptions propagate - every call site keeps its own error semantics."""
    req = urllib.request.Request(_control_url(base) + path,
                                 data=json.dumps(payload).encode(),
                                 headers=_control_headers(base), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


PULLGEN = {"n": 0}   # bumps when a newer pick supersedes the watch on an in-flight load


def _vllm_load_worker(base, model, gen):
    """vLLM can't hot-load, so ask the box's mwboot-control plane to restart it
    with `model`. Maps the control plane's status onto the shared PULL state."""
    def stale():
        return gen != PULLGEN["n"]
    ctrl = _control_url(base)
    try:
        try:
            res = _agent_post(base, "/control/load", {"model": model}, timeout=10)
        except Exception:  # noqa: BLE001
            with LOCK:
                if not stale():
                    PULL["error"] = ("This model server can't swap models by itself. Start the loader on that "
                                     "machine with  python3 mwboot-control.py  (see mwboot.sh), then try again.")
            return
        if not res.get("ok"):
            with LOCK:
                if not stale():
                    PULL["error"] = res.get("error") or "the control plane refused the request"
            return
        deadline = time.time() + 1900
        while time.time() < deadline:
            if stale():
                return   # a newer pick took over this machine; its worker owns PULL now
            try:
                with urllib.request.urlopen(ctrl + "/control/status", timeout=8) as r:
                    st = json.loads(r.read())
            except Exception:  # noqa: BLE001
                time.sleep(2)
                continue
            if st.get("model") != model and st.get("loading"):
                time.sleep(2)   # the agent is momentarily on the outgoing model; wait for ours
                continue
            with LOCK:
                if stale():
                    return
                if st.get("error"):
                    PULL["error"] = str(st["error"])[:200]
                elif st.get("loading"):
                    PULL["status"] = f"restarting vLLM with {model}… (this can take a few minutes)"
            if st.get("error") or (st.get("ready") and st.get("model") == model):
                if st.get("ready"):
                    with LOCK:
                        if not stale():
                            PULL["status"] = "ready"
                break
            time.sleep(2)
        else:
            with LOCK:
                if not stale():
                    PULL["error"] = PULL["error"] or "vLLM did not come up in time"
    finally:
        with LOCK:
            if not stale():
                PULL["active"] = False
                PULL["done"] = True


def start_pull(base, model, kind="openai"):
    """Load `model` onto an endpoint: a vLLM restart via the box's control plane.
    Runs in the background; the client polls /run/pull/status. Newest pick wins:
    asking the SAME machine for a DIFFERENT model cancels the in-flight load."""
    b = (base or "").rstrip("/")
    with LOCK:
        if PULL["active"]:
            if PULL["base"] == b and PULL["model"] == model:
                return True   # same intent already in flight: attach
            if PULL["base"] != b:
                return False   # a different machine is mid-load; one transfer at a time
            PULLGEN["n"] += 1   # same machine, newer pick: supersede
        gen = PULLGEN["n"]
        PULL.update(active=True, model=model, base=b,
                    status="starting", completed=0, total=0, done=False, error=None)
    threading.Thread(target=_vllm_load_worker, args=(base, model, gen), daemon=True).start()
    return True


# ---- portable setup bundle: move an entire configuration between machines/labs, no files or git ----
def export_bundle():
    """Everything that makes 'what we do here' - machines, the lineup, the model library,
    saved question sets - as one self-contained dict. No secrets (no dashboard login, no SSH)."""
    return {
        "llm_bench_setup": 1,
        "models": [{k: m.get(k) for k in ("key",) + EDITABLE_FIELDS} for m in MODELS],
        # machines travel WITHOUT their API keys or control keys - no secrets in the file
        "machines": [{k: v for k, v in m.items() if k not in ("key", "ckey", "ssh")} for m in MACHINES],
        "library": LIBRARY,
        "datasets": {"saved": SAVED_DATASETS, "hidden": HIDDEN_DATASETS,
                     "deleted": DELETED_DATASETS, "used": DATASET_USED,
                     "pinned": PINNED_DATASETS},
        "settings": dict(SETTINGS),
    }


def import_bundle(data):
    """Apply an exported setup. Tolerant: applies whatever sections are present, ignores the rest."""
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("machines"), list) and data["machines"]:
        apply_machines(data["machines"])
    if isinstance(data.get("library"), list):
        LIBRARY[:] = [x for x in data["library"] if isinstance(x, dict) and x.get("id")]
        _save_library()
        _see_cache.clear()
    if isinstance(data.get("models"), list) and data["models"]:
        apply_models_config(data["models"])                       # updates the three slots + persists
    # legacy exports may still carry "presets" / "lineups" sections - ignored silently
    d = data.get("datasets")
    if isinstance(d, dict):
        SAVED_DATASETS[:] = d.get("saved") or []
        HIDDEN_DATASETS[:] = d.get("hidden") or []
        DELETED_DATASETS[:] = d.get("deleted") or []
        DATASET_USED.clear(); DATASET_USED.update(d.get("used") or {})
        PINNED_DATASETS[:] = d.get("pinned") or []
        _save_datasets()
    s = data.get("settings")
    if isinstance(s, dict):
        try:
            SETTINGS["rounds"] = max(1, min(MAX_ROUNDS, int(s.get("rounds") or SETTINGS["rounds"])))
            if s.get("text_baseline") in ("ocr", "parse", "both"):
                SETTINGS["text_baseline"] = s["text_baseline"]
            _save_settings()
        except Exception:  # noqa: BLE001
            pass
    return True


def _load_questions(rows, cols, name):
    """Set the active question set and clear any prior run's results."""
    if STATE["running"]:
        stop_run()
    with LOCK:
        STATE["rows"], STATE["columns"], STATE["csv_name"] = rows, cols, name
    _clear_results()


def stats_snapshot():
    now = time.time()
    if STATE["started_at"]:
        end = now if STATE["running"] else (STATE["ended_at"] or now)
        paused_now = (now - STATE["pause_started"]) if (STATE["paused"] and STATE["pause_started"]) else 0
        elapsed = max(0, end - STATE["started_at"] - STATE["pause_total"] - paused_now)   # excludes paused time
    else:
        elapsed = 0
    has_answers = any(any(k.startswith("__expected") for k in r) for r in STATE["rows"])
    task = "doc" if any(r.get("__image") for r in STATE["rows"]) else "qa"
    has_req = any(s["requests"] > 0 for s in STATE["models"].values())
    completed = (not STATE["running"]) and (STATE["ended_at"] is not None) and has_req   # a real run has finished
    think_by_key = {m["key"]: model_can_think(m) for m in _active_models()}   # compute outside LOCK (may hit /api/show once)
    see_by_key = {m["key"]: model_can_see(m) for m in _active_models()}
    out = {"running": STATE["running"], "paused": STATE["paused"], "elapsed": round(elapsed, 1),
           "max_tokens": STATE["max_tokens"], "csv_name": STATE["csv_name"], "rounds": STATE["rounds"],
           "num_rows": len(STATE["rows"]), "has_answers": has_answers, "completed": completed, "task": task,
           "think": STATE["think"], "lifetime_tokens": STATE["lifetime"], "models": []}
    with LOCK:
        for m in _active_models():
            s = STATE["models"].get(m["key"])
            if s is None:
                continue
            graded = s["passes"] + s["fails"]
            tot_tok = s["prompt_tokens"] + s["completion_tokens"]
            v = model_view(m)   # name/desc/size of the model ACTUALLY on this slot
            mm = machine_by_id(m.get("machine")) or _mach_for_base(m.get("base")) or {}
            hw = {"machine": m.get("system") or mm.get("name") or "",
                  "gpu": mm.get("gpu") or "", "vram_mb": mm.get("vram_mb") or 0,
                  "agent": bool(mm.get("agent")), "keyed": bool(mm.get("key"))}
            out["models"].append({
                "key": m["key"], "label": v["label"], "system": m["system"], "desc": v["desc"], "hw": hw,
                "color": m["color"], "model": m["model"], "params_b": v["params_b"], "specs": model_specs(m),
                "can_think": think_by_key[m["key"]], "can_see": see_by_key[m["key"]],
                "requests": s["requests"], "passes": s["passes"], "fails": s["fails"],
                "ungraded": s["ungraded"], "errors": s["errors"], "loops": s["loops"], "done": s["done"],
                "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
                "total_tokens": tot_tok,
                "field_acc": (100.0 * s["fields_ok"] / s["fields_total"]) if s["fields_total"] else None,
                "fields_ok": s["fields_ok"], "fields_total": s["fields_total"],
                "pass_rate": (100.0 * s["passes"] / graded) if graded else None,
                "avg_latency_ms": (s["total_latency_ms"] / s["requests"]) if s["requests"] else None,
                "tps": (s["completion_tokens"] / elapsed) if elapsed > 0 else 0.0,
                "saved_tokens": STATE["saved"][m["key"]],
                "avg_ttft_ms": (s["ttft_ms_total"] / s["ttft_count"]) if s["ttft_count"] else None,
                "prefill_tps": (s["prompt_tokens"] / (s["ttft_ms_total"] / 1000)) if s["ttft_ms_total"] > 0 else 0.0,
                "last_ms": s["last_ms"], "last_input": s["last_input"],
                "last_output": s["last_output"], "last_in_tokens": s["last_in_tokens"],
                "last_out_tokens": s["last_out_tokens"], "last_error": s["last_error"],
            })
    return out


# ---------- replacement scoring (current vs candidate) ----------
def similarity(a, b):
    """0..1: 1.0 on exact (normalized) match, else blend of char-ratio + token Jaccard.
    Lexical for now; swap in embedding cosine here when an embedding model is available."""
    a2 = " ".join(a.lower().split())
    b2 = " ".join(b.lower().split())
    if a2 == b2:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a2, b2).ratio()
    sa, sb = set(a2.split()), set(b2.split())
    jacc = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    return round(0.5 * ratio + 0.5 * jacc, 4)


COMPARE = {"running": False, "models": [], "done": 0,
           "total": 0, "rows": [], "agg": None, "max_tokens": 256, "task": "qa"}

SIM_AGREE = 50   # pairwise answer similarity >= this counts as two models "agreeing"


def _compare_takeaway(labels, scores, consensus, n):
    """One plain-English line a business analyst can act on."""
    if not n:
        return ""
    if scores:   # graded: lead with who was most correct
        top = max(s["correct"] for s in scores)
        winners = [labels[i] for i, s in enumerate(scores) if s["correct"] == top]
        tot = max((s["graded"] for s in scores), default=n) or n
        lead = (f"{winners[0]} was the most accurate: {top} of {tot} correct."
                if len(winners) == 1 else
                f"{' and '.join(winners)} tied at {top} of {tot} correct.")
        if consensus == n:
            return f"{lead} The models fully agreed on all {n} questions."
        return f"{lead} The models fully agreed on {consensus} of {n} - trust those, review the rest."
    return (f"The models fully agreed on {consensus} of {n} questions. "
            f"Where they split, open the row to compare answers.")


def compare_worker(slot_models, max_tokens):
    """slot_models: 2-3 model dicts (A, B, C). Per question, compute pairwise
    answer similarity: A vs B, A vs C, B vs C (a repeated model is only called once)."""
    rows = list(STATE["rows"])
    with LOCK:
        COMPARE.update(running=True, done=0, total=len(rows), rows=[], agg=None, task="qa",
                       models=[{"label": model_view(m)["label"], "color": m["color"]} for m in slot_models],
                       max_tokens=max_tokens)
    results = []
    for row in rows:
        if not COMPARE["running"]:
            break
        q = row["question"]
        cache = {}
        answers = []
        for m in slot_models:
            if m["key"] not in cache:
                try:
                    out, p, c, _ = call_model(m, q, max_tokens)
                    add_lifetime(p + c)
                except Exception as e:  # noqa: BLE001
                    out = "[error] " + str(e)[:120]
                cache[m["key"]] = out
            answers.append(cache[m["key"]])
        a, b, c = (answers + ["", "", ""])[:3]
        ab, ac, bc = round(similarity(a, b) * 100), round(similarity(a, c) * 100), round(similarity(b, c) * 100)
        alike = sum(1 for s in (ab, ac, bc) if s >= SIM_AGREE)   # how many of the 3 answer pairs are alike
        correct = [grade(row, ans) for ans in (a, b, c)][:len(slot_models)]   # True / False / None per model
        gv = [x for x in correct if x is not None]
        if gv:   # graded: "agreement" = did they converge on the right answer (ignore phrasing)
            nc = sum(1 for x in gv if x)
            agree = "all" if nc == len(gv) else ("allwrong" if nc == 0 else "split")
        else:    # open-ended: no key, so fall back to answer text similarity
            agree = "all" if alike == 3 else ("some" if alike >= 1 else "none")
        rec = {"question": q, "answers": [a[:500], b[:500], c[:500]], "expected": _row_answer(row),
               "ab": ab, "ac": ac, "bc": bc, "agree": agree, "correct": correct}
        results.append(rec)
        with LOCK:
            COMPARE["rows"] = list(results)
            COMPARE["done"] = len(results)
    n = len(results) or 1
    labels = [model_view(m)["label"] for m in slot_models]
    graded = any(any(x is not None for x in r["correct"]) for r in results)
    scores = None
    if graded:
        scores = [{"correct": sum(1 for r in results if i < len(r["correct"]) and r["correct"][i] is True),
                   "graded": sum(1 for r in results if i < len(r["correct"]) and r["correct"][i] is not None)}
                  for i in range(len(slot_models))]
    consensus = sum(1 for r in results if r["agree"] == "all")
    agg = {"n": len(results), "labels": labels, "graded": graded, "scores": scores, "consensus": consensus,
           "ab": round(sum(r["ab"] for r in results) / n), "ac": round(sum(r["ac"] for r in results) / n),
           "bc": round(sum(r["bc"] for r in results) / n),
           "takeaway": _compare_takeaway(labels, scores, consensus, len(results))}
    with LOCK:
        COMPARE["agg"] = agg
        COMPARE["running"] = False


# the two text-extraction baselines a document run can race the vision path against.
# "ocr" is raw RapidOCR text; "parse" is layout-aware markdown from Nemotron Parse
# (tables kept as tables). Both are precomputed into the docset manifests.
_TEXT_LANES = {
    "ocr":   {"row_key": "__ocr", "phrase": "OCR text",
              "intro": "Below is the text an OCR engine extracted from a scanned document."},
    "parse": {"row_key": "__parse", "phrase": "parser text",
              "intro": "Below is what a layout-aware document parser extracted from a scanned document. "
                       "Tables are preserved as markdown tables."},
}


def _doc_lanes(rows):
    """Which text baselines this run races, per Settings. A lane only runs if the
    docset actually carries its text (older sets may predate the parse bake)."""
    want = {"ocr": ["ocr"], "parse": ["parse"], "both": ["ocr", "parse"]}.get(
        SETTINGS.get("text_baseline") or "ocr", ["ocr"])
    have = [k for k in want if any(r.get(_TEXT_LANES[k]["row_key"]) for r in rows)]
    return have or ["ocr"]


def _doc_takeaway(labels, scores, lanes=("ocr",)):
    """The one line an executive needs from a document run: best page reader, and
    what reading the page directly bought over the text-extraction pipeline(s)."""
    if not scores:
        return ""
    best = max(range(len(scores)), key=lambda i: scores[i]["img_pct"])
    b = scores[best]
    vs = " and ".join(f"{b[k + '_pct']}% from {_TEXT_LANES[k]['phrase']}" for k in lanes)
    lead = (f"{labels[best]} read the documents best: {b['img_pct']}% of fields correct from the page image, "
            f"versus {vs} when the same model only saw extracted text.")
    gap = {k: round(sum(s["img_pct"] - s[k + "_pct"] for s in scores) / len(scores)) for k in lanes}
    if len(lanes) == 2:
        if gap["parse"] > 2:
            return lead + (f" Reading the page directly beat OCR text by {gap['ocr']} points and the layout-aware "
                           f"parser by {gap['parse']} points on average - even structured extraction loses what only the page shows.")
        if gap["ocr"] > 2:
            return lead + (f" The layout-aware parser closed the OCR gap ({gap['ocr']} points down to {gap['parse']}) - "
                           "tables survive extraction, but vision still wins where the answer is visual.")
        return lead + " All three paths were close on these pages - mostly plain printed text."
    k = lanes[0]
    if gap[k] > 2:
        return lead + f" Across all models, reading the page directly beat the {_TEXT_LANES[k]['phrase']} pipeline by {gap[k]} points on average - that gap is what a vision model buys."
    if gap[k] < -2:
        return lead + f" The {_TEXT_LANES[k]['phrase']} pipeline held its own here - these pages are mostly plain text."
    return lead + " On these pages the two approaches were close - the difference shows up on tables, charts, and forms."


def compare_worker_doc(slot_models, max_tokens):
    """Document ingestion compare: every model reads each page as the page image
    (the vision path) and once per enabled text baseline - precomputed OCR text
    and/or layout-aware parser text. Every answer is graded against the same exact
    ground-truth fields, so the result IS the vision-vs-extraction judgement, per
    model, with no judge model."""
    rows = list(STATE["rows"])
    lanes = _doc_lanes(rows)
    with LOCK:
        COMPARE.update(running=True, done=0, total=len(rows), rows=[], agg=None, task="doc", lanes=lanes,
                       models=[{"label": model_view(m)["label"], "color": m["color"]} for m in slot_models],
                       max_tokens=max_tokens)
    results = []
    tot = {m["key"]: {"img_ok": 0, "fields": 0, "img_ms": 0.0, "n": 0,
                      **{k + "_ok": 0 for k in lanes}} for m in slot_models}
    for row in rows:
        if not COMPARE["running"]:
            break
        fields = row.get("__expected_fields") or {}
        lane_q = {k: (_TEXT_LANES[k]["intro"] + "\n\n--- extracted text ---\n"
                      + (row.get(_TEXT_LANES[k]["row_key"]) or "(no text recognized)")
                      + "\n--- end of extracted text ---\n\n" + row["question"]) for k in lanes}
        per = [None] * len(slot_models)

        def work(i, m):
            rec = {}
            try:
                t0 = time.time()
                out, p, c, _ = call_model(m, row["question"], max_tokens, image=row.get("__image"))
                add_lifetime(p + c)
                rec["img"], rec["img_ms"] = out, int((time.time() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                rec["img"], rec["img_ms"] = "[error] " + str(e)[:120], 0
            for k in lanes:
                try:
                    out2, p2, c2, _ = call_model(m, lane_q[k], max_tokens)
                    add_lifetime(p2 + c2)
                    rec[k] = out2
                except Exception as e:  # noqa: BLE001
                    rec[k] = "[error] " + str(e)[:120]
            per[i] = rec

        threads = [threading.Thread(target=work, args=(i, m), daemon=True) for i, m in enumerate(slot_models)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        img_hits = [grade_fields(fields, (r or {}).get("img", "")) for r in per]
        lane_hits = {k: [grade_fields(fields, (r or {}).get(k, "")) for r in per] for k in lanes}
        for i, m in enumerate(slot_models):
            t = tot[m["key"]]
            t["img_ok"] += sum(img_hits[i].values())
            for k in lanes:
                t[k + "_ok"] += sum(lane_hits[k][i].values())
            t["fields"] += len(fields)
            t["img_ms"] += (per[i] or {}).get("img_ms", 0)
            t["n"] += 1
        rec = {"question": row.get("__label") or row["question"], "image": row.get("__image_url"),
               "fields": fields, "nf": len(fields),
               "answers": [((r or {}).get("img", ""))[:2000] for r in per],
               "img_ok": [sum(h.values()) for h in img_hits],
               "hits": [{k: bool(v) for k, v in h.items()} for h in img_hits]}
        for k in lanes:
            rec[k + "_answers"] = [((r or {}).get(k, ""))[:2000] for r in per]
            rec[k + "_ok"] = [sum(h.values()) for h in lane_hits[k]]
        results.append(rec)
        with LOCK:
            COMPARE["rows"] = list(results)
            COMPARE["done"] = len(results)
    labels = [model_view(m)["label"] for m in slot_models]
    scores = []
    for m in slot_models:
        t = tot[m["key"]]
        nf = max(1, t["fields"])
        sc = {"img_ok": t["img_ok"], "fields": t["fields"], "img_pct": round(100 * t["img_ok"] / nf),
              "avg_ms": round(t["img_ms"] / t["n"]) if t["n"] else None}
        for k in lanes:
            sc[k + "_ok"] = t[k + "_ok"]
            sc[k + "_pct"] = round(100 * t[k + "_ok"] / nf)
        scores.append(sc)
    agg = {"n": len(results), "labels": labels, "task": "doc", "scores": scores, "lanes": lanes,
           "nf": sum(r["nf"] for r in results),
           "takeaway": _doc_takeaway(labels, scores, lanes)}
    with LOCK:
        COMPARE["agg"] = agg
        COMPARE["running"] = False


def start_compare(keys, max_tokens):
    models = [MODEL_BY_KEY.get(k) for k in (keys or [])]
    models = [m for m in models if m]
    if len(models) < 2:
        return False
    COMPARE["running"] = False
    time.sleep(0.1)
    task_doc = any(r.get("__image") for r in STATE["rows"])
    threading.Thread(target=compare_worker_doc if task_doc else compare_worker,
                     args=(models, int(max_tokens or 256)), daemon=True).start()
    return True


def compare_snapshot():
    with LOCK:
        return {"running": COMPARE["running"], "models": list(COMPARE["models"]),
                "done": COMPARE["done"], "total": COMPARE["total"],
                "max_tokens": COMPARE["max_tokens"], "task": COMPARE.get("task") or "qa",
                "lanes": list(COMPARE.get("lanes") or []),
                "rows": list(COMPARE["rows"]), "agg": COMPARE["agg"]}


# ---------- HTTP ----------
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def end_headers(self):
        # never cache: the dashboard is edited live, always serve fresh assets
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        if _AUTH_EXPECT is None:
            return True
        if hmac.compare_digest(self.headers.get("Authorization", ""), _AUTH_EXPECT):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="llm-bench"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _client(self):
        return self.headers.get("X-Owner", "") or "anon"

    def _busy_for(self):
        """True if a run is active and this caller isn't the one who started it."""
        return STATE["running"] and STATE["owner"] and self._client() != STATE["owner"]

    def do_GET(self):
        if not self._authed():
            return
        if self.path == "/run/stats":
            snap = stats_snapshot()
            snap["locked"] = STATE["running"] and bool(STATE["owner"])
            snap["you_own"] = (not snap["locked"]) or (self._client() == STATE["owner"])
            # only the active driver receives watchers' messages (watchers get a send ack, not each other's notes)
            is_driver = bool(STATE["owner"]) and self._client() == STATE["owner"]
            snap["messages"] = list(STATE["messages"]) if is_driver else []
            return self._send(snap)
        if self.path == "/run/rows":
            return self._send({"rows": STATE["rows"], "columns": STATE["columns"],
                               "csv_name": STATE["csv_name"]})
        if self.path == "/compare/stats":
            return self._send(compare_snapshot())
        if self.path == "/compare/models":
            return self._send({"models": [{"key": m["key"], "label": model_view(m)["label"],
                                           "color": m["color"], "system": m["system"]} for m in MODELS]})
        if self.path == "/config/models":   # editable per-slot endpoints for the warm-up editor
            return self._send({"models": _models_editable()})
        if self.path == "/config/export":   # the whole setup as one portable bundle
            return self._send(export_bundle())
        if self.path == "/version":   # installed version, read locally - never phones home
            return self._send({"installed": _installed_version()})
        if self.path == "/machines/pushenroll/status":   # live progress of a one-click enrollment
            return self._send(PUSHENROLL)
        if self.path == "/machines/unenroll/status":   # live progress of an unenroll teardown
            return self._send(UNENROLL)
        if self.path == "/config/machines":   # the registry: the one place addresses live
            return self._send({"machines": MACHINES})
        if self.path == "/shelf":   # the Model Shelf: machines + served/gettable models + agent hardware
            sv = shelf_view()
            lib = []
            for x in LIBRARY:   # per-model fit notes, computed against the live registry
                fits = [mm["name"] for mm in MACHINES if _machine_fit(x, mm)[0] is not False
                        and any(e["id"] == mm["id"] and e.get("can_swap") for e in sv)]
                lib.append({**x, "need_gb": _model_need_gb(x), "fits": fits})
            return self._send({"machines": sv, "catalog": merged_catalog(), "library": lib,
                               "trending_ids": [e["id"] for e in (TRENDING.get("entries") or []) if e.get("id")],
                               "trending_vision": [e["id"] for e in (TRENDING.get("entries") or [])
                                                   if e.get("id") and e.get("vision")],
                               "trending_at": TRENDING.get("fetched_at") or ""})
        if self.path == "/config/settings":   # user preferences (Settings hub)
            return self._send(dict(SETTINGS))
        if self.path == "/run/pull/status":   # live progress of a model download
            with LOCK:
                return self._send(dict(PULL))
        if self.path.split("?", 1)[0] == "/update/check":   # phones github.com; cached an hour to stay polite
            fresh = "fresh=1" in self.path   # the Settings button asks fresh; the quiet banner check takes the cache
            installed = _installed_version()
            source = os.path.isdir(os.path.join(REPO_ROOT, ".git"))
            latest, notes, err = "", "", ""
            if not source:
                if not fresh and _UPDATE_CACHE["latest"] and time.time() - _UPDATE_CACHE["ts"] < 3600:
                    latest, notes = _UPDATE_CACHE["latest"], _UPDATE_CACHE["notes"]
                else:
                    try:
                        latest = _http_get(_PUBLIC_RAW + "VERSION").strip()
                        try:   # the release's one-line "what's new", shown in the banner
                            notes = _http_get(_PUBLIC_RAW + "NOTES").strip().splitlines()[0][:160]
                        except Exception:  # noqa: BLE001 - releases before notes existed
                            notes = ""
                        _UPDATE_CACHE.update(ts=time.time(), latest=latest, notes=notes)
                    except Exception:  # noqa: BLE001
                        err = "Couldn't reach github.com to check."
            return self._send({"installed": installed, "latest": latest, "notes": notes,
                               "source_install": source,
                               "available": bool(latest) and latest != installed, "error": err or None})
        if self.path == "/run/datasets":
            return self._send({"types": TASK_TYPES, "pinned": PINNED_DATASETS,
                               "datasets": [_dataset_meta(d) for d in _all_datasets()],
                               "deleted": [_dataset_meta(d) for d in _deleted_datasets()]})
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(ln) if ln else b""
        if self.path == "/enroll":
            # the ONLY endpoint outside Basic Auth: a GPU box announces itself with the token
            # its bootloader carries. The registered host is the source address we actually
            # see - never what the box claims - so the record is reachable by construction.
            try:
                opts = json.loads(body or b"{}")
            except Exception:  # noqa: BLE001
                return self._send({"ok": False, "error": "bad request"}, 400)
            e, err = take_enrollment(str(opts.get("token") or ""))
            if not e:
                return self._send({"ok": False, "error": err}, 403)
            try:
                vport, cport = int(opts.get("vllm_port") or 0), int(opts.get("control_port") or 0)
            except Exception:  # noqa: BLE001
                vport = cport = 0
            if not (0 < vport < 65536):
                return self._send({"ok": False, "error": "vllm_port is required"}, 400)
            host = self.client_address[0]
            if ":" in host:
                host = f"[{host}]"   # IPv6 in a URL
            base = f"http://{host}:{vport}/v1"
            gname = str(opts.get("gpu_name") or "GPU").strip()[:40]
            hostn = str(opts.get("hostname") or host).strip()[:24]
            try:
                vram = int(opts.get("vram_mb") or 0)
            except Exception:  # noqa: BLE001
                vram = 0
            name = f"{hostn} {gname} (GPU {opts.get('gpu_index', 0)})"[:60]
            mm = _mach_for_base(base)
            if mm:   # re-running the bootloader updates, never duplicates
                mm.update(name=name, ckey=e["ckey"], cport=cport or None, vram_mb=vram or mm.get("vram_mb"))
            else:
                mm = {"id": f"mach-{int(time.time())}-{vport}", "name": name, "base": base,
                      "kind": "openai", "key": "", "ckey": e["ckey"], "cport": cport or None,
                      "vram_mb": vram or None}
                MACHINES.append(mm)
            _save_machines()
            return self._send({"ok": True, "id": mm["id"], "name": mm["name"]})
        if not self._authed():
            return
        # one-at-a-time lock: while a run is active, only its owner may drive or change
        # anything; everyone else is told it's busy (they can still watch via GET).
        if self.path in _GATED_WHILE_RUNNING and self._busy_for():
            return self._send({"error": "busy", "busy": stats_snapshot()}, 409)
        if self.path == "/run/message":   # a watcher sends a note to the current driver (deliberately NOT gated)
            try:
                opts = json.loads(body or b"{}")
            except Exception:  # noqa: BLE001
                opts = {}
            text = (opts.get("text") or "").strip()[:500]
            name = (opts.get("name") or "").strip()[:40] or "A watcher"
            if not text:
                return self._send({"error": "type a message first"}, 400)
            with LOCK:
                active = bool(STATE["running"] and STATE["owner"])
                is_owner = self._client() == STATE["owner"]
                if active and not is_owner:
                    STATE["msg_seq"] += 1
                    STATE["messages"].append({"id": STATE["msg_seq"], "name": name,
                                              "text": text, "ts": time.time()})
                    del STATE["messages"][:-30]   # keep only the most recent notes
            if not active:
                return self._send({"error": "no test drive is running right now"}, 409)
            if is_owner:
                return self._send({"error": "you're the one driving"}, 409)
            return self._send({"ok": True})
        if self.path == "/run/upload":
            try:
                rows, cols = parse_questions(body.decode("utf-8", "replace"))
                if not rows:
                    return self._send({"error": "no questions found"}, 400)
                err = _task_lineup_error(doc=False)
                if err:
                    return self._send({"error": err}, 409)
                name = self.headers.get("X-Filename", "uploaded")
                _load_questions(rows, cols, name)
                return self._send({"rows": len(rows), "columns": cols, "csv_name": name})
            except Exception as e:  # noqa: BLE001
                return self._send({"error": str(e)}, 400)
        if self.path == "/run/parse":   # parse a link OR pasted text WITHOUT loading; returns a preview to review
            try:
                opts = json.loads(body or b"{}")
                raw = (opts.get("text") or "").strip()
                if not raw:
                    return self._send({"error": "paste a link or some questions first"}, 400)
                single_line = "\n" not in raw
                is_url = bool(re.match(r'^https?://', raw)) or (
                    single_line and re.search(r'\b(github\.com|gist\.github\.com|raw\.githubusercontent\.com|gitlab\.[^/\s]+)/', raw))
                hf_total = 0
                if single_line and "huggingface.co/datasets" in raw:
                    text, name, hf_total = fetch_hf_dataset(raw if raw.startswith("http") else "https://" + raw,
                                                            opts.get("token"))
                elif is_url:
                    url = raw if raw.startswith("http") else "https://" + raw
                    text, resolved = fetch_remote(url, opts.get("token"))
                    name = resolved.rstrip("/").split("/")[-1] or "remote"
                else:
                    text, name = raw, "Pasted questions"
                name = (opts.get("name") or "").strip() or name   # caller (file upload) can supply a name
                rows, _ = parse_questions(text)
                if not rows:
                    return self._send({"error": "no questions found"}, 400)
                extra = {"total": hf_total} if hf_total and hf_total > len(rows) else {}
                return self._send({"ok": True, "name": name, "csv": text, **extra, **_preview_rows(rows)})
            except Exception as e:  # noqa: BLE001
                return self._send({"error": "could not read that - " + str(e)[:160]}, 400)
        if self.path == "/run/dataset/get":   # preview a built-in/saved dataset (by id) WITHOUT loading
            opts = json.loads(body or b"{}")
            d = _find_dataset(opts.get("id"))
            if not d:
                return self._send({"error": "unknown dataset"}, 404)
            if d.get("type") == "doc":        # document set: pages + the fields each page is graded on
                rows = _docset_rows(d)
                sample = [{"q": r["__label"], "a": ", ".join(r["__expected_fields"].keys()),
                           "img": r["__image_url"]} for r in rows]
                return self._send({"ok": True, "name": d["name"], "type": "doc", "id": d["id"],
                                   "count": len(rows), "graded": True, "sample": sample,
                                   "fields": sum(len(r["__expected_fields"]) for r in rows),
                                   "suggest": d.get("suggest") or 512})
            rows, _ = parse_questions(d["csv"])
            pv = _preview_rows(rows)
            if d.get("suggest"):
                pv["suggest"] = d["suggest"]   # dataset can override the auto answer-length
            return self._send({"ok": True, "name": d["name"], "csv": d["csv"],
                               "type": d.get("dtype") or "qa", **pv})
        if self.path == "/run/dataset":   # load a built-in/saved dataset (by id) into the run
            opts = json.loads(body or b"{}")
            d = _find_dataset(opts.get("id"))
            if not d:
                return self._send({"error": "unknown dataset"}, 404)
            err = _task_lineup_error(doc=d.get("type") == "doc")
            if err:
                return self._send({"error": err}, 409)
            if d.get("type") == "doc":
                rows = _docset_rows(d)
                _load_questions(rows, ["question"], d["name"])
                DATASET_USED[d["id"]] = int(time.time())
                _save_datasets()
                return self._send({"rows": len(rows), "columns": ["question"], "csv_name": d["name"]})
            rows, cols = parse_questions(d["csv"])
            _load_questions(rows, cols, d["name"])
            DATASET_USED[d["id"]] = int(time.time())
            _save_datasets()
            return self._send({"rows": len(rows), "columns": cols, "csv_name": d["name"]})
        if self.path == "/run/dataset/save":   # persist a reviewed dataset as a reusable card
            opts = json.loads(body or b"{}")
            name = (opts.get("name") or "").strip()
            csv_text = (opts.get("csv") or "").strip()
            if not name or not csv_text:
                return self._send({"error": "name and content are required"}, 400)
            rows, _ = parse_questions(csv_text)
            if not rows:
                return self._send({"error": "no questions to save"}, 400)
            slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:40] or "set"
            rec = {"id": f"saved-{slug}-{len(SAVED_DATASETS) + 1}", "name": name,
                   "desc": (opts.get("desc") or "").strip(), "csv": csv_text,
                   "created": int(time.time())}
            SAVED_DATASETS.append(rec)
            _save_datasets()
            return self._send({"ok": True, **_dataset_meta(rec)})
        if self.path == "/update/apply":   # re-run the one-line installer; settings survive
            if os.path.isdir(os.path.join(REPO_ROOT, ".git")):
                return self._send({"error": "This copy runs from a source checkout - update it with git."}, 400)
            if os.name == "nt":
                _run_detached(f"irm {_PUBLIC_RAW}get.ps1 | iex", "update")
            else:
                _run_detached(f"curl -fsSL {_PUBLIC_RAW}get.sh | bash", "update")
            return self._send({"ok": True})
        if self.path == "/uninstall":   # remove services + launcher; files stay for a clean delete
            if os.name == "nt":
                _run_detached(f"& '{os.path.join(REPO_ROOT, 'uninstall-bakeoff.cmd')}'", "uninstall")
            else:
                _run_detached(f"bash {shlex.quote(os.path.join(REPO_ROOT, 'uninstall.sh'))}", "uninstall")
            return self._send({"ok": True})
        if self.path == "/run/dataset/pin":   # toggle a card on/off the pinned strip
            opts = json.loads(body or b"{}")
            did = opts.get("id")
            if not did or not _find_dataset(did):
                return self._send({"error": "unknown task"}, 400)
            if did in PINNED_DATASETS:
                PINNED_DATASETS.remove(did)
            else:
                PINNED_DATASETS.append(did)
            _save_datasets()
            return self._send({"ok": True, "pinned": PINNED_DATASETS})
        if self.path == "/run/dataset/delete":   # move a card to Deleted tasks (restorable)
            opts = json.loads(body or b"{}")
            did = opts.get("id")
            gone = next((d for d in SAVED_DATASETS if d.get("id") == did), None)
            removed = False
            if gone is not None:
                SAVED_DATASETS[:] = [d for d in SAVED_DATASETS if d.get("id") != did]
                DELETED_DATASETS.append(gone)
                removed = True
            elif any(d["id"] == did for d in BUILTIN_DATASETS + BUILTIN_DOCSETS):
                if did not in HIDDEN_DATASETS:
                    HIDDEN_DATASETS.append(did)
                removed = True
            _save_datasets()
            return self._send({"ok": removed})
        if self.path == "/run/dataset/restore":   # bring a card back from Deleted tasks
            opts = json.loads(body or b"{}")
            did = opts.get("id")
            back = next((d for d in DELETED_DATASETS if d.get("id") == did), None)
            if back is not None:
                DELETED_DATASETS[:] = [d for d in DELETED_DATASETS if d.get("id") != did]
                SAVED_DATASETS.append(back)
            elif did in HIDDEN_DATASETS:
                HIDDEN_DATASETS.remove(did)
                if did in PURGED_DATASETS:
                    PURGED_DATASETS.remove(did)
            else:
                return self._send({"ok": False, "error": "not in deleted tasks"})
            _save_datasets()
            return self._send({"ok": True})
        if self.path == "/run/dataset/purge":   # empty Deleted tasks - saved cards are gone for good
            DELETED_DATASETS[:] = []
            PURGED_DATASETS[:] = sorted(set(PURGED_DATASETS) | set(HIDDEN_DATASETS))
            _save_datasets()
            return self._send({"ok": True})
        if self.path == "/run/pull":   # load a model onto a vLLM endpoint via its swap agent
            opts = json.loads(body or b"{}")
            base = str(opts.get("base") or "").strip()
            model = str(opts.get("model") or "").strip()
            if not base or not model:
                return self._send({"ok": False, "error": "enter a model name first"})
            ok = start_pull(base, model, opts.get("kind"))
            return self._send({"ok": ok, "error": None if ok else "a model is already loading"})
        if self.path == "/run/warmup":   # ping one model (tiny call) to confirm it's reachable + load it
            opts = json.loads(body or b"{}")
            if opts.get("base") and opts.get("model"):   # ad-hoc test of unsaved edits
                m = {"key": "_test", "label": "_test", "system": "", "color": "#888",
                     "kind": str(opts.get("kind") or "openai").strip().lower(),
                     "base": str(opts["base"]).strip(), "model": str(opts["model"]).strip()}
            else:
                m = MODEL_BY_KEY.get(opts.get("key"))
            if not m:
                return self._send({"error": "unknown model"}, 404)
            # reality check before pinging: ask the endpoint what it actually serves.
            # - single-model endpoint (e.g. vLLM): auto-adopt the served name (stale preset names heal).
            # - multi-model endpoint missing the model: report it as MISSING with what to do
            #   about it (swap-capable boxes restart on it; others offer the real list).
            if not _is_gateway(m):
                try:
                    pr = probe_endpoint(m.get("kind"), m.get("base"), (machine_by_id(m.get("machine")) or {}).get("key"))
                    served = (pr.get("models") or []) if pr.get("ok") else []
                    if served and m.get("model") not in served:
                        can_swap = _has_control(m.get("base"))   # box runs a control agent -> we can restart it on the asked-for model
                        if len(served) == 1 and not can_swap:
                            # read-only single-model server (plain vLLM): reality wins, adopt what's loaded
                            if m.get("label") in (m.get("model"), "", None):
                                m["label"] = served[0]   # display name was tracking the model - keep them in step
                            m["model"] = served[0]
                            if not (opts.get("base") and opts.get("model")):   # a real saved slot, not an ad-hoc test
                                _save_models_config()
                        else:
                            return self._send({"ok": False, "missing": True,
                                               "can_pull": False,
                                               "can_swap": can_swap,
                                               "served": served[:50],
                                               "error": "model not on this server"})
                except Exception:  # noqa: BLE001
                    pass
            t0 = time.time()
            if _is_gateway(m):   # make the chosen variant AWAKE on this slot before pinging
                gateway_ensure(m["model"], GATEWAY_SLOTS[m["key"]])
            try:
                call_model(m, "ping", 8)
                return self._send({"ok": True, "ms": int((time.time() - t0) * 1000)})
            except Exception as e:  # noqa: BLE001
                # server down is not always broken: its agent may be mid-restart onto a
                # model (downloads take a while). Report that as loading, not as dead.
                st = _control_status(m.get("base") or "")
                if st and st.get("loading"):
                    return self._send({"ok": False, "loading": True,
                                       "model": st.get("model") or m.get("model")})
                if st and st.get("error"):
                    # the swap agent knows WHY the port is dead (model failed to boot,
                    # wrong arch for this box's vLLM...) - say that, not the socket errno
                    return self._send({"ok": False, "load_failed": True,
                                       "model": st.get("model") or m.get("model"),
                                       "error": f"{str(st.get('model') or m.get('model') or '').split('/')[-1]} failed to start on this machine"})
                return self._send({"ok": False, "ms": int((time.time() - t0) * 1000), "error": str(e)[:140]})
        if self.path == "/run/probe":   # discover what models an endpoint serves (self-service onboarding)
            opts = json.loads(body or b"{}")
            if not str(opts.get("base") or "").strip():
                return self._send({"ok": False, "error": "enter an address first"})
            t0 = time.time()
            res = probe_endpoint(opts.get("kind"), opts.get("base"), (opts.get("key") or "").strip() or None)
            if res.get("ok") and res.get("kind") == "openai" and not res.get("gateway"):
                res["control"] = _has_control(opts.get("base"))   # can models be swapped here, or read-only?
            res["ms"] = int((time.time() - t0) * 1000)
            return self._send(res)
        if self.path == "/config/models":   # save user-edited endpoints (bring your own infrastructure)
            opts = json.loads(body or b"{}")
            return self._send({"ok": True, "models": apply_models_config(opts.get("models"))})
        if self.path == "/machines/unenroll":   # the reverse: clean the box, forget the machines
            opts = json.loads(body or b"{}")
            ok, err = start_unenroll(str(opts.get("id") or ""))
            return self._send({"ok": ok, "error": err})
        if self.path == "/machines/pushenroll":   # one-click: enroll a GPU box over ssh
            opts = json.loads(body or b"{}")
            addr = str(opts.get("addr") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?", addr):
                return self._send({"ok": False, "error": "enter an ssh address like user@gpu-box (or an ssh config alias)"})
            ok = start_push_enroll(addr, str(opts.get("name") or "").strip())
            return self._send({"ok": ok, "error": None if ok else "an enrollment is already running"})
        if self.path == "/config/machines":   # replace the machine registry, re-point slots
            opts = json.loads(body or b"{}")
            return self._send({"ok": True, "machines": apply_machines(opts.get("machines")),
                               "models": _models_editable()})
        if self.path == "/catalog/refresh":   # manual, user-pressed: ask Hugging Face for trending models
            try:
                n, nvis = refresh_trending()
                return self._send({"ok": True, "added": n, "vision_added": nvis,
                                   "fetched_at": TRENDING.get("fetched_at")})
            except Exception as e:  # noqa: BLE001
                return self._send({"ok": False, "error": "could not reach Hugging Face - " + str(e)[:120]})
        if self.path == "/lineup/auto":   # one button: field the best race lineup for a task
            opts = json.loads(body or b"{}")
            picks = auto_lineup(str(opts.get("task") or "qa"))
            if not picks:
                return self._send({"ok": False, "error": "no machines are reachable - check Configure > Machines"})
            return self._send({"ok": True, "picks": picks, "models": _models_editable()})
        if self.path == "/machines/bootloader":   # mint a one-file onboarder for a new GPU box
            opts = json.loads(body or b"{}")
            url = str(opts.get("url") or "").strip().rstrip("/")
            if not url.lower().startswith("http"):
                return self._send({"ok": False, "error": "missing this dashboard's address"}, 400)
            try:
                script, tok = build_bootloader(url)
            except Exception as e:  # noqa: BLE001
                return self._send({"ok": False, "error": "could not build the bootloader - " + str(e)[:120]}, 500)
            return self._send({"ok": True, "script": script, "token": tok,
                               "hours": ENROLL_TTL // 3600, "max_uses": ENROLL_MAX_USES})
        if self.path == "/library/add":   # a pasted Hugging Face model joins the shared catalog
            opts = json.loads(body or b"{}")
            mid = str(opts.get("id") or "").strip().strip("/")
            if not re.match(r"^[\w.-]+/[\w.-]+$", mid):
                return self._send({"ok": False, "error": "that doesn't look like a Hugging Face model id"}, 400)
            existing = lib_entry(mid)
            meta = hf_model_meta(mid)   # always fetch fresh - sizes and tags improve over time
            meta.pop("hf_ok", None)   # transient trust flag, not library data
            if existing:
                existing.update({k: meta[k] for k in ("params_b", "weight_gb", "vision", "gated", "vendor", "blurb")})
                meta = existing
            else:
                meta["added"] = int(time.time())
                LIBRARY.append(meta)
            _save_library()
            _see_cache.clear()   # capability answers may change now the library knows this model
            # start a cache-only prefetch on the best-fitting swap machine that isn't seated in a
            # slot - the running servers are never disturbed by an add
            seated = {m.get("machine") for m in MODELS if m.get("model") and m.get("base")}
            cands = []
            for mm in MACHINES:
                fit, _need = _machine_fit(meta, mm)
                if fit is False or not _has_control(mm.get("base")):
                    continue
                cands.append((mm.get("id") in seated, -(mm.get("vram_mb") or 0), mm))
            cands.sort(key=lambda c: (c[0], c[1]))
            prefetch_to, prefetch_err = "", ""
            if cands:
                mm = cands[0][2]
                try:
                    res = _agent_post(mm["base"], "/control/prefetch",
                                      {"model": mid, "total_bytes": int((meta.get("weight_gb") or 0) * 1e9)},
                                      timeout=8)
                    if res.get("ok"):
                        prefetch_to = mm["name"]
                    else:
                        prefetch_err = f"{mm['name']}: {res.get('error') or 'refused the download'}"
                except urllib.error.HTTPError as e:  # noqa: BLE001
                    prefetch_err = (f"{mm['name']}'s agent doesn't support background downloads yet - "
                                    "re-run its bootloader to update it" if e.code == 404
                                    else f"{mm['name']} answered {e.code}")
                except Exception:  # noqa: BLE001
                    prefetch_err = f"couldn't reach {mm['name']}'s agent"
            fits = [mm["name"] for mm in MACHINES if _machine_fit(meta, mm)[0] is not False
                    and _has_control(mm.get("base"))]
            return self._send({"ok": True, "model": meta, "already": bool(existing),
                               "need_gb": _model_need_gb(meta), "fits": fits,
                               "prefetch_to": prefetch_to, "prefetch_err": prefetch_err})
        if self.path == "/machines/prefetch":   # live download progress across all agents, cheap to poll
            out, threads = [], []
            def probe(mm):
                st = _control_status(mm.get("base"))
                pf = (st or {}).get("prefetch") or {}
                if pf.get("model"):
                    out.append({"id": mm["id"], "name": mm["name"], "prefetch": pf})
            for mm in list(MACHINES):
                t = threading.Thread(target=probe, args=(mm,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=3)
            return self._send({"ok": True, "prefetches": out})
        if self.path == "/machine/model-delete":   # free a cached model's weights from a box's disk
            opts = json.loads(body or b"{}")
            mm = machine_by_id(opts.get("id"))
            model = str(opts.get("model") or "").strip()
            if not mm or not model:
                return self._send({"ok": False, "error": "unknown machine or model"}, 400)
            if any(s.get("model") == model and s.get("machine") == mm["id"] for s in MODELS):
                return self._send({"ok": False, "error": f"{model} is seated in a slot on {mm['name']} - empty the slot first"}, 409)
            try:
                res = _agent_post(mm["base"], "/control/delete", {"model": model}, timeout=30)
            except urllib.error.HTTPError as e:
                try:
                    res = json.loads(e.read())
                except Exception:  # noqa: BLE001
                    res = {"ok": False, "error": f"the agent answered {e.code}"}
            except Exception:  # noqa: BLE001
                res = {"ok": False, "error": f"couldn't reach {mm['name']}'s agent"}
            return self._send(res)
        if self.path == "/library/remove":
            opts = json.loads(body or b"{}")
            mid = str(opts.get("id") or "").strip()
            LIBRARY[:] = [x for x in LIBRARY if x.get("id") != mid]
            _save_library()
            _see_cache.clear()
            return self._send({"ok": True})
        if self.path == "/machine/evacuate":   # free the GPU and empty its slots - the box goes dormant
            opts = json.loads(body or b"{}")
            mm = machine_by_id(opts.get("id"))
            if not mm:
                return self._send({"ok": False, "error": "unknown machine"}, 400)
            freed = False
            try:
                freed = bool(_agent_post(mm["base"], "/control/unload", {}, timeout=20).get("ok"))
            except Exception:  # noqa: BLE001
                pass   # no agent (plain endpoint) or unreachable: still clear our side
            cleared = 0
            for s in MODELS:
                if s.get("machine") == mm["id"] and (s.get("model") or s.get("base")):
                    s.update(model="", label="", base="", system="", machine=None)
                    cleared += 1
            if cleared:
                _save_models_config()
                if STATE["running"]:
                    stop_run()
                _clear_results()
            return self._send({"ok": True, "freed": freed, "cleared": cleared, "models": _models_editable()})
        if self.path == "/slot/clear":   # remove the machine from a slot - the slot goes Empty
            opts = json.loads(body or b"{}")
            slot = MODEL_BY_KEY.get(opts.get("key"))
            if not slot:
                return self._send({"ok": False, "error": "unknown slot"}, 400)
            slot.update(model="", label="", base="", system="", machine=None)
            _save_models_config()
            if STATE["running"]:
                stop_run()
            _clear_results()
            return self._send({"ok": True, "models": _models_editable()})
        if self.path == "/race/enter":   # one tap from Machines: this machine joins the race, no slot-thinking
            opts = json.loads(body or b"{}")
            mm = machine_by_id(opts.get("machine"))
            if not mm:
                return self._send({"ok": False, "error": "unknown machine"}, 400)
            pr = probe_endpoint(mm.get("kind"), mm.get("base"), mm.get("key"))
            served = (pr.get("models") or []) if pr.get("ok") else []
            # match the lineup kind: if the other slots read images this machine must field a
            # vision model too (and the other way around) - vision and text never race together
            lineup_vis = [m for m in MODELS if m.get("machine") != mm["id"]
                          and m.get("model") and m.get("base") and model_can_see(m)]
            lineup_txt = [m for m in MODELS if m.get("machine") != mm["id"]
                          and m.get("model") and m.get("base") and not model_can_see(m)]
            loading = False
            if not served and _has_control(mm.get("base")):
                # a freshly enrolled (or evacuated) box serves nothing yet. Tapping it must
                # still just work: pick a curated model that fits its GPU and the lineup kind,
                # point the slot at it, and start the swap - warm-up shows it landing.
                want_vision = bool(lineup_vis and not lineup_txt)
                vram = int(mm.get("vram_mb") or 0)
                pool = [(n, pb) for (n, pb) in VLLM_PICKS
                        if not vram or pb * 1200 <= vram]   # ~1.2 GB per B in bf16 with headroom; quantized picks pass easily
                for x in LIBRARY:   # user-added models are first-class here too
                    if _machine_fit(x, mm)[0] is not False:
                        pool.append((x["id"], float(x.get("params_b") or 4)))
                fit = [(n, pb) for (n, pb) in pool
                       if model_can_see({"kind": mm["kind"], "base": mm["base"], "model": n}) == want_vision]
                if not fit:
                    return self._send({"ok": False, "error": f"{mm['name']} has no model loaded and none of the "
                                       "available models fit its GPU - pick one from the Models list instead."})
                cached = set((_control_status(mm["base"]) or {}).get("cached") or [])
                libmeta = {x["id"]: x for x in LIBRARY}
                # models the user ADDED outrank the curated list (they were added on purpose,
                # newest first), then warm weights, then size
                fit.sort(key=lambda c: (c[0] not in libmeta,
                                        -(libmeta.get(c[0], {}).get("added") or 0),
                                        c[0] not in cached, -c[1]))
                served = [fit[0][0]]
                loading = start_pull(mm["base"], fit[0][0], "openai")
            if not served:
                return self._send({"ok": False, "error": f"{mm['name']} is not reachable or has no models loaded"})
            if lineup_vis and not lineup_txt:
                served = [n for n in served
                          if model_can_see({"kind": mm["kind"], "base": mm["base"], "model": n})]
                if not served:
                    return self._send({"ok": False, "error": f"{mm['name']} has no vision models loaded, and the "
                                       "current lineup compares vision models. Load a vision model on it first, "
                                       "or switch the lineup to text models."})
            elif lineup_txt and not lineup_vis:
                served = [n for n in served
                          if not model_can_see({"kind": mm["kind"], "base": mm["base"], "model": n})]
                if not served:
                    return self._send({"ok": False, "error": f"{mm['name']} only has vision models loaded, and the "
                                       "current lineup compares text models. Vision models only play other vision "
                                       "models - switch the lineup, or load a text model on it."})
            rank = {n: i for i, (n, _) in enumerate(TASK_PICKS.get("qa") or [])}
            model = sorted(served, key=lambda n: rank.get(n, 50 + max(0, 40 - _params_of(n))))[0]
            current = next((m for m in MODELS if m.get("machine") == mm["id"]), None)
            target = MODEL_BY_KEY.get(opts.get("slot")) if opts.get("slot") else None
            if target is None:
                # no slot named: already-mine > slot on an unreachable machine > refuse
                target = current
                if target is None:
                    for m in MODELS:
                        other = machine_by_id(m.get("machine"))
                        if other is None or not probe_endpoint(other.get("kind"), other.get("base"), other.get("key") or (machine_by_id(other.get("machine")) or {}).get("key")).get("ok"):
                            target = m
                            break
                if target is None:
                    return self._send({"ok": False, "error": "All three slots are racing healthy machines. "
                                       "Pick a slot for this machine instead."})
            entries = [{"key": target["key"], "label": model, "system": mm["name"],
                        "kind": mm["kind"], "base": mm["base"], "model": model,
                        "machine": mm["id"]}]
            if current is not None and current["key"] != target["key"]:
                if target.get("model") and target.get("base"):
                    # the machine moves slots: the displaced assignment takes the vacated slot (swap)
                    entries.append({"key": current["key"], "label": target.get("label"),
                                    "system": target.get("system"), "kind": target.get("kind"),
                                    "base": target.get("base"), "model": target.get("model"),
                                    "machine": target.get("machine")})
                else:
                    # nothing displaced (target slot was empty): the old slot goes empty, not duplicated.
                    # apply_models_config drops empty values by design, so clear it directly.
                    current.update(model="", label="", base="", system="", machine=None)
                    _save_models_config()
            apply_models_config(entries)
            return self._send({"ok": True, "slot": target["key"], "model": model,
                               "loading": loading, "models": _models_editable()})
        if self.path == "/slot/assign":   # the Shelf tap: put (machine, model) into a slot
            opts = json.loads(body or b"{}")
            slot = MODEL_BY_KEY.get(opts.get("key"))
            mm = machine_by_id(opts.get("machine"))
            model = str(opts.get("model") or "").strip()
            if not slot or not mm or not model:
                return self._send({"ok": False, "error": "unknown slot, machine, or model"}, 400)
            other = next((m for m in MODELS if m["key"] != slot["key"] and m.get("machine") == mm["id"]), None)
            if other:
                return self._send({"ok": False, "error": f"{mm['name']} is already racing in another slot "
                                   f"({other.get('label') or other.get('model')}). Each slot needs its own machine."})
            # self-heal the catalog: a seated model unknown to both the library and the
            # trending shelf gets one real HF metadata lookup, persisted, so the mix check
            # and the VRAM fit check below run on facts instead of name fragments
            if not lib_entry(model) and re.match(r"^[\w.-]+/[\w.-]+$", model) \
                    and not any(e.get("id") == model for e in TRENDING.get("entries") or []):
                meta = hf_model_meta(model)
                if meta.pop("hf_ok", False):
                    meta["added"] = int(time.time())
                    LIBRARY.append(meta)
                    _save_library()
                    _see_cache.clear()
            err = _slot_mix_error(slot["key"], mm["kind"], mm["base"], model)
            if err:
                return self._send({"ok": False, "error": err}, 409)
            lib = lib_entry(model)
            if lib:   # honest at seat time: never half-load a model onto a GPU that can't hold it
                fit, need = _machine_fit(lib, mm)
                if fit is False:
                    have = (mm.get("vram_mb") or 0) / 1024.0
                    return self._send({"ok": False, "error": f"{model} needs about {need:g} GB of GPU memory; "
                                       f"{mm['name']} has {have:.0f} GB. Pick a machine with a bigger GPU."}, 409)
            apply_models_config([{"key": slot["key"], "label": model, "system": mm["name"],
                                  "kind": mm["kind"], "base": mm["base"], "model": model,
                                  "machine": mm["id"]}])
            return self._send({"ok": True, "models": _models_editable()})
        if self.path == "/config/settings":   # save user preferences
            opts = json.loads(body or b"{}")
            try:
                SETTINGS["rounds"] = max(1, min(MAX_ROUNDS, int(opts.get("rounds") or SETTINGS["rounds"])))
                if opts.get("text_baseline") in ("ocr", "parse", "both"):
                    SETTINGS["text_baseline"] = opts["text_baseline"]
            except Exception:  # noqa: BLE001
                return self._send({"ok": False, "error": "bad value"}, 400)
            _save_settings()
            return self._send({"ok": True, **SETTINGS})
        if self.path == "/config/import":   # apply a portable setup bundle exported from another machine
            try:
                ok = import_bundle(json.loads(body or b"{}"))
            except Exception as e:  # noqa: BLE001
                return self._send({"ok": False, "error": "not a valid setup file - " + str(e)[:120]}, 400)
            return self._send({"ok": ok, "models": _models_editable()})
        if self.path == "/run/start":
            opts = json.loads(body or b"{}")
            if not STATE["rows"]:
                return self._send({"error": "upload a CSV first"}, 400)
            if not _active_models():
                return self._send({"error": "no machines are assigned to slots - open Settings (gear) > Machines"}, 400)
            # the lineup may have changed since the set was loaded - re-check the match
            err = _task_lineup_error(doc="__image" in STATE["rows"][0])
            if err:
                return self._send({"error": err}, 409)
            # never play against a model that isn't actually being served - a run of
            # 404s teaches the user nothing. Ask each endpoint what it holds right now.
            not_ready = []
            for m in _active_models():
                pr = probe_endpoint(m.get("kind"), m.get("base"), (machine_by_id(m.get("machine")) or {}).get("key"))
                if pr.get("gateway"):
                    continue   # the gateway swaps models in on demand
                if not pr.get("ok"):
                    not_ready.append({"key": m["key"], "model": m.get("model"), "why": "not reachable"})
                elif m.get("model") not in (pr.get("models") or []):
                    not_ready.append({"key": m["key"], "model": m.get("model"), "why": "not on this server"})
            if not_ready:
                names = ", ".join(x["model"] or x["key"] for x in not_ready)
                verb = "isn't" if len(not_ready) == 1 else "aren't"
                return self._send({"error": f"{names} {verb} ready to play - the machines "
                                            "aren't serving these models yet.",
                                   "not_ready": not_ready}, 409)
            start_run(opts.get("max_tokens", 256), opts.get("rounds"), opts.get("think"))
            STATE["owner"] = self._client()   # this caller now owns the run (after start_run, which clears it)
            return self._send({"running": True, "you_own": True})
        if self.path == "/run/pause":
            pause_run()
            return self._send({"paused": STATE["paused"]})
        if self.path == "/run/resume":
            resume_run()
            return self._send({"paused": STATE["paused"]})
        if self.path == "/run/stop":
            stop_run()
            return self._send({"running": False})
        if self.path == "/run/reset":
            reset_all()
            return self._send({"ok": True})
        if self.path == "/run/clear":   # start over: wipe the run's results but keep the loaded dataset
            if STATE["running"]:
                stop_run()
            _clear_results()
            return self._send({"ok": True})
        if self.path == "/compare/start":
            opts = json.loads(body or b"{}")
            if not STATE["rows"]:
                return self._send({"error": "upload a CSV first"}, 400)
            ok = start_compare(opts.get("models"), opts.get("max_tokens", 256))
            return self._send({"running": ok} if ok else {"error": "need at least 2 valid models"}, 200 if ok else 400)
        if self.path == "/compare/stop":
            COMPARE["running"] = False
            return self._send({"running": False})
        return self._send({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _tls_files():
    """Resolve the https certificate pair, or ('', '') for plain http.
    DASH_TLS_CERT/DASH_TLS_KEY point at your own pair; DASH_TLS=self mints a
    self-signed pair once (browsers warn once per device) and reuses it."""
    cert = os.environ.get("DASH_TLS_CERT", "").strip()
    key = os.environ.get("DASH_TLS_KEY", "").strip()
    if cert and key:
        return cert, key
    if os.environ.get("DASH_TLS", "").strip().lower() != "self":
        return "", ""
    tdir = os.path.join(HERE, "tls")
    cert, key = os.path.join(tdir, "cert.pem"), os.path.join(tdir, "key.pem")
    if not (os.path.exists(cert) and os.path.exists(key)):
        if not shutil.which("openssl"):
            print("DASH_TLS=self needs the openssl command; serving plain http instead",
                  file=sys.stderr, flush=True)
            return "", ""
        os.makedirs(tdir, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", key, "-out", cert, "-days", "3650", "-subj", "/CN=bake-off"],
                       check=True, capture_output=True)
        os.chmod(key, 0o600)
    return cert, key


if __name__ == "__main__":
    if "--scheme" in sys.argv:   # scripts ask the server - the one authority on http vs https
        c, k = _tls_files()
        print("https" if c and k else "http")
        raise SystemExit(0)
    STATE["lifetime"] = _load_lifetime()
    cert, key = _tls_files()
    with Server((HOST, PORT), Handler) as httpd:
        scheme = "http"
        if cert and key:
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
            scheme = "https"
        print(f"bake-off dashboard on {scheme}://{HOST}:{PORT}", flush=True)
        httpd.serve_forever()
