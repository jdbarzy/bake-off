#!/usr/bin/env python3
"""Fill in the `parse` field on every docset manifest row.

Sends each page image to a local Nemotron Parse NIM and stores the cleaned
markdown in the manifest, right beside the RapidOCR `ocr` text. The dashboard
uses this as a second text-extraction baseline in the "read the page image vs
read extracted text" comparison - so the server never needs the NIM at run
time, only this script does.

The NIM is not instruction-following: it takes a fixed task-token prompt and
emits line-level text with bbox/class annotations, tables as LaTeX tabular
blocks. This script strips the coordinates and converts tables to markdown.

Needs: the nemotron-parse NIM running (docker start nemotron-parse)
Usage: python3 parse_precompute.py [docset-dir ...]   (default: all docsets here)
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NIM_URL = os.environ.get("PARSE_NIM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = "nvidia/nemotron-parse-v1.2"
# fixed task-token prompt from the NIM's shipped vllm_example.py - plain-language
# prompts derail the model, and skip_special_tokens=False is what makes table
# content survive (the table markup is emitted as special tokens)
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def parse_page(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "model": MODEL, "max_tokens": 8000, "temperature": 0,
        "repetition_penalty": 1.1, "top_k": 1, "skip_special_tokens": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": TASK_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
    }
    req = urllib.request.Request(NIM_URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in (1, 2, 3):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=300))
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt} after error: {e}", flush=True)
            time.sleep(5 * attempt)


def tabular_to_markdown(m):
    """One LaTeX tabular block -> a markdown table (first row = header)."""
    rows = [r.strip() for r in m.group(1).split(r"\\") if r.strip()]
    cells = [[c.strip() for c in r.split("&")] for r in rows]
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    lines = []
    for i, r in enumerate(cells):
        r = r + [""] * (width - len(r))
        lines.append("| " + " | ".join(r) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def clean(raw):
    """NIM output -> plain markdown: tables converted, coordinates dropped."""
    txt = re.sub(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                 tabular_to_markdown, raw, flags=re.S)
    txt = txt.replace("<br>", " ").replace("<tbc>", "")
    txt = re.sub(r"<x_[0-9.]+>|<y_[0-9.]+>|<class_[^>]*>|</?s>", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def bake(dirname):
    mpath = os.path.join(HERE, dirname, "manifest.json")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    todo = [r for r in manifest["rows"] if not r.get("parse")]
    print(f"{dirname}: {len(todo)} of {len(manifest['rows'])} pages to parse", flush=True)
    for n, row in enumerate(todo, 1):
        t0 = time.time()
        row["parse"] = clean(parse_page(os.path.join(HERE, dirname, row["file"])))
        # write after every page so an interrupted run resumes where it stopped
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [{n}/{len(todo)}] {row['file']}: {len(row['parse'])} chars "
              f"in {time.time() - t0:.1f}s", flush=True)


names = sys.argv[1:] or sorted(
    d for d in os.listdir(HERE)
    if os.path.isfile(os.path.join(HERE, d, "manifest.json")))
for name in names:
    bake(name)
print("done")
