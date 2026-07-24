#!/usr/bin/env python3
"""Fill in the `ocr` field on every docset manifest row.

Runs RapidOCR (PaddleOCR detection + recognition models, ONNX runtime, CPU) over
each page image and stores the raw recognized text in the manifest. The dashboard
uses this as the OCR half of the "read the page image vs read OCR text" comparison,
so the server never needs an OCR runtime of its own - the docsets are static and the
OCR text is baked in beside them.

Needs: pip install rapidocr-onnxruntime   (a throwaway venv is fine)
Usage: python3 ocr_precompute.py
"""
import json
import os

from rapidocr_onnxruntime import RapidOCR

HERE = os.path.dirname(os.path.abspath(__file__))
engine = RapidOCR()

def ocr_text(path):
    result, _ = engine(path)
    if not result:
        return ""
    # keep reading order (RapidOCR returns boxes top-to-bottom already); join per line
    return "\n".join(item[1] for item in result)

for name in sorted(os.listdir(HERE)):
    mpath = os.path.join(HERE, name, "manifest.json")
    if not os.path.isfile(mpath):
        continue
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    for row in manifest["rows"]:
        img = os.path.join(HERE, name, row["file"])
        row["ocr"] = ocr_text(img)
        print(f"{name}/{row['file']}: {len(row['ocr'])} chars of OCR text")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
print("done")
