#!/usr/bin/env python3
"""Generate docs-stress-100: one hundred synthetic business documents for long VLM runs.

Same contract as generate.py (read that first): every value in the manifest `fields`
is written into the render from the same variable, so grading never needs a judge.
This generator is procedural where generate.py is hand-authored - each document gets
unique fictional data (companies, people, amounts, dates) from a fixed-seed RNG, so
re-running reproduces the set byte-for-byte (PNGs + manifest; PDFs differ only in
Pillow's embedded creation timestamp).

Mix (100 docs, ~116 pages):
  15 invoices, 12 photographed receipts, 12 handwritten notes, 12 filled forms,
  12 customer orders / POs, 10 charts, 8 banking contracts (2-4 pages each),
  10 records, 9 work orders / task sheets.

Conditions: ~1/3 clean digital, ~40% rough scan, ~1/4 hard mode (heavy skew, coffee
ring, hole punches, fade); receipts get a photo-on-desk treatment instead. One doc in
ten asks for a field that is genuinely absent (ground truth "not found").

Output: docs-stress-100/page-NNN.png + manifest.json (rows per PAGE; contract rows are
scoped to their own page) and docs-stress-100/pdf/doc-NNN.pdf (contracts multi-page).
`ocr` is filled in afterwards by ocr_precompute.py. Handwriting uses fonts/Caveat.ttf
and fonts/HomemadeApple.ttf (downloaded, OFL/Apache licensed).
"""
import json
import os
import random
import shutil
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from _docgen import (W, H, INK, MUT, LINE, TEAL, SANS, BOLD, SERIF, MONO,
                     CAVEAT, APPLE, page, text, hline, kv, table, stamp, question)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs-stress-100")
R = random.Random(20260714)

PEN_BLUE = (30, 54, 118)
PENCIL = (88, 92, 100)

def scanify(img, angle=None, speckle=None):
    angle = R.uniform(0.7, 2.3) * R.choice((-1, 1)) if angle is None else angle
    speckle = R.randrange(700, 1500) if speckle is None else speckle
    img = img.rotate(angle, expand=True, fillcolor=(246, 246, 244), resample=Image.BICUBIC)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(speckle):
        x, y = R.randrange(w), R.randrange(h)
        g = R.randrange(140, 225)
        d.point((x, y), fill=(g, g, g))
    return img.filter(ImageFilter.GaussianBlur(0.55))

def hardify(img):
    """Hard mode: heavier skew, coffee ring, hole punches, faded band, low contrast."""
    img = img.rotate(R.uniform(2.2, 3.4) * R.choice((-1, 1)), expand=True,
                     fillcolor=(243, 242, 239), resample=Image.BICUBIC)
    w, h = img.size
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    if R.random() < 0.75:  # coffee ring
        cx, cy, r = R.randrange(120, w - 160), R.randrange(160, h - 200), R.randrange(60, 105)
        for rr, a in ((r, 68), (r - 7, 30), (r + 5, 40)):
            od.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(126, 84, 40, a), width=9)
    if R.random() < 0.6:   # hole punches
        for py in (int(h * 0.25), int(h * 0.5), int(h * 0.75)):
            od.ellipse([26, py - 17, 60, py + 17], fill=(170, 170, 166, 255))
    if R.random() < 0.7:   # faded band
        fy = R.randrange(int(h * 0.2), int(h * 0.7))
        od.rectangle([0, fy, w, fy + R.randrange(90, 190)], fill=(252, 252, 250, 118))
    for _ in range(R.randrange(1600, 2600)):
        g = R.randrange(120, 215)
        od.point((R.randrange(w), R.randrange(h)), fill=(g, g, g, 200))
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(0.86)
    img = ImageEnhance.Brightness(img).enhance(R.uniform(0.96, 1.05))
    return img.filter(ImageFilter.GaussianBlur(0.8))

def hand(d, xy, s, f, fill=PEN_BLUE, wobble=1.8):
    """Handwriting: per-character baseline wobble and loose tracking."""
    x, y = xy
    for ch in s:
        d.text((x, y + R.uniform(-wobble, wobble)), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + R.uniform(-0.5, 0.9)
    return x

def lined_paper(graph=False):
    img, d = page((251, 250, 244))
    if graph:
        for gx in range(40, W, 34):
            d.line([(gx, 0), (gx, H)], fill=(214, 228, 226), width=1)
        for gy in range(40, H, 34):
            d.line([(0, gy), (W, gy)], fill=(214, 228, 226), width=1)
    else:
        for gy in range(150, H - 40, 42):
            d.line([(50, gy), (W - 40, gy)], fill=(196, 214, 232), width=2)
        d.line([(110, 60), (110, H - 40)], fill=(226, 160, 160), width=2)
    return img, d

# ------------------------------------------------------------------ fictional data
FIRST = ["Marta", "Deshawn", "Priya", "Colin", "Yuki", "Rosa", "Hank", "Ingrid", "Omar",
         "Lena", "Travis", "Amara", "Felix", "Dana", "Ruben", "Sofie", "Gavin", "Noor",
         "Pete", "Callie", "Ivan", "June", "Marcus", "Tessa"]
LAST = ["Okafor", "Lindqvist", "Beaumont", "Reyes", "Tanaka", "Whitfield", "Novak",
        "Ferreira", "Aldrich", "Kaminski", "Boone", "Iyer", "Castellano", "Mercer",
        "Holt", "Vance", "Dupree", "Ashworth", "Calloway", "Strand", "Pemberton", "Rhee"]
CITIES = [("Columbus", "OH", "43219"), ("Bend", "OR", "97701"), ("Denver", "CO", "80216"),
          ("Mesa", "AZ", "85201"), ("Augusta", "GA", "30901"), ("Duluth", "MN", "55802"),
          ("Provo", "UT", "84601"), ("Erie", "PA", "16501"), ("Waco", "TX", "76701"),
          ("Salem", "OR", "97301"), ("Fresno", "CA", "93701"), ("Toledo", "OH", "43604")]
STREETS = ["Commerce Way", "Alder Court", "Foundry Rd", "Pike Street", "Harbor Blvd",
           "Mill Race Ln", "Cannery Row", "Summit Ave", "Juniper Dr", "Depot Street"]
CO_A = ["Meridian", "Apex", "Bluecrest", "Harborline", "Stonegate", "Cascade", "Ironwood",
        "Lakeshore", "Pinnacle", "Redbud", "Silverbell", "Northwind", "Copperfield",
        "Brightwell", "Oakhaven", "Fairbanks", "Kestrel", "Larkspur", "Marlowe", "Quarry"]
CO_B = ["Office Supply", "Components Inc.", "Logistics LLC", "Manufacturing Co.",
        "Distribution Group", "Industrial Parts", "Print Works", "Packaging Corp.",
        "Building Services", "Equipment Rental", "Foods Inc.", "Textiles Ltd."]
BANKS = ["First Meridian Bank", "Stonegate Savings Bank", "Harborline Credit Union",
         "Bluecrest National Bank", "Copperfield Trust Company", "Northwind Community Bank"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MOS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def person():
    return f"{R.choice(FIRST)} {R.choice(LAST)}"

def company():
    return f"{R.choice(CO_A)} {R.choice(CO_B)}"

def address():
    c = R.choice(CITIES)
    return f"{R.randrange(12, 9800)} {R.choice(STREETS)}, {c[0]}, {c[1]} {c[2]}", c

def a_date(y=None):
    y = y or R.choice((2025, 2026))
    m = R.randrange(12)
    day = R.randrange(1, 29)
    return f"{MONTHS[m]} {day}, {y}", (y, m, day)

def date_after(t, days):
    y, m, day = t
    day += days
    while day > 28:
        day -= 28
        m += 1
        if m > 11:
            m, y = 0, y + 1
    return f"{MONTHS[m]} {day}, {y}"

def money(cents):
    return f"${cents // 100:,}.{cents % 100:02d}"

def phone():
    return f"(555) 01{R.randrange(10, 99)}-{R.randrange(1000, 9999):04d}"

# Docs are collected as dicts:
#   {pages: [PIL], label, intro, ask: {name: value}, effect: clean|scan|hard|none}
DOCS = []

def add(pages, label, intro, ask, effect):
    DOCS.append({"pages": pages if isinstance(pages, list) else [pages],
                 "label": label, "intro": intro,
                 "ask": {k: str(v) for k, v in ask.items()}, "effect": effect})

def pick_effect(weights=(0.34, 0.40, 0.26)):
    r = R.random()
    return "clean" if r < weights[0] else ("scan" if r < weights[0] + weights[1] else "hard")

def trap(ask, name):
    """1-in-10 docs ask for a field the document genuinely does not carry."""
    if R.random() < 0.10:
        ask[name] = "not found"
    return ask

# ------------------------------------------------------------------ 15 invoices
def make_invoice(i):
    vend, cust = company(), (company() if R.random() < 0.6 else f"{person()} LLP")
    vaddr, _ = address()
    inv_no = f"INV-{R.randrange(10000, 99999)}"
    po_no = f"PO-{R.randrange(10000, 99999)}"
    idate, t = a_date()
    net = R.choice((15, 30, 45))
    ddate = date_after(t, net)
    items, sub = [], 0
    pool = [("Copy paper, letter, 10-ream case", 4250), ("Toner cartridge, black", 11800),
            ("Ergonomic desk chair", 18900), ("Whiteboard markers, 12-pack", 1350),
            ("Steel shelving unit, 72 in", 24400), ("Packing tape, 36-roll case", 5150),
            ("LED panel light, 2x4", 8925), ("Nitrile gloves, case of 1000", 6480),
            ("Shipping labels, 4x6, 8 rolls", 3220), ("Break room coffee, 5 lb", 4675),
            ("HDMI cable, 10 ft, 10-pack", 5590), ("Folding table, 6 ft", 7850)]
    for name, unit in R.sample(pool, R.randrange(3, 7)):
        qty = R.randrange(1, 12)
        amt = qty * unit
        sub += amt
        items.append((name, str(qty), f"{unit / 100:,.2f}", f"{amt / 100:,.2f}"))
    rate = R.randrange(40, 95) / 10          # 4.0-9.5 %
    tax = round(sub * rate / 100)
    total = sub + tax
    layout = i % 3
    img, d = page()
    if layout == 0:
        text(d, (64, 58), vend.upper(), SERIF(32))
        text(d, (64, 104), vaddr, SANS(17), MUT)
        text(d, (W - 64, 58), "INVOICE", BOLD(30), TEAL, anchor="ra")
        hline(d, 64, W - 64, 150, TEAL, 4)
    elif layout == 1:
        d.rectangle([0, 0, W, 132], fill=(38, 52, 74))
        text(d, (64, 44), vend.upper(), BOLD(30), (255, 255, 255))
        text(d, (W - 64, 52), "TAX INVOICE", SANS(20), (200, 214, 232), anchor="ra")
        text(d, (64, 148), vaddr, SANS(16), MUT)
    else:
        text(d, (W // 2, 58), vend.upper(), SERIF(30), anchor="ma")
        text(d, (W // 2, 104), "I N V O I C E", SANS(20), MUT, anchor="ma")
        hline(d, 220, W - 220, 140, INK, 2)
    kv(d, 530, 700, 180, "Invoice no.", inv_no)
    kv(d, 530, 700, 214, "Invoice date", idate)
    kv(d, 530, 700, 248, "Due date", ddate)
    kv(d, 530, 700, 282, "PO number", po_no)
    text(d, (64, 190), "Bill to", SANS(18), MUT)
    text(d, (64, 218), cust, BOLD(21))
    yy = table(d, 64, 340, ["Description", "Qty", "Unit price", "Amount"],
               [470, 80, 120, 102], items)
    kv(d, 560, 730, yy + 26, "Subtotal", money(sub))
    kv(d, 560, 730, yy + 60, f"Tax ({rate}%)", money(tax))
    hline(d, 560, W - 64, yy + 96)
    kv(d, 560, 730, yy + 110, "Total due", money(total), SANS(21), BOLD(24))
    text(d, (64, H - 90), f"Payment terms: Net {net}. Make checks payable to {vend}.",
         SANS(16), MUT)
    lbl = f"Invoice - {vend}"
    if R.random() < 0.28:
        word = R.choice(("PAST DUE", "PAID"))
        stamp(img, word, (R.randrange(120, 300), R.randrange(560, 780)),
              color=(196, 44, 44) if word == "PAST DUE" else (44, 128, 70))
        lbl += f" ({word.lower()} stamp)"
    ask = dict(R.sample([("Invoice number", inv_no), ("Invoice date", idate),
                         ("PO number", po_no), ("Total due", money(total)),
                         ("Due date", ddate), ("Vendor name", vend)], R.randrange(3, 5)))
    add(img, lbl, "This is a scanned business document.",
        trap(ask, "Remittance email"), pick_effect())

# ------------------------------------------------------------------ 12 photo receipts
def photoify(strip):
    """Photo-on-a-desk: warm background, keystone warp, shadow, vignette."""
    bg = Image.new("RGB", (W, H), (172, 148, 122))
    bd = ImageDraw.Draw(bg)
    for gy in range(0, H, R.randrange(38, 64)):     # rough wood grain
        c = R.randrange(-14, 14)
        bd.rectangle([0, gy, W, gy + 30], fill=(172 + c, 148 + c, 122 + c))
    bg = bg.filter(ImageFilter.GaussianBlur(3))
    rw, rh = strip.size
    strip = strip.convert("RGBA")
    dx = R.randrange(8, 26)
    # QUAD source corners in NW, SW, SE, NE order (else the strip mirrors)
    strip = strip.transform((rw + dx, rh + 12), Image.QUAD,
                            (0, 0, dx // 3, rh - 6, rw - dx // 2, rh, rw, dx // 2),
                            resample=Image.BICUBIC)
    strip = strip.rotate(R.uniform(-6, 6), expand=True, resample=Image.BICUBIC)
    scale = min(1.0, (H - 140) / strip.height)
    if scale < 1.0:
        strip = strip.resize((int(strip.width * scale), int(strip.height * scale)),
                             Image.LANCZOS)
    sh = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    px = (W - strip.width) // 2 + R.randrange(-30, 30)
    py = (H - strip.height) // 2 + R.randrange(-20, 20)
    sd.rounded_rectangle([px + 14, py + 16, px + strip.width + 6, py + strip.height + 10],
                         radius=24, fill=(20, 14, 8, 110))
    bg = Image.alpha_composite(bg.convert("RGBA"), sh.filter(ImageFilter.GaussianBlur(9)))
    bg.paste(strip, (px, py), strip)
    vig = Image.new("L", bg.size, 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse([-W // 3, -H // 3, W + W // 3, H + H // 3], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(120))
    dark = ImageEnhance.Brightness(bg.convert("RGB")).enhance(0.55)
    bg = Image.composite(bg.convert("RGB"), dark, vig)
    return bg.filter(ImageFilter.GaussianBlur(0.6))

def make_receipt(i):
    store = R.choice(["Corner Cafe", "Hilltop Grocery", "Fixit Hardware", "Union Diner",
                      "Petal & Stem Florist", "Northside Fuel Stop", "Bindery Books",
                      "Cactus Taqueria", "Polar Freeze Ice Cream", "Summit Outfitters"])
    _, c = address()
    idate, t = a_date(2026)
    hh, mm = R.randrange(7, 21), R.randrange(60)
    when = f"{idate}  {hh % 12 or 12}:{mm:02d} {'AM' if hh < 12 else 'PM'}"
    pool = [("Latte, oat milk", 575), ("Day-old bagels x3", 320), ("Wrench set, metric", 2499),
            ("Bouquet, seasonal", 1850), ("Paperback, used", 799), ("Carnitas taco x2", 690),
            ("Waffle cone, double", 645), ("Trail mix, 1 lb", 899), ("Duct tape", 549),
            ("House salad", 850), ("Regular unleaded, 9.2 gal", 3082), ("Sticker pack", 350),
            ("Coffee, drip, large", 320), ("Pie slice, cherry", 525)]
    items = R.sample(pool, R.randrange(3, 7))
    total = sum(p for _, p in items)
    card = R.random() < 0.7
    last4 = f"{R.randrange(1000, 9999)}"
    ink = (66, 66, 72)  # thermal grey, not black
    rh = 330 + 34 * len(items) + 150
    strip = Image.new("RGB", (400, rh), (250, 249, 246))
    d = ImageDraw.Draw(strip)
    cx = 200
    text(d, (cx, 34), store.upper(), BOLD(26), ink, anchor="ma")
    text(d, (cx, 74), f"{c[0]}, {c[1]}", MONO(15), ink, anchor="ma")
    text(d, (cx, 98), when, MONO(15), ink, anchor="ma")
    d.line([(24, 132), (376, 132)], fill=(180, 180, 176), width=2)
    y = 150
    for name, p in items:
        text(d, (28, y), name[:24], MONO(15), ink)
        text(d, (372, y), f"{p / 100:.2f}", MONO(15), ink, anchor="ra")
        y += 34
    d.line([(24, y + 4), (376, y + 4)], fill=(180, 180, 176), width=2)
    text(d, (28, y + 20), "TOTAL", MONO(20), ink)
    text(d, (372, y + 20), money(total), MONO(20), ink, anchor="ra")
    pay = f"VISA  ****{last4}   APPROVED" if card else "CASH TENDERED"
    text(d, (28, y + 62), pay, MONO(14), ink)
    text(d, (cx, y + 102), "Thank you!", MONO(14), ink, anchor="ma")
    for _ in range(3):  # thermal fade bands
        fy = R.randrange(120, rh - 60)
        band = strip.crop((0, fy, 400, min(rh, fy + R.randrange(16, 42))))
        strip.paste(ImageEnhance.Brightness(band).enhance(1.18), (0, fy))
    ask = {"Total": money(total), "Store name": store,
           ("Card last four digits" if card else "Payment method"):
               (last4 if card else "CASH TENDERED")}
    add(photoify(strip), f"Receipt - {store} (photo)", "This is a photo of a paper receipt.",
        trap(ask, "Cashier name"), "none")

# ------------------------------------------------------------------ 12 handwritten notes
def make_note(i):
    kind = i % 3
    fnt = CAVEAT if R.random() < 0.6 else APPLE
    ink = PEN_BLUE if R.random() < 0.7 else PENCIL
    if kind == 0:  # phone message on a printed pad
        img, d = lined_paper()
        d.rectangle([50, 40, W - 40, 118], fill=(240, 220, 150))
        text(d, (W // 2, 78), "WHILE YOU WERE OUT", BOLD(28), (110, 84, 20), anchor="mm")
        caller, org, num = person(), company(), phone()
        hh = R.randrange(8, 18)
        when = f"{hh % 12 or 12}:{R.randrange(0, 6)}0 {'AM' if hh < 12 else 'PM'}"
        msg = R.choice(["please call back today", "will try again tomorrow",
                        "re: the March order", "needs the signed contract",
                        "says the delivery slipped a week", "wants a revised quote"])
        rows = [("For:", person()), ("From:", caller), ("Of:", org),
                ("Phone:", num), ("Time:", when), ("Message:", msg)]
        y = 170
        for lab, val in rows:
            text(d, (130, y), lab, BOLD(20), MUT)
            hand(d, (270, y - 14), val, fnt(40), ink)
            y += 84
        ask = {"Caller name": caller, "Callback number": num, "Time of call": when}
        add(img, "Phone message (handwritten)", "This is a handwritten phone message slip.",
            trap(ask, "Case number"), pick_effect((0.5, 0.35, 0.15)))
    elif kind == 1:  # meeting notes
        img, d = lined_paper()
        proj = f"{R.choice(CO_A)} {R.choice(('rollout', 'redesign', 'migration', 'launch', 'audit'))}"
        mdate, _ = a_date(2026)
        hand(d, (130, 60), f"{proj} - notes", fnt(52), ink)
        hand(d, (W - 380, 130), mdate, fnt(40), ink)
        n_items = R.randrange(3, 7)
        y = 210
        hand(d, (130, y), "Discussed:", fnt(42), ink)
        y += 76
        for s in R.sample(["budget is tight until Q3", "vendor demo went fine",
                           "need sign-off from legal", "staging env still flaky",
                           "hire two more testers", "ship date holds for now"], 3):
            hand(d, (170, y), "- " + s, fnt(38), ink)
            y += 76
        hand(d, (130, y), "Action items:", fnt(42), ink)
        y += 76
        for k in range(n_items):
            owner = R.choice(FIRST)
            hand(d, (170, y), f"{k + 1}. {owner}: " +
                 R.choice(["draft the memo", "fix the login bug", "call the vendor",
                           "update the forecast", "book the room", "review the deck",
                           "send the invoice"]), fnt(38), ink)
            y += 76
        ask = {"Meeting date": mdate, "Number of action items": n_items}
        add(img, "Meeting notes (handwritten)", "This is a page of handwritten meeting notes.",
            trap(ask, "Conference room number"), pick_effect((0.5, 0.35, 0.15)))
    else:  # to-do list on graph paper
        img, d = lined_paper(graph=True)
        title = R.choice(["Saturday jobs", "Before the trip", "Shop tasks",
                          "Week 28 to-dos", "Garage cleanout"])
        hand(d, (90, 56), title, fnt(56), ink)
        n = R.randrange(5, 9)
        done = R.randrange(1, n)
        y = 190
        todo_pool = ["oil change", "return library books", "patch drywall", "back up laptop",
                     "order filters", "sharpen mower blade", "renew insurance", "clean gutters",
                     "pack toolbox", "drop off donations", "water heater flush", "fix gate latch"]
        for k, item in enumerate(R.sample(todo_pool, n)):
            d.rectangle([94, y + 6, 122, y + 34], outline=ink, width=3)
            if k < done:
                hand(d, (96, y - 10), "x", fnt(44), ink)
            hand(d, (140, y - 6), item, fnt(42), ink)
            y += 78
        ask = {"List title": title, "Total items": n, "Completed (checked) items": done}
        add(img, "To-do list (handwritten)",
            "This is a handwritten to-do list; checked boxes are completed.",
            trap(ask, "Page number"), pick_effect((0.5, 0.35, 0.15)))

# ------------------------------------------------------------------ 12 filled forms
def boxed(d, x, y, w, label, value, hand_it, fnt):
    text(d, (x, y), label, SANS(16), MUT)
    d.rectangle([x, y + 24, x + w, y + 66], outline=LINE, width=2)
    if hand_it:
        hand(d, (x + 12, y + 14), value, fnt(38), PEN_BLUE)
    else:
        text(d, (x + 12, y + 34), value, MONO(19))

def checkbox(d, x, y, label, checked, hand_it, fnt):
    d.rectangle([x, y, x + 26, y + 26], outline=INK, width=2)
    if checked:
        if hand_it:
            hand(d, (x + 2, y - 12), "x", fnt(40), PEN_BLUE)
        else:
            text(d, (x + 4, y - 1), "X", BOLD(22), PEN_BLUE)
    text(d, (x + 38, y + 2), label, SANS(18))

def make_form(i):
    hand_it = R.random() < 0.55
    fnt = CAVEAT
    img, d = page()
    kind = i % 4
    if kind == 0:  # job application
        co = company()
        text(d, (64, 50), co.upper(), BOLD(26))
        text(d, (64, 92), "Employment application", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        name = person()
        pos = R.choice(["Warehouse Associate", "Office Manager", "Delivery Driver",
                        "Line Cook", "Bookkeeper"])
        sal = f"${R.randrange(18, 42)}.00 / hour"
        start, _ = a_date(2026)
        boxed(d, 64, 170, 380, "Full name", name, hand_it, fnt)
        boxed(d, 480, 170, 356, "Phone", phone(), hand_it, fnt)
        boxed(d, 64, 270, 380, "Position applied for", pos, hand_it, fnt)
        boxed(d, 480, 270, 356, "Desired pay", sal, hand_it, fnt)
        boxed(d, 64, 370, 380, "Available start date", start, hand_it, fnt)
        ft = R.random() < 0.6
        text(d, (64, 490), "Employment type", SANS(16), MUT)
        checkbox(d, 64, 520, "Full-time", ft, hand_it, fnt)
        checkbox(d, 260, 520, "Part-time", not ft, hand_it, fnt)
        text(d, (64, 620), "I certify the information above is accurate.", SANS(17), MUT)
        text(d, (64, 680), "Signature", SANS(16), MUT)
        hline(d, 64, 440, 740)
        hand(d, (80, 686), name, APPLE(34), PEN_BLUE)
        ask = {"Applicant name": name, "Position applied for": pos,
               "Employment type (checked box)": "Full-time" if ft else "Part-time",
               "Desired pay": sal}
        lbl = f"Job application - {co}"
    elif kind == 1:  # insurance claim
        ins = R.choice(BANKS).replace("Bank", "Insurance").replace("Credit Union", "Mutual")
        text(d, (64, 50), ins.upper(), BOLD(26))
        text(d, (64, 92), "Property claim form", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        claim = f"CLM-{R.randrange(2020, 2027)}-{R.randrange(10000, 99999)}"
        holder = person()
        idate, _ = a_date()
        amt = money(R.randrange(20000, 2400000))
        boxed(d, 64, 170, 380, "Claim number", claim, False, fnt)
        boxed(d, 480, 170, 356, "Policyholder", holder, hand_it, fnt)
        boxed(d, 64, 270, 380, "Date of incident", idate, hand_it, fnt)
        boxed(d, 480, 270, 356, "Amount claimed", amt, hand_it, fnt)
        cause = R.choice(["Hail", "Water leak", "Theft", "Wind", "Fire"])
        text(d, (64, 390), "Cause of loss", SANS(16), MUT)
        cx = 64
        for c in ["Hail", "Water leak", "Theft", "Wind", "Fire"]:
            checkbox(d, cx, 420, c, c == cause, hand_it, fnt)
            cx += 160
        text(d, (64, 510), "Describe the damage:", SANS(16), MUT)
        d.rectangle([64, 540, W - 64, 760], outline=LINE, width=2)
        desc = R.choice(["roof and two windows on the north side",
                         "kitchen ceiling and cabinets", "detached garage door",
                         "basement carpet and drywall"])
        if hand_it:
            hand(d, (80, 552), "Damage to the " + desc + ".", fnt(38), PEN_BLUE)
        else:
            text(d, (80, 560), "Damage to the " + desc + ".", MONO(18))
        ask = {"Claim number": claim, "Policyholder": holder,
               "Amount claimed": amt, "Cause of loss (checked box)": cause}
        lbl = f"Insurance claim - {ins}"
    elif kind == 2:  # maintenance request
        text(d, (64, 50), "TENANT MAINTENANCE REQUEST", BOLD(26))
        text(d, (64, 92), R.choice(CO_A) + " Property Management", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        unit = f"{R.randrange(1, 48)}{R.choice('ABCD')}"
        who = person()
        rdate, _ = a_date(2026)
        issue = R.choice(["Garbage disposal jammed", "Bathroom fan rattles",
                          "Front door lock sticks", "Radiator not heating",
                          "Kitchen faucet drips"])
        pri = R.choice(["Low", "Normal", "Urgent"])
        boxed(d, 64, 170, 250, "Unit", unit, hand_it, fnt)
        boxed(d, 350, 170, 486, "Reported by", who, hand_it, fnt)
        boxed(d, 64, 270, 250, "Date reported", rdate, hand_it, fnt)
        boxed(d, 350, 270, 486, "Issue", issue, hand_it, fnt)
        text(d, (64, 390), "Priority", SANS(16), MUT)
        cx = 64
        for p in ["Low", "Normal", "Urgent"]:
            checkbox(d, cx, 420, p, p == pri, hand_it, fnt)
            cx += 180
        text(d, (64, 520), "Entry permission granted:  YES  /  NO", SANS(18))
        ask = {"Unit": unit, "Reported by": who, "Priority (checked box)": pri,
               "Date reported": rdate}
        lbl = "Maintenance request form"
    else:  # expense report
        co = company()
        text(d, (64, 50), co.upper(), BOLD(26))
        text(d, (64, 92), "Expense reimbursement form", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        emp, dept = person(), R.choice(["Sales", "Field Ops", "Finance", "Support"])
        rows, tot = [], 0
        for nm, lo, hi in [("Mileage", 1200, 9800), ("Meals", 1800, 9200),
                           ("Lodging", 9900, 32000), ("Parking", 600, 2400)]:
            if R.random() < 0.8:
                cts = R.randrange(lo, hi)
                tot += cts
                rows.append((nm, money(cts)))
        if not rows:
            rows, tot = [("Meals", money(4400))], 4400
        boxed(d, 64, 170, 380, "Employee", emp, hand_it, fnt)
        boxed(d, 480, 170, 356, "Department", dept, hand_it, fnt)
        y = 300
        for nm, v in rows:
            text(d, (64, y), nm, SANS(20))
            text(d, (520, y), v, MONO(20))
            y += 46
        hline(d, 64, 640, y + 6)
        text(d, (64, y + 22), "Total requested", BOLD(22))
        text(d, (520, y + 22), money(tot), BOLD(22))
        ask = {"Employee": emp, "Department": dept, "Total requested": money(tot)}
        lbl = f"Expense report - {co}"
    add(img, lbl, "This is a filled-in form.", trap(ask, "Approver name"), pick_effect())

# ------------------------------------------------------------------ 12 customer orders
def make_order(i):
    vend = company()
    ono = f"SO-{R.randrange(100000, 999999)}"
    odate, t = a_date(2026)
    sdate = date_after(t, R.randrange(3, 21))
    status = R.choice(["Processing", "Shipped", "Delivered", "On hold"])
    _, bc = address()
    _, sc = address()
    ship_addr = f"{R.randrange(12, 9800)} {R.choice(STREETS)}, {sc[0]}, {sc[1]} {sc[2]}"
    bill_addr = f"{R.randrange(12, 9800)} {R.choice(STREETS)}, {bc[0]}, {bc[1]} {bc[2]}"
    pool = [("ALU-3300", "Aluminum bracket, anodized", 2240), ("FST-1108", "Fastener kit, M6", 980),
            ("PLT-0904", "Steel base plate, 6 mm", 28872), ("GSK-2210", "Gasket set, EPDM", 1460),
            ("BRG-5521", "Ball bearing, sealed", 3320), ("HSE-077", "Hydraulic hose, 2 m", 5410),
            ("CBL-914", "Control cable, 5 m", 2170), ("VLV-208", "Check valve, brass", 4890)]
    items, units, tot = [], 0, 0
    for sku, name, unit in R.sample(pool, R.randrange(2, 6)):
        qty = R.randrange(5, 120)
        amt = qty * unit
        units += qty
        tot += amt
        items.append((sku, name, str(qty), f"{unit / 100:,.2f}", f"{amt / 100:,.2f}"))
    img, d = page()
    text(d, (64, 58), "SALES ORDER" if i % 2 else "ORDER CONFIRMATION", SERIF(32))
    text(d, (W - 64, 62), ono, BOLD(28), TEAL, anchor="ra")
    hline(d, 64, W - 64, 122, INK, 3)
    kv(d, 64, 250, 150, "Supplier", vend)
    kv(d, 64, 250, 186, "Order date", odate)
    kv(d, 64, 250, 222, "Ship by", sdate)
    kv(d, 64, 250, 258, "Status", status)
    text(d, (64, 310), "Bill to", SANS(17), MUT)
    text(d, (64, 336), bill_addr, SANS(18))
    text(d, (480, 310), "Ship to", SANS(17), MUT)
    text(d, (480, 336), ship_addr, SANS(18))
    yy = table(d, 64, 400, ["Item", "Description", "Qty", "Unit", "Amount"],
               [120, 330, 70, 110, 142], items)
    kv(d, 560, 720, yy + 30, "Order total", money(tot), SANS(21), BOLD(24))
    text(d, (64, yy + 30), f"Total units: {units}", SANS(19), MUT)
    text(d, (64, H - 100), f"Questions? Call {phone()} or write to orders@" +
         vend.split()[0].lower() + ".example.", SANS(16), MUT)
    ask = dict(R.sample([("Order number", ono), ("Order status", status),
                         ("Ship-to city", sc[0]), ("Order total", money(tot)),
                         ("Ship by date", sdate), ("Total units", str(units))],
                        R.randrange(3, 5)))
    add(img, f"Order - {vend}", "This is a scanned business document.",
        trap(ask, "Tracking number"), pick_effect())

# ------------------------------------------------------------------ 10 charts
def make_chart(i):
    kind = i % 4
    img, d = page()
    if kind == 0:  # labeled vertical bars
        co = R.choice(CO_A)
        title = f"{co} monthly revenue ($ thousands)"
        n = R.randrange(6, 9)
        start = R.randrange(12)
        labs = [MOS[(start + k) % 12] for k in range(n)]
        vals = [R.randrange(40, 185) for _ in range(n)]
        mi = vals.index(max(vals))
        vals[mi] += 12   # make the max unique so "tallest" has one right answer
        text(d, (W // 2, 70), title, BOLD(26), anchor="ma")
        x0, y0, y1 = 100, 180, 880
        x1 = W - 80
        d.line([(x0, y1), (x1, y1)], fill=INK, width=3)
        d.line([(x0, y0), (x0, y1)], fill=INK, width=3)
        for gv in range(0, 201, 50):
            gy = y1 - (y1 - y0) * gv / 200
            d.line([(x0, gy), (x1, gy)], fill=(230, 233, 237), width=1)
            text(d, (x0 - 12, gy), str(gv), SANS(16), MUT, anchor="rm")
        bw = (x1 - x0 - 40) / n
        for k, (lb, v) in enumerate(zip(labs, vals)):
            bx = x0 + 24 + k * bw
            by = y1 - (y1 - y0) * v / 200
            d.rectangle([bx, by, bx + bw * 0.66, y1], fill=(58, 122, 158))
            text(d, (bx + bw * 0.33, by - 10), str(v), BOLD(18), anchor="mb")
            text(d, (bx + bw * 0.33, y1 + 14), lb, SANS(17), anchor="ma")
        ask_month = R.choice([l for l in labs if l != labs[mi]])
        ask = {"Chart title": title, "Tallest bar month": labs[mi],
               f"Value printed above the {ask_month} bar": vals[labs.index(ask_month)]}
    elif kind == 1:  # pie with % legend
        co_pool = R.sample(CO_A, 4)
        title = "Regional market share, 2026"
        while True:
            shares = R.sample(range(6, 46), 3)
            shares.append(100 - sum(shares))
            if shares[-1] >= 5 and len(set(shares)) == 4:
                break
        big = co_pool[shares.index(max(shares))]
        text(d, (W // 2, 70), title, BOLD(26), anchor="ma")
        colors = [(58, 122, 158), (196, 120, 44), (94, 148, 88), (150, 96, 158)]
        a = -90
        for co, sh, col in zip(co_pool, shares, colors):
            a2 = a + 360 * sh / 100
            d.pieslice([170, 180, 730, 740], a, a2, fill=col,
                       outline=(252, 252, 251), width=3)
            a = a2
        y = 800
        for co, sh, col in zip(co_pool, shares, colors):
            d.rectangle([180, y, 210, y + 30], fill=col)
            text(d, (226, y + 2), f"{co} - {sh}%", SANS(20))
            y += 52
        other = R.choice([c for c in co_pool if c != big])
        ask = {"Chart title": title, "Largest share company": big,
               f"Share shown for {other}": f"{shares[co_pool.index(other)]}%"}
    elif kind == 2:  # line chart with peak label
        title = "Website visits by day (thousands)"
        labs = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        vals = [R.randrange(12, 95) for _ in labs]
        pi = vals.index(max(vals))
        vals[pi] += 8
        text(d, (W // 2, 70), title, BOLD(26), anchor="ma")
        x0, y0, y1 = 100, 180, 860
        x1 = W - 80
        d.line([(x0, y1), (x1, y1)], fill=INK, width=3)
        d.line([(x0, y0), (x0, y1)], fill=INK, width=3)
        pts = []
        for k, v in enumerate(vals):
            px = x0 + 40 + k * (x1 - x0 - 80) / 6
            py = y1 - (y1 - y0) * v / 110
            pts.append((px, py))
            text(d, (px, y1 + 14), labs[k], SANS(17), anchor="ma")
        d.line(pts, fill=(178, 84, 44), width=5, joint="curve")
        for k, (px, py) in enumerate(pts):
            d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(178, 84, 44))
            if k == pi:
                text(d, (px, py - 18), str(vals[k]), BOLD(20), anchor="mb")
        ask = {"Chart title": title, "Peak day": labs[pi],
               "Value printed at the peak": vals[pi]}
    else:  # horizontal bars, labeled
        title = "Units sold by product, Q2 2026"
        prods = R.sample(["Anchor kit", "Flow meter", "Relay board", "Hinge set",
                          "Pump seal", "Timer module"], 5)
        vals = [R.randrange(120, 940) for _ in prods]
        mi = vals.index(max(vals))
        vals[mi] += 40
        text(d, (W // 2, 70), title, BOLD(26), anchor="ma")
        y = 210
        for p, v in zip(prods, vals):
            text(d, (90, y + 12), p, SANS(19), anchor="lm")
            bw = 180 + (W - 420) * v / 1000
            d.rectangle([250, y, 250 + bw, y + 44], fill=(94, 148, 88))
            text(d, (258 + bw, y + 22), str(v), BOLD(19), anchor="lm")
            y += 96
        low = prods[vals.index(min(vals))]
        ask = {"Chart title": title, "Top product": prods[mi],
               f"Units shown for {low}": min(vals)}
    add(img, f"Chart - {ask['Chart title']}", "This page shows a chart; read it visually.",
        trap(ask, "Data source"), pick_effect((0.6, 0.4, 0.0)))

# ------------------------------------------------------------------ 8 banking contracts
FILLER = [
    "Each party shall bear its own costs and expenses incurred in connection with the "
    "preparation, negotiation, and execution of this Agreement, except as expressly "
    "provided herein.",
    "No amendment or waiver of any provision of this Agreement shall be effective unless "
    "made in writing and signed by both parties, and no failure to exercise any right "
    "shall operate as a waiver thereof.",
    "If any provision of this Agreement is held to be invalid or unenforceable, the "
    "remaining provisions shall continue in full force and effect, and the parties shall "
    "negotiate in good faith a substitute provision.",
    "The Borrower represents that the financial statements delivered to the Lender fairly "
    "present its financial condition and that no material adverse change has occurred "
    "since the date of those statements.",
    "All notices under this Agreement shall be delivered in writing to the addresses set "
    "out on the signature page and shall be deemed received three business days after "
    "posting by certified mail.",
    "The Borrower shall maintain insurance on the collateral in amounts and against risks "
    "customarily carried by companies engaged in similar businesses, naming the Lender as "
    "loss payee.",
    "Time is of the essence with respect to every obligation of the Borrower under this "
    "Agreement and each related loan document executed in connection herewith.",
]

def wrap_text(d, x, y, wpx, s, f, leading=30, fill=INK):
    words, line = s.split(), ""
    for wd in words:
        t = (line + " " + wd).strip()
        if d.textlength(t, font=f) > wpx and line:
            text(d, (x, y), line, f, fill)
            y += leading
            line = wd
        else:
            line = t
    if line:
        text(d, (x, y), line, f, fill)
        y += leading
    return y

def make_contract(i, npages):
    bank = R.choice(BANKS)
    borrower = company()
    adate, t = a_date()
    principal = money(R.randrange(2500000, 75000000))
    rate = f"{R.randrange(45, 129) / 10}%"
    term = R.choice((24, 36, 48, 60, 84))
    mdate = date_after(t, term)  # printed as-is; readers never derive it
    late = f"{R.choice((3, 4, 5))}%"
    state = R.choice(["Ohio", "Oregon", "Colorado", "Minnesota", "Texas"])
    officer, sig2 = person(), person()
    f18 = SANS(18)
    pages, rows = [], []

    def contract_page(no):
        img, d = page()
        text(d, (64, 44), bank.upper(), BOLD(20), MUT)
        text(d, (W - 64, 44), f"Page {no} of {npages}", SANS(16), MUT, anchor="ra")
        hline(d, 64, W - 64, 80)
        return img, d

    img, d = contract_page(1)
    text(d, (W // 2, 110), "TERM LOAN AGREEMENT", SERIF(30), anchor="ma")
    y = 190
    y = wrap_text(d, 64, y, W - 128,
                  f'This Term Loan Agreement (the "Agreement") is entered into as of {adate}, '
                  f'by and between {bank}, a banking corporation organized under the laws of '
                  f'the State of {state} (the "Lender"), and {borrower}, a company having its '
                  f'principal place of business as set out on the signature page (the '
                  f'"Borrower").', f18) + 16
    for s in R.sample(FILLER, 3):
        y = wrap_text(d, 64, y, W - 128, s, f18) + 16
    pages.append(img)
    rows.append(("p.1: parties and date",
                 {"Lender name": bank, "Borrower name": borrower, "Agreement date": adate}))

    img, d = contract_page(2)
    text(d, (64, 110), "ARTICLE II - THE LOAN", BOLD(22))
    y = 160
    y = wrap_text(d, 64, y, W - 128,
                  f"Subject to the terms of this Agreement, the Lender agrees to advance to "
                  f"the Borrower a term loan in the principal amount of {principal} (the "
                  f'"Loan"). The Loan shall bear interest on the outstanding principal '
                  f"balance at a fixed annual rate of {rate}, computed on the basis of a "
                  f"360-day year, and shall be repayable in {term} equal monthly installments, "
                  f'with the final installment due on {mdate} (the "Maturity Date").',
                  f18) + 16
    for s in R.sample(FILLER, 3):
        y = wrap_text(d, 64, y, W - 128, s, f18) + 16
    pages.append(img)
    rows.append(("p.2: loan terms",
                 {"Principal amount": principal, "Interest rate": rate,
                  "Number of monthly installments": str(term)}))

    if npages >= 3:
        img, d = contract_page(3)
        text(d, (64, 110), "ARTICLE V - COVENANTS AND DEFAULT", BOLD(22))
        y = 160
        y = wrap_text(d, 64, y, W - 128,
                      f"Any installment not received within ten days of its due date shall "
                      f"incur a late charge equal to {late} of the overdue amount. This "
                      f"Agreement shall be governed by and construed in accordance with the "
                      f"laws of the State of {state}, without regard to its conflict of laws "
                      f"principles.", f18) + 16
        for s in R.sample(FILLER, 3):
            y = wrap_text(d, 64, y, W - 128, s, f18) + 16
        pages.append(img)
        rows.append(("p.3: covenants",
                     {"Late charge": late, "Governing law state": state}))
    if npages >= 4:
        img, d = contract_page(4)
        text(d, (64, 110), "IN WITNESS WHEREOF, the parties have executed this Agreement.",
             SANS(19))
        y = 240
        for role, co, nm in (("LENDER", bank, officer), ("BORROWER", borrower, sig2)):
            text(d, (64, y), role + ":", BOLD(20), MUT)
            text(d, (64, y + 34), co, BOLD(21))
            hand(d, (80, y + 90), nm, APPLE(36), PEN_BLUE)
            hline(d, 64, 460, y + 150)
            text(d, (64, y + 162), f"Name: {nm}", SANS(18))
            title2 = "Senior Loan Officer" if role == "LENDER" else "Managing Director"
            text(d, (64, y + 192), f"Title: {title2}", SANS(18))
            y += 300
        pages.append(img)
        rows.append(("p.4: signatures",
                     {"Lender signatory name": officer, "Borrower signatory name": sig2}))

    DOCS.append({"pages": pages, "label": f"Loan agreement - {bank}",
                 "intro": "This is one page of a multi-page loan agreement.",
                 "ask": None, "effect": pick_effect((0.4, 0.45, 0.15)),
                 "page_rows": [(tag, {k: str(v) for k, v in a.items()})
                               for tag, a in rows]})

# ------------------------------------------------------------------ 10 records
def make_record(i):
    kind = i % 3
    img, d = page()
    if kind == 0:  # lab report
        lab = f"{R.choice(CO_A)} Clinical Laboratory"
        pat, dob = person(), a_date(R.randrange(1955, 2004))[0]
        cdate, _ = a_date(2026)
        acc = f"ACC-{R.randrange(100000, 999999)}"
        text(d, (64, 50), lab.upper(), BOLD(26))
        text(d, (64, 92), "Laboratory report", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        kv(d, 64, 260, 160, "Patient", pat)
        kv(d, 64, 260, 196, "Date of birth", dob)
        kv(d, 64, 260, 232, "Collected", cdate)
        kv(d, 530, 700, 160, "Accession", acc)
        tests = [("Glucose", str(R.randrange(70, 160)), "mg/dL", "70-99"),
                 ("Hemoglobin", f"{R.randrange(110, 175) / 10}", "g/dL", "13.5-17.5"),
                 ("WBC", f"{R.randrange(35, 120) / 10}", "10^3/uL", "4.5-11.0"),
                 ("Cholesterol", str(R.randrange(140, 260)), "mg/dL", "< 200")]
        table(d, 64, 300, ["Test", "Result", "Units", "Reference"],
              [280, 140, 160, 192], tests)
        ask = {"Patient": pat, "Collected": cdate, "Glucose result": tests[0][1],
               "Accession": acc}
        lbl = f"Lab report - {lab}"
    elif kind == 1:  # vehicle service record
        shop = f"{R.choice(CO_A)} Auto Service"
        veh = R.choice(["2019 Camry", "2021 F-150", "2017 Outback", "2022 Civic",
                        "2015 Tacoma"])
        odo = f"{R.randrange(18, 190) * 1000 + R.randrange(999):,} mi"
        sdate, _ = a_date()
        ro = f"RO-{R.randrange(10000, 99999)}"
        jobs, tot = [], 0
        for nm, lo, hi in [("Oil and filter change", 6900, 12900),
                           ("Brake pads, front", 18900, 34900),
                           ("Cabin air filter", 3900, 6900), ("Tire rotation", 2500, 4500),
                           ("Coolant flush", 10900, 16900)]:
            if R.random() < 0.6:
                c = R.randrange(lo, hi)
                tot += c
                jobs.append((nm, money(c)))
        if not jobs:
            jobs, tot = [("Oil and filter change", money(8900))], 8900
        text(d, (64, 50), shop.upper(), BOLD(26))
        text(d, (64, 92), "Service record", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        kv(d, 64, 260, 160, "Vehicle", veh)
        kv(d, 64, 260, 196, "Odometer", odo)
        kv(d, 64, 260, 232, "Service date", sdate)
        kv(d, 530, 700, 160, "Order no.", ro)
        y = 320
        for nm, v in jobs:
            text(d, (64, y), nm, SANS(20))
            text(d, (W - 64, y), v, SANS(20), anchor="ra")
            y += 46
        hline(d, 64, W - 64, y + 4)
        text(d, (64, y + 26), "Total", BOLD(24))
        text(d, (W - 64, y + 26), money(tot), BOLD(24), anchor="ra")
        ask = {"Odometer": odo, "Service date": sdate, "Total": money(tot), "Vehicle": veh}
        lbl = f"Service record - {shop}"
    else:  # employee record
        co = company()
        emp, eid = person(), f"E-{R.randrange(1000, 9999)}"
        hire, _ = a_date(R.randrange(2015, 2026))
        dept = R.choice(["Operations", "Finance", "Sales", "Maintenance", "Dispatch"])
        mgr = person()
        sal = money(R.randrange(4200000, 12800000))
        text(d, (64, 50), co.upper(), BOLD(26))
        text(d, (64, 92), "Employee record - confidential", SANS(20), MUT)
        hline(d, 64, W - 64, 130, INK, 3)
        rows = [("Employee", emp), ("Employee ID", eid), ("Department", dept),
                ("Hire date", hire), ("Reports to", mgr), ("Annual salary", sal),
                ("Status", "Active"), ("Location", R.choice(CITIES)[0])]
        y = 180
        for k2, v in rows:
            kv(d, 64, 320, y, k2, v)
            y += 52
        ask = {"Employee ID": eid, "Hire date": hire, "Annual salary": sal,
               "Reports to": mgr}
        lbl = f"Employee record - {co}"
    add(img, lbl, "This is a scanned record document.", trap(ask, "Fax number"),
        pick_effect())

# ------------------------------------------------------------------ 9 work orders
def make_workorder(i):
    co = f"{R.choice(CO_A)} Facilities"
    wo = f"WO-{R.randrange(2020, 2027)}-{R.randrange(1000, 9999)}"
    pri = R.choice(["Low", "Medium", "High", "Emergency"])
    tech = person()
    ddate, _ = a_date(2026)
    steps = R.sample(["Isolate power at panel", "Replace worn belt", "Grease bearings",
                      "Check alignment", "Test run 15 minutes", "Clear intake screen",
                      "Torque mounting bolts", "Update service tag", "Photograph nameplate",
                      "Verify guard reinstalled"], R.randrange(5, 9))
    done = R.randrange(1, len(steps))
    img, d = page()
    d.rectangle([0, 0, W, 120], fill=(120, 62, 30))
    text(d, (64, 40), "WORK ORDER", BOLD(30), (255, 255, 255))
    text(d, (W - 64, 46), wo, BOLD(24), (255, 224, 190), anchor="ra")
    text(d, (64, 150), co, SANS(20), MUT)
    kv(d, 64, 260, 200, "Assigned to", tech)
    kv(d, 64, 260, 236, "Priority", pri)
    kv(d, 64, 260, 272, "Due date", ddate)
    kv(d, 530, 700, 200, "Asset", f"AHU-{R.randrange(1, 24):02d}")
    text(d, (64, 340), "Checklist (checked = complete):", BOLD(20))
    y = 390
    for k, s in enumerate(steps):
        d.rectangle([70, y, 98, y + 28], outline=INK, width=3)
        if k < done:
            d.line([(74, y + 14), (82, y + 24)], fill=(30, 100, 50), width=4)
            d.line([(82, y + 24), (96, y + 2)], fill=(30, 100, 50), width=4)
        text(d, (116, y + 2), s, SANS(20))
        y += 54
    text(d, (64, y + 30), "Technician notes: parts on hand, no downtime expected.",
         SANS(18), MUT)
    ask = {"Work order number": wo, "Priority": pri, "Due date": ddate,
           "Completed (checked) steps": done, "Total steps": len(steps)}
    ask = dict(R.sample(list(ask.items()), 4))
    add(img, f"Work order {wo}",
        "This is a maintenance work order; checked boxes are complete.",
        trap(ask, "Supervisor signature"), pick_effect())

# ------------------------------------------------------------------ build everything
for i in range(15):
    make_invoice(i)
for i in range(12):
    make_receipt(i)
for i in range(12):
    make_note(i)
for i in range(12):
    make_form(i)
for i in range(12):
    make_order(i)
for i in range(10):
    make_chart(i)
for i, np_ in enumerate([4, 3, 2, 3, 4, 2, 3, 3]):
    make_contract(i, np_)
for i in range(10):
    make_record(i)
for i in range(9):
    make_workorder(i)

assert len(DOCS) == 100, f"expected 100 docs, built {len(DOCS)}"
R.shuffle(DOCS)

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "pdf"))

EFFECTS = {"clean": lambda im: im, "scan": scanify, "hard": hardify, "none": lambda im: im}
SUFFIX = {"clean": "", "scan": " (rough scan)", "hard": " (hard scan)", "none": ""}

rows, pageno = [], 0
counts = {"clean": 0, "scan": 0, "hard": 0, "photo": 0}
for docno, doc in enumerate(DOCS, 1):
    eff = doc["effect"]
    counts["photo" if "(photo)" in doc["label"] else eff] += 1
    out_pages = []
    for p in doc["pages"]:
        img = EFFECTS[eff](p)
        img.thumbnail((W + 80, H + 100))  # rotation expands; keep pages near canonical size
        out_pages.append(img.convert("RGB"))
    out_pages[0].save(os.path.join(OUT, "pdf", f"doc-{docno:03d}.pdf"), save_all=True,
                      append_images=out_pages[1:], resolution=96)
    if doc.get("page_rows"):   # contracts: one row per page, scoped questions
        for (tag, ask), img in zip(doc["page_rows"], out_pages):
            pageno += 1
            fn = f"page-{pageno:03d}.png"
            img.save(os.path.join(OUT, fn))
            rows.append({"file": fn, "label": f"{doc['label']}, {tag}{SUFFIX[eff]}",
                         "question": question(doc["intro"] +
                                              " Answer only from this page.", ask),
                         "fields": ask})
    else:
        pageno += 1
        fn = f"page-{pageno:03d}.png"
        out_pages[0].save(os.path.join(OUT, fn))
        rows.append({"file": fn, "label": doc["label"] + SUFFIX[eff],
                     "question": question(doc["intro"], doc["ask"]),
                     "fields": doc["ask"]})

manifest = {
    "id": "docs-stress-100",
    "name": "Stress set - 100 documents",
    "desc": "One hundred synthetic documents: invoices, photographed receipts, handwritten "
            "notes, forms, orders, charts, multi-page loan agreements, records, and work "
            "orders. Clean prints, rough scans, and hard-mode pages; one question in ten "
            "asks for a field that is not there.",
    "type": "doc",
    "suggest": 512,
    "rows": rows,
}
with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)

traps = sum(1 for r in rows if "not found" in r["fields"].values())
print(f"docs-stress-100: {len(DOCS)} docs, {len(rows)} graded pages, "
      f"{len(os.listdir(os.path.join(OUT, 'pdf')))} PDFs")
print(f"conditions (docs): {counts}   absent-field traps: {traps}")
