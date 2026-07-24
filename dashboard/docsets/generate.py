#!/usr/bin/env python3
"""Generate the built-in Document ingestion docsets (synthetic business pages + ground truth).

Each docset is a folder of PNG pages plus a manifest.json:
    {id, name, desc, type:"doc", suggest, rows:[{file, label, question, fields:{name:value}, ocr}]}

Every value in `fields` is exact ground truth known at render time, so grading never
depends on a judge model. Pages are deliberately spread across the three regimes the
NVIDIA RAG accuracy benchmarks identified (docs.nvidia.com/rag - accuracy-benchmarks):
prose-y pages where OCR keeps up, table/chart pages where visual structure matters,
and forms/stamps where the answer is not machine text at all.

Deterministic (fixed seed). Re-run any time: python3 generate.py
The optional `ocr` field per row is filled in afterwards by ocr_precompute.py.
"""
import json
import math
import os
import random
from PIL import Image, ImageDraw, ImageFilter

from _docgen import (W, H, INK, MUT, LINE, TEAL, SANS, BOLD, SERIF, MONO,
                     page, text, hline, kv as kv_right, table, stamp, question)

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(20260714)

def scanify(img, angle=1.4, speckle=900):
    """Make a rendered page look scanned: slight rotation, speckle, soft blur."""
    img = img.rotate(angle, expand=True, fillcolor=(246, 246, 244), resample=Image.BICUBIC)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(speckle):
        x, y = random.randrange(w), random.randrange(h)
        g = random.randrange(140, 225)
        d.point((x, y), fill=(g, g, g))
    img = img.filter(ImageFilter.GaussianBlur(0.55))
    return img

# ---------------------------------------------------------------- docset A: invoices
def a1_invoice():
    img, d = page()
    text(d, (64, 58), "MERIDIAN OFFICE SUPPLY", SERIF(34))
    text(d, (64, 104), "4410 Commerce Way, Columbus, OH 43219", SANS(17), MUT)
    text(d, (W - 64, 58), "INVOICE", BOLD(30), TEAL, anchor="ra")
    hline(d, 64, W - 64, 150, TEAL, 4)
    kv_right(d, 530, 690, 180, "Invoice no.", "INV-20417")
    kv_right(d, 530, 690, 214, "Invoice date", "March 12, 2026")
    kv_right(d, 530, 690, 248, "Due date", "April 11, 2026")
    kv_right(d, 530, 690, 282, "PO number", "PO-88213")
    text(d, (64, 180), "Bill to", SANS(18), MUT)
    text(d, (64, 208), "Harlow & Grant LLP", BOLD(21))
    text(d, (64, 238), "220 Fifth Avenue, Suite 900", SANS(18))
    text(d, (64, 264), "New York, NY 10001", SANS(18))
    rows = [("Copy paper, letter, 10-ream case", "6", "42.50", "255.00"),
            ("Toner cartridge, black, HP 58X", "4", "118.00", "472.00"),
            ("Ergonomic desk chair", "2", "189.00", "378.00"),
            ("Whiteboard markers, 12-pack", "9", "13.50", "121.50")]
    yy = table(d, 64, 340, ["Description", "Qty", "Unit price", "Amount"], [470, 80, 120, 102], rows)
    kv_right(d, 560, 730, yy + 26, "Subtotal", "$1,226.50")
    kv_right(d, 560, 730, yy + 60, "Tax (4.7%)", "$58.00")
    hline(d, 560, W - 64, yy + 96)
    kv_right(d, 560, 730, yy + 110, "Total due", "$1,284.50", SANS(21), BOLD(24))
    text(d, (64, H - 90), "Payment terms: Net 30. Make checks payable to Meridian Office Supply.", SANS(16), MUT)
    return img, "Invoice - Meridian Office Supply", {
        "Invoice number": "INV-20417", "Invoice date": "March 12, 2026",
        "PO number": "PO-88213", "Total due": "$1,284.50"}

def a2_utility():
    img, d = page()
    d.rectangle([0, 0, W, 130], fill=(23, 88, 129))
    text(d, (64, 44), "CITY WATER WORKS", BOLD(32), (255, 255, 255))
    text(d, (W - 64, 52), "Statement of account", SANS(19), (205, 224, 238), anchor="ra")
    kv_right(d, 64, 300, 176, "Account number", "4471-9902-05")
    kv_right(d, 64, 300, 212, "Service address", "18 Alder Court, Bend, OR")
    kv_right(d, 64, 300, 248, "Billing period", "May 1 - May 31, 2026")
    hline(d, 64, W - 64, 300)
    rows = [("Previous balance", "$81.20"), ("Payment received - thank you", "-$81.20"),
            ("Water service (14 CCF)", "$64.40"), ("Sewer service", "$18.90"), ("Stormwater fee", "$4.30")]
    y = 330
    for k, v in rows:
        text(d, (64, y), k, SANS(20))
        text(d, (W - 64, y), v, SANS(20), anchor="ra")
        y += 44
    hline(d, 64, W - 64, y + 4)
    text(d, (64, y + 28), "Amount due", BOLD(26))
    text(d, (W - 64, y + 28), "$87.60", BOLD(26), anchor="ra")
    text(d, (64, y + 74), "Due date: June 21, 2026", BOLD(21), (170, 60, 40))
    text(d, (64, H - 120), "Questions? Call 555-0142 weekdays 8am to 5pm, or visit citywater.example.", SANS(16), MUT)
    return img, "Utility bill - City Water Works", {
        "Account number": "4471-9902-05", "Amount due": "$87.60", "Due date": "June 21, 2026"}

def a3_receipt():
    img, d = page()
    x0, x1 = 260, 640
    d.rectangle([x0 - 20, 60, x1 + 20, 1000], fill=(255, 255, 254), outline=(228, 228, 224))
    cx = (x0 + x1) // 2
    text(d, (cx, 100), "CORNER CAFE", BOLD(30), INK, anchor="ma")
    text(d, (cx, 142), "812 Pike Street, Seattle, WA", SANS(16), MUT, anchor="ma")
    text(d, (cx, 168), "June 3, 2026  08:41 AM", SANS(16), MUT, anchor="ma")
    d.line([(x0, 208), (x1, 208)], fill=(190, 190, 186), width=2)
    items = [("Latte, oat milk, 16 oz", "5.75"), ("Breakfast burrito", "9.50"),
             ("Blueberry muffin", "4.25"), ("Orange juice", "4.35")]
    y = 232
    for k, v in items:
        text(d, (x0 + 8, y), k, MONO(18))
        text(d, (x1 - 8, y), v, MONO(18), anchor="ra")
        y += 38
    d.line([(x0, y + 6), (x1, y + 6)], fill=(190, 190, 186), width=2)
    text(d, (x0 + 8, y + 24), "TOTAL", MONO(22))
    text(d, (x1 - 8, y + 24), "$23.85", MONO(22), anchor="ra")
    text(d, (x0 + 8, y + 70), "VISA  ****4421   APPROVED", MONO(17))
    text(d, (cx, y + 130), "Thank you - see you tomorrow!", SANS(16), MUT, anchor="ma")
    img = scanify(img, angle=1.8, speckle=1400)
    return img, "Receipt - Corner Cafe (scanned)", {
        "Total": "$23.85", "Date": "June 3, 2026", "Card last four digits": "4421"}

def a4_po():
    img, d = page()
    text(d, (64, 58), "PURCHASE ORDER", SERIF(32))
    text(d, (W - 64, 62), "PO-77120", BOLD(28), TEAL, anchor="ra")
    hline(d, 64, W - 64, 122, INK, 3)
    kv_right(d, 64, 240, 150, "Supplier", "Apex Components Inc.")
    kv_right(d, 64, 240, 186, "Order date", "April 2, 2026")
    kv_right(d, 64, 240, 222, "Ship to", "Receiving Dock 4, 55 Foundry Rd")
    kv_right(d, 64, 240, 258, "", "Denver, CO 80216")
    rows = [("ALU-3300", "Aluminum bracket, anodized", "80", "22.40", "1,792.00"),
            ("FST-1108", "Fastener kit, M6 stainless", "40", "9.80", "392.00"),
            ("PLT-0904", "Steel base plate, 6 mm", "25", "288.72", "7,218.00")]
    yy = table(d, 64, 320, ["Item", "Description", "Qty", "Unit", "Amount"], [130, 340, 70, 110, 122], rows)
    kv_right(d, 560, 720, yy + 30, "Order total", "$9,402.00", SANS(21), BOLD(24))
    text(d, (64, yy + 30), "Total units: 145", SANS(19), MUT)
    text(d, (64, H - 130), "Authorized by: R. Calloway, Procurement", SANS(18))
    text(d, (64, H - 100), "Delivery required by April 24, 2026.", SANS(18))
    return img, "Purchase order - Apex Components", {
        "PO number": "PO-77120", "Supplier": "Apex Components Inc.",
        "Ship-to city": "Denver", "Order total": "$9,402.00"}

def a5_invoice2():
    img, d = page()
    d.rectangle([0, 0, 260, H], fill=(20, 60, 90))
    text(d, (36, 60), "NORTHWIND", BOLD(26), (255, 255, 255))
    text(d, (36, 96), "ENERGY LTD", BOLD(26), (140, 190, 226))
    text(d, (36, 170), "Billing services", SANS(15), (170, 200, 224))
    text(d, (36, H - 90), "northwind.example", SANS(14), (140, 170, 196))
    x = 320
    text(d, (x, 70), "Tax invoice", SERIF(34))
    kv_right(d, x, x + 220, 150, "Invoice no.", "NW-55302")
    kv_right(d, x, x + 220, 186, "Customer", "Brightline Studios")
    kv_right(d, x, x + 220, 222, "Issued", "April 18, 2026")
    rows = [("Electricity supply, March", "$296.40"), ("Green power option", "$21.10"), ("Network charge", "$24.68")]
    y = 300
    for k, v in rows:
        text(d, (x, y), k, SANS(20))
        text(d, (W - 64, y), v, SANS(20), anchor="ra")
        y += 46
    hline(d, x, W - 64, y + 6)
    text(d, (x, y + 30), "Amount due", BOLD(25))
    text(d, (W - 64, y + 30), "$342.18", BOLD(25), anchor="ra")
    text(d, (x, y + 80), "Pay by May 1, 2026 to avoid a late fee.", SANS(19), (170, 60, 40))
    text(d, (x, y + 140), "Direct debit reference: DD-90218", SANS(18), MUT)
    return img, "Tax invoice - Northwind Energy", {
        "Invoice number": "NW-55302", "Customer": "Brightline Studios",
        "Amount due": "$342.18", "Pay-by date": "May 1, 2026"}

# ------------------------------------------------------------ docset B: tables & charts
def b1_revenue():
    img, d = page()
    text(d, (64, 58), "Helio Systems - Quarterly revenue by region", SERIF(28))
    text(d, (64, 104), "Fiscal year 2025, USD millions", SANS(18), MUT)
    data = {"North America": [48.2, 51.6, 55.1, 61.3], "EMEA": [31.4, 30.9, 33.8, 36.2],
            "Asia Pacific": [22.7, 24.5, 27.9, 30.4], "Latin America": [9.1, 9.8, 10.6, 11.2]}
    rows = [(k, *[f"{v:.1f}" for v in vals], f"{sum(vals):.1f}") for k, vals in data.items()]
    q3_total = sum(v[2] for v in data.values())
    totals = [f"{sum(v[i] for v in data.values()):.1f}" for i in range(4)]
    rows.append(("Total", *totals, f"{sum(sum(v) for v in data.values()):.1f}"))
    table(d, 64, 170, ["Region", "Q1", "Q2", "Q3", "Q4", "FY25"], [280, 96, 96, 96, 96, 108], rows, rh=52)
    text(d, (64, 520), "Note: Q4 includes the Meridian contract renewal in North America.", SANS(17), MUT)
    text(d, (64, 580), "Prepared by Finance Operations, January 2026.", SANS(17), MUT)
    return img, "Revenue table - Helio Systems FY25", {
        "Q3 revenue all regions": f"{q3_total:.1f}", "EMEA full-year revenue": f"{sum(data['EMEA']):.1f}",
        "Region with highest Q4 revenue": "North America"}

def b2_barchart():
    img, d = page()
    text(d, (64, 58), "Retail unit sales by month", SERIF(30))
    text(d, (64, 106), "Store 114, first half of 2026", SANS(18), MUT)
    months = ["January", "February", "March", "April", "May", "June"]
    vals = [820, 640, 1180, 900, 760, 1010]
    x0, y0, x1, y1 = 110, 220, 830, 760
    for i in range(5):
        gy = y0 + (y1 - y0) * i / 4
        hline(d, x0, x1, int(gy), (230, 233, 237))
        text(d, (x0 - 14, gy), f"{1200 - 300 * i:,}", SANS(15), MUT, anchor="rm")
    bw = 74
    for i, (m, v) in enumerate(zip(months, vals)):
        bx = x0 + 40 + i * ((x1 - x0 - 60) / 6)
        bh = (v / 1200) * (y1 - y0)
        d.rectangle([bx, y1 - bh, bx + bw, y1], fill=(38, 128, 158))
        text(d, (bx + bw / 2, y1 + 16), m, SANS(15), INK, anchor="ma")
    d.line([(x0, y1), (x1, y1)], fill=INK, width=3)
    text(d, (64, 830), "Units per month. Source: point-of-sale exports, store 114.", SANS(16), MUT)
    return img, "Bar chart - monthly unit sales", {
        "Month with highest sales": "March", "Month with lowest sales": "February"}

def b3_expenses():
    img, d = page()
    text(d, (64, 58), "Operating expense breakdown", SERIF(30))
    text(d, (64, 106), "Crestway Logistics - fiscal 2025", SANS(18), MUT)
    data = [("Fleet fuel & maintenance", 412_000), ("Salaries & benefits", 1_240_000),
            ("Warehouse leases", 386_000), ("Insurance", 154_000),
            ("IT & software", 121_000), ("Marketing", 87_000)]
    tot = sum(v for _, v in data)
    rows = [(k, f"${v:,.0f}", f"{100 * v / tot:.1f}%") for k, v in data]
    rows.append(("Total operating expense", f"${tot:,.0f}", "100.0%"))
    table(d, 64, 170, ["Category", "FY25 spend", "Share of total"], [420, 200, 152], rows, rh=52)
    text(d, (64, 620), "Salaries & benefits remain the largest category for the third year running.", SANS(18))
    text(d, (64, 660), "Fuel spend fell 6% year over year on route optimization.", SANS(18))
    return img, "Expense table - Crestway Logistics", {
        "Largest expense category": "Salaries & benefits", "Largest category spend": "$1,240,000",
        "Share of total for largest category": "51.7%"}

def b4_pie():
    img, d = page()
    text(d, (64, 58), "Market share, smart sensor segment", SERIF(29))
    text(d, (64, 106), "Calendar 2025, by revenue", SANS(18), MUT)
    segs = [("Veltrix", 38, (38, 128, 158)), ("Orbita", 27, (222, 148, 68)),
            ("Kastel", 21, (94, 160, 90)), ("Others", 14, (150, 150, 156))]
    cx, cy, r = 340, 480, 230
    a = -90.0
    for name, pct, col in segs:
        a2 = a + 360 * pct / 100
        d.pieslice([cx - r, cy - r, cx + r, cy + r], a, a2, fill=col, outline=(252, 252, 251), width=4)
        mid = math.radians((a + a2) / 2)
        d.text((cx + 0.62 * r * math.cos(mid), cy + 0.62 * r * math.sin(mid)),
               f"{pct}%", font=BOLD(24), fill=(255, 255, 255), anchor="mm")
        a = a2
    ly = 340
    for name, pct, col in segs:
        d.rectangle([640, ly, 668, ly + 28], fill=col)
        text(d, (684, ly + 14), name, SANS(21), INK, anchor="lm")
        ly += 54
    text(d, (64, 800), "Shares are rounded to the nearest percent.", SANS(16), MUT)
    return img, "Pie chart - sensor market share", {
        "Company with the largest share": "Veltrix", "Largest company share": "38%",
        "Kastel share": "21%"}

def b5_pricing():
    img, d = page()
    text(d, (64, 58), "Stackline platform pricing", SERIF(30))
    text(d, (64, 106), "Billed monthly, per workspace", SANS(18), MUT)
    cols = ["", "Basic", "Team", "Business", "Enterprise"]
    rows = [("Price per month", "$0", "$29", "$79", "$249"),
            ("Included seats", "3", "10", "25", "Unlimited"),
            ("Storage", "5 GB", "200 GB", "2 TB", "10 TB"),
            ("Audit log", "-", "-", "Yes", "Yes"),
            ("Support", "Community", "Email", "Priority", "Dedicated")]
    table(d, 64, 180, cols, [220, 130, 130, 140, 152], rows, rh=56)
    text(d, (64, 560), "Annual billing saves 20% on every paid plan.", SANS(18))
    text(d, (64, 600), "All plans include unlimited viewers and two-factor sign-in.", SANS(18), MUT)
    return img, "Pricing grid - Stackline platform", {
        "Business plan monthly price": "$79", "Plan that includes 2 TB storage": "Business",
        "Included seats on the Team plan": "10"}

# ------------------------------------------------------------ docset C: forms & records
def c1_application():
    img, d = page()
    text(d, (64, 58), "Employment application", SERIF(30))
    text(d, (64, 106), "Lakeshore Property Group - Human Resources", SANS(18), MUT)
    def box(y, label, value, wbox=W - 128):
        text(d, (64, y), label, SANS(16), MUT)
        d.rectangle([64, y + 26, 64 + wbox, y + 74], outline=LINE, width=2)
        text(d, (78, y + 50), value, SANS(21), INK, anchor="lm")
        return y + 96
    y = box(170, "Full name", "Dana R. Whitfield")
    y = box(y, "Position applied for", "Facilities Manager")
    y = box(y, "Phone", "(503) 555-0187", 380)
    text(d, (64 + 420, y - 96), "Date", SANS(16), MUT)
    d.rectangle([64 + 420, y - 70, W - 64, y - 22], outline=LINE, width=2)
    text(d, (64 + 434, y - 46), "July 6, 2026", SANS(21), INK, anchor="lm")
    y = box(y, "Most recent employer", "Cascade Facilities Services")
    text(d, (64, y + 10), "Available to start immediately:   Yes", SANS(19))
    text(d, (64, H - 130), "Signature: D. Whitfield", SERIF(22))
    return img, "Form - employment application", {
        "Applicant name": "Dana R. Whitfield", "Position applied for": "Facilities Manager",
        "Phone": "(503) 555-0187", "Most recent employer": "Cascade Facilities Services"}

def c2_checkboxes():
    img, d = page()
    text(d, (64, 58), "Building service request", SERIF(30))
    text(d, (64, 106), "Complete one form per location. Mark all services required.", SANS(18), MUT)
    kv_right(d, 64, 250, 170, "Location", "Warehouse B, Mezzanine")
    kv_right(d, 64, 250, 206, "Requested by", "K. Osei")
    text(d, (64, 280), "Services required (check all that apply)", BOLD(19))
    opts = [("Electrical", True), ("Plumbing", False), ("HVAC", True),
            ("Roofing", False), ("Painting", False), ("Pest control", False)]
    y = 330
    for name, checked in opts:
        d.rectangle([70, y, 102, y + 32], outline=INK, width=3)
        if checked:
            d.line([(76, y + 16), (86, y + 26)], fill=INK, width=5)
            d.line([(86, y + 26), (98, y + 4)], fill=INK, width=5)
        text(d, (120, y + 16), name, SANS(21), INK, anchor="lm")
        y += 58
    text(d, (64, y + 20), "Priority", BOLD(19))
    for i, (p, on) in enumerate([("Routine", False), ("Urgent", True), ("Emergency", False)]):
        px = 64 + i * 220
        d.ellipse([px, y + 54, px + 30, y + 84], outline=INK, width=3)
        if on:
            d.ellipse([px + 8, y + 62, px + 22, y + 76], fill=INK)
        text(d, (px + 44, y + 69), p, SANS(20), INK, anchor="lm")
    return img, "Form - service request checkboxes", {
        "First service checked": "Electrical", "Second service checked": "HVAC",
        "Priority selected": "Urgent"}

def c3_label():
    img, d = page()
    d.rectangle([90, 120, W - 90, 940], outline=INK, width=5)
    d.rectangle([90, 120, W - 90, 240], fill=INK)
    text(d, (120, 180), "PRIORITY  2-DAY AIR", BOLD(34), (255, 255, 255), anchor="lm")
    kv_right(d, 130, 340, 290, "Tracking no.", "TRK-4415-8890-22", SANS(20), MONO(26))
    kv_right(d, 130, 340, 350, "Weight", "12.4 kg", SANS(20), BOLD(24))
    kv_right(d, 130, 340, 410, "Service", "2-Day Air", SANS(20), BOLD(24))
    hline(d, 130, W - 130, 470)
    text(d, (130, 500), "SHIP TO", BOLD(18), MUT)
    text(d, (130, 536), "Juniper Analytics", BOLD(27))
    text(d, (130, 580), "900 SW Salmon Street, Floor 6", SANS(22))
    text(d, (130, 616), "Portland, OR 97205", BOLD(24))
    for i in range(72):
        bx = 130 + i * 9
        bw2 = 3 if i % 3 else 6
        d.rectangle([bx, 720, bx + bw2, 880], fill=INK)
    img = scanify(img, angle=-1.1, speckle=700)
    return img, "Shipping label (scanned)", {
        "Tracking number": "TRK-4415-8890-22", "Weight": "12.4 kg",
        "Destination city": "Portland"}

def c4_badge():
    img, d = page()
    x0, y0, x1, y1 = 150, 220, W - 150, 830
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, outline=INK, width=4, fill=(255, 255, 255))
    d.rectangle([x0, y0, x1, y0 + 96], fill=(20, 60, 90))
    text(d, ((x0 + x1) / 2, y0 + 48), "HALCYON RESEARCH GROUP", BOLD(26), (255, 255, 255), anchor="mm")
    d.rectangle([x0 + 50, y0 + 150, x0 + 230, y0 + 330], fill=(228, 232, 236), outline=LINE)
    text(d, (x0 + 140, y0 + 240), "PHOTO", SANS(19), MUT, anchor="mm")
    kv_right(d, x0 + 280, x0 + 470, y0 + 160, "Name", "Priya Raman")
    kv_right(d, x0 + 280, x0 + 470, y0 + 210, "Employee ID", "EMP-30291")
    kv_right(d, x0 + 280, x0 + 470, y0 + 260, "Department", "Research & Development")
    kv_right(d, x0 + 280, x0 + 470, y0 + 310, "Clearance", "Level 3")
    kv_right(d, x0 + 50, x0 + 240, y0 + 400, "Issued", "February 9, 2026")
    kv_right(d, x0 + 50, x0 + 240, y0 + 450, "Expires", "February 9, 2028")
    text(d, ((x0 + x1) / 2, y1 - 60), "Property of Halcyon Research Group. If found, return to any office.",
         SANS(15), MUT, anchor="mm")
    return img, "Employee record card", {
        "Employee ID": "EMP-30291", "Department": "Research & Development",
        "Clearance": "Level 3", "Expiry date": "February 9, 2028"}

def c5_memo():
    img, d = page()
    text(d, (64, 58), "INTERNAL MEMO", SERIF(30))
    hline(d, 64, W - 64, 116, INK, 3)
    kv_right(d, 64, 220, 140, "To", "All department heads")
    kv_right(d, 64, 220, 176, "From", "M. Adeyemi, Operations")
    kv_right(d, 64, 220, 212, "Date", "May 22, 2026")
    kv_right(d, 64, 220, 248, "Subject", "Vendor consolidation plan")
    body = ["Following the Q1 review, procurement will consolidate our hardware",
            "vendors from nine suppliers to four by the end of Q3. Framework",
            "agreements are being finalized with the remaining suppliers now.",
            "",
            "Department heads should submit any sole-source exceptions to",
            "procurement by June 15, 2026. Orders placed after July 1 must use",
            "the new vendor list in the purchasing portal."]
    y = 320
    for ln in body:
        text(d, (64, y), ln, SANS(20))
        y += 36
    stamp(img, "APPROVED", (430, 620))
    d = ImageDraw.Draw(img)
    text(d, (64, H - 120), "Distribution: operations, finance, facilities", SANS(16), MUT)
    return img, "Memo with approval stamp", {
        "Memo subject": "Vendor consolidation plan", "Memo date": "May 22, 2026",
        "Approval status": "APPROVED", "Exception deadline": "June 15, 2026"}

# ---------------------------------------------------------------- assembly
DOCSETS = [
    {"id": "docs-invoices", "name": "Invoices & receipts",
     "desc": "Five billing documents, clean prints and rough scans. Reads like real accounts-payable intake.",
     "intro": "This is a scanned business document.",
     "pages": [a1_invoice, a2_utility, a3_receipt, a4_po, a5_invoice2]},
    {"id": "docs-tables", "name": "Tables & charts",
     "desc": "Financial tables, a bar chart, a pie chart and a pricing grid. The answers live in visual structure, not plain text.",
     "intro": "This is a page from a business report.",
     "pages": [b1_revenue, b2_barchart, b3_expenses, b4_pie, b5_pricing]},
    {"id": "docs-forms", "name": "Forms & records",
     "desc": "Filled-in forms, checkboxes, a shipping label and a stamped memo. Tests what a reader sees, not just the words.",
     "intro": "This is a scanned form or record.",
     "pages": [c1_application, c2_checkboxes, c3_label, c4_badge, c5_memo]},
]

def main():
    for ds in DOCSETS:
        outdir = os.path.join(HERE, ds["id"])
        os.makedirs(outdir, exist_ok=True)
        rows = []
        for i, fn in enumerate(ds["pages"], 1):
            img, label, fields = fn()
            name = f"page-{i}.png"
            img.save(os.path.join(outdir, name), optimize=True)
            rows.append({"file": name, "label": label,
                         "question": question(ds["intro"], fields), "fields": fields})
            print(f"  {ds['id']}/{name}  {label}  ({len(fields)} fields)")
        manifest = {"id": ds["id"], "name": ds["name"], "desc": ds["desc"],
                    "type": "doc", "suggest": 512,
                    "order": DOCSETS.index(ds) + 1, "rows": rows}
        with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print("done")

if __name__ == "__main__":
    main()
