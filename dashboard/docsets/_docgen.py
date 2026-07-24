"""Shared, RNG-free rendering helpers for the docset generators.

Everything here is deterministic: font factories, page geometry, primitive draw
helpers, the ruled table used by generate.py and generate_stress.py, the rubber
stamp, and the extraction-question builder. Anything that consumes a generator's
seeded RNG stream (scanify/hardify/photoify/hand/lined_paper) or that differs
between generators (fictional-data lists, PEN_BLUE, shared table drawing)
stays in its own module so the committed docsets keep reproducing byte-for-byte.
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

F = "/usr/share/fonts/truetype/dejavu/"
def font(name, size):
    return ImageFont.truetype(F + name, size)

SANS = lambda s: font("DejaVuSans.ttf", s)
BOLD = lambda s: font("DejaVuSans-Bold.ttf", s)
SERIF = lambda s: font("DejaVuSerif-Bold.ttf", s)
MONO = lambda s: font("DejaVuSansMono.ttf", s)
CAVEAT = lambda s: ImageFont.truetype(os.path.join(HERE, "fonts", "Caveat.ttf"), s)
APPLE = lambda s: ImageFont.truetype(os.path.join(HERE, "fonts", "HomemadeApple.ttf"), s)

W, H = 900, 1150
INK = (28, 32, 38)
MUT = (105, 112, 122)
LINE = (208, 213, 220)
TEAL = (26, 122, 133)

def page(bg=(252, 252, 251)):
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)

def text(d, xy, s, f, fill=INK, anchor=None):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)

def hline(d, x0, x1, y, fill=LINE, w=2):
    d.line([(x0, y), (x1, y)], fill=fill, width=w)

def kv(d, x_lab, x_val, y, label, value, fl=None, fv=None):
    text(d, (x_lab, y), label, fl or SANS(20), MUT)
    text(d, (x_val, y), value, fv or BOLD(20))

def table(d, x, y, cols, widths, rows, rh=44, header_bg=(238, 241, 244), fs=19):
    """Simple ruled table. cols = header labels, rows = list of value lists."""
    x1 = x + sum(widths)
    d.rectangle([x, y, x1, y + rh], fill=header_bg)
    cx = x
    for c, w in zip(cols, widths):
        text(d, (cx + 12, y + rh / 2), c, BOLD(fs - 1), (60, 66, 74), anchor="lm")
        cx += w
    yy = y + rh
    for r in rows:
        cx = x
        for v, w in zip(r, widths):
            text(d, (cx + 12, yy + rh / 2), str(v), SANS(fs), INK, anchor="lm")
            cx += w
        hline(d, x, x1, yy)
        yy += rh
    hline(d, x, x1, y)
    hline(d, x, x1, yy)
    cx = x
    for w in list(widths) + [0]:
        d.line([(cx, y), (cx, yy)], fill=LINE, width=2)
        cx += w
    return yy

def stamp(img, s, xy, color=(196, 44, 44), size=64, angle=18):
    """Diagonal rubber stamp overlaid on the page."""
    tmp = Image.new("RGBA", (620, 170), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.rounded_rectangle([4, 4, 616, 166], radius=18, outline=color + (215,), width=7)
    td.text((310, 82), s, font=BOLD(size), fill=color + (210,), anchor="mm")
    tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    img.paste(tmp, xy, tmp)

QTAIL = (' Answer with one line per field in the form "Field name: value". '
         'If a field is not on the document, write "not found".')

def question(intro, fields):
    """Build the per-page extraction question from the ground-truth field names."""
    names = "; ".join(fields)
    return f"{intro} Extract these fields exactly as printed: {names}.{QTAIL}"
