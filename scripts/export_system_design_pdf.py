"""Write NSF_System_Design.pdf into the project root.

Produces a two-part System Design document, each part with a drawn diagram:
  1. High-Level System Design  - the big picture (silos -> federation ->
     global model -> neuro-symbolic inference -> explanation -> dashboard).
  2. Detailed System Design     - the per-record pipeline, the decision /
     override logic, the federated training loop, and the dashboard views.

Core (Helvetica) fonts only, so all text is ASCII; ">=" and "<" are written
literally where needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpdf import FPDF

OUT = ROOT / "NSF_System_Design.pdf"

INK = (26, 35, 50)
MUTED = (80, 90, 102)
TEAL = (12, 92, 86)
TEAL_SOFT = (229, 243, 241)
AMBER = (176, 120, 20)
AMBER_SOFT = (252, 243, 224)
BLUE = (34, 78, 128)
BLUE_SOFT = (229, 238, 248)
BOX = (240, 246, 245)
WHITE = (255, 255, 255)


class DesignPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "NSF system design (high-level and detailed)",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()}  |  Domain-agnostic federated neuro-symbolic framework",
                  align="C")


def heading(pdf, text):
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_color(*INK)


def para(pdf, text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.3, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.6)


def bullets(pdf, items):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    for item in items:
        pdf.multi_cell(0, 5.3, f"-  {item}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.6)
    pdf.ln(1.2)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def box(pdf, x, y, w, h, title, sub=None, fill=TEAL_SOFT, ink=INK,
        border=TEAL, title_size=9.0, sub_size=7.3):
    """Draw a rounded-ish rectangle with a centered title and optional subtitle."""
    pdf.set_draw_color(*border)
    pdf.set_line_width(0.3)
    pdf.set_fill_color(*fill)
    pdf.rect(x, y, w, h, "DF")
    if sub:
        pdf.set_xy(x, y + h / 2 - 4.2)
    else:
        pdf.set_xy(x, y + h / 2 - 2.5)
    pdf.set_font("Helvetica", "B", title_size)
    pdf.set_text_color(*ink)
    pdf.multi_cell(w, 4.2, title, align="C")
    if sub:
        pdf.set_xy(x, y + h / 2 + 0.6)
        pdf.set_font("Helvetica", "", sub_size)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(w, 3.6, sub, align="C")


def varrow(pdf, x, y1, y2, color=TEAL, label=None):
    """Vertical arrow pointing down from y1 to y2."""
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.line(x, y1, x, y2)
    pdf.line(x, y2, x - 1.6, y2 - 2.4)
    pdf.line(x, y2, x + 1.6, y2 - 2.4)
    if label:
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*MUTED)
        pdf.set_xy(x + 2, (y1 + y2) / 2 - 2)
        pdf.cell(60, 4, label)


def line_arrow(pdf, x1, y1, x2, y2, color=TEAL):
    """Straight arrow from (x1,y1) to (x2,y2) with a small arrowhead at the end."""
    import math
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.4)
    pdf.line(x1, y1, x2, y2)
    ang = math.atan2(y2 - y1, x2 - x1)
    a = 0.5
    ln = 2.6
    pdf.line(x2, y2, x2 - ln * math.cos(ang - a), y2 - ln * math.sin(ang - a))
    pdf.line(x2, y2, x2 - ln * math.cos(ang + a), y2 - ln * math.sin(ang + a))


# ---------------------------------------------------------------------------
# Diagram 1: high-level system design
# ---------------------------------------------------------------------------

def draw_highlevel(pdf):
    top = pdf.get_y() + 2
    page_w = 210
    cx = page_w / 2

    # Row of 3 client silos
    cw, ch = 50, 15
    gap = 9
    total = cw * 3 + gap * 2
    x0 = (page_w - total) / 2
    client_y = top
    labels = ["Client 1 silo", "Client 2 silo", "Client N silo"]
    for i, lab in enumerate(labels):
        bx = x0 + i * (cw + gap)
        box(pdf, bx, client_y, cw, ch, lab, "private raw data", BLUE_SOFT, INK, BLUE)

    # Federated server
    sw, sh = 120, 17
    sx = cx - sw / 2
    sy = client_y + ch + 16
    box(pdf, sx, sy, sw, sh,
        "Federated Server",
        "FedAvg / FedProx / FedAdagrad  (+ optional DP noise)",
        TEAL_SOFT, INK, TEAL, title_size=10)

    # Converging arrows from clients to server
    for i in range(3):
        bx = x0 + i * (cw + gap) + cw / 2
        line_arrow(pdf, bx, client_y + ch, cx, sy, BLUE)
    # privacy annotation (placed to the right of the server box)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*BLUE)
    pdf.set_xy(sx + sw + 2, sy + 2)
    pdf.multi_cell(28, 3.6, "only model\nweights shared\n(no raw data)")

    # Vertical chain below the server
    bw, bh = 120, 15
    bx = cx - bw / 2
    chain = [
        ("Global Neural Model", "shared trained MLP", TEAL_SOFT),
        ("Neuro-Symbolic Inference", "neural prediction + rule verification + override", TEAL_SOFT),
        ("Verified Explanation", "fact-locked, hallucination-free", TEAL_SOFT),
        ("Dashboard", "Coordinator view  +  Client login view", AMBER_SOFT),
    ]
    y = sy + sh
    for i, (t, s, f) in enumerate(chain):
        y_box = y + 10
        varrow(pdf, cx, y, y_box)
        box(pdf, bx, y_box, bw, bh, t, s, f, INK, TEAL if f is TEAL_SOFT else AMBER)
        y = y_box + bh

    pdf.set_y(y + 3)


# ---------------------------------------------------------------------------
# Diagram 2: detailed system design
# ---------------------------------------------------------------------------

def draw_detailed(pdf):
    top = pdf.get_y() + 2
    page_w = 210

    # Main spine on the left, federated-training panel on the right.
    spine_w = 100
    spine_x = 15
    cx = spine_x + spine_w / 2

    bh = 13
    gap = 8

    spine = [
        ("Input record", "patient vitals / transaction fields", BOX),
        ("Standardize features", "train mean / std", TEAL_SOFT),
        ("Layer 1: PerceptionMLP", "-> class probabilities", TEAL_SOFT),
        ("Layer 2: Neural-to-Symbolic Bridge", "typed SymbolicFact", TEAL_SOFT),
        ("Layer 3: Symbolic Reasoner", "score domain rules", TEAL_SOFT),
    ]
    y = top
    for t, s, f in spine:
        box(pdf, spine_x, y, spine_w, bh, t, s, f, INK, TEAL)
        varrow(pdf, cx, y + bh, y + bh + gap)
        y += bh + gap

    # Decision-logic block (amber, taller, multi-line rules inside)
    dh = 26
    pdf.set_draw_color(*AMBER)
    pdf.set_line_width(0.4)
    pdf.set_fill_color(*AMBER_SOFT)
    pdf.rect(spine_x, y, spine_w, dh, "DF")
    pdf.set_xy(spine_x, y + 1.6)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(spine_w, 4.2, "Decision logic (symbolic vs neural)", align="C")
    pdf.set_xy(spine_x + 3, y + 7)
    pdf.set_font("Helvetica", "", 7.6)
    pdf.set_text_color(*INK)
    pdf.multi_cell(spine_w - 6, 3.9,
                   "- agree -> use that class\n"
                   "- conflict + hard safety rule OR confidence < 0.55 -> override to symbolic\n"
                   "- otherwise -> keep neural class and log the conflict",
                   align="L")
    varrow(pdf, cx, y + dh, y + dh + gap)
    y += dh + gap

    tail = [
        ("Layer 5: Verified Explainer + guard", "verified facts ONLY", TEAL_SOFT),
        ("Output", "decision + explanation + confidence + warnings + trace", AMBER_SOFT),
    ]
    for i, (t, s, f) in enumerate(tail):
        box(pdf, spine_x, y, spine_w, bh, t, s, f, INK,
            TEAL if f is TEAL_SOFT else AMBER)
        if i < len(tail) - 1:
            varrow(pdf, cx, y + bh, y + bh + gap)
            y += bh + gap
        else:
            y += bh

    # ---- Federated training panel on the right ----
    pw = 68
    px = page_w - 15 - pw
    py = top
    ph = 78
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.3)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(px, py, pw, ph, "D")
    pdf.set_xy(px, py + 1.6)
    pdf.set_font("Helvetica", "B", 8.6)
    pdf.set_text_color(*TEAL)
    pdf.multi_cell(pw, 4, "Layer 4: Federated training\n(produces the model)", align="C")

    # three client-train boxes -> aggregation -> global model
    inpad = 5
    cbw = pw - 2 * inpad
    cbx = px + inpad
    cby = py + 11
    cbh = 9
    cgap = 4
    for i, lab in enumerate(["Client 1  local SGD", "Client 2  local SGD", "Client N  local SGD"]):
        box(pdf, cbx, cby, cbw, cbh, lab, None, BLUE_SOFT, INK, BLUE, title_size=7.6)
        cby += cbh + cgap
    varrow(pdf, px + pw / 2, cby - cgap + cbh - cbh, cby - 1)  # tiny connector handled below
    # aggregation box
    agg_y = cby + 1
    box(pdf, cbx, agg_y, cbw, 11, "Server aggregation", "+ optional DP noise",
        TEAL_SOFT, INK, TEAL, title_size=7.6, sub_size=6.6)
    varrow(pdf, px + pw / 2, cby - cgap, agg_y)
    glob_y = agg_y + 11 + 6
    box(pdf, cbx, glob_y, cbw, 9, "Global model", None, AMBER_SOFT, INK, AMBER, title_size=7.6)
    varrow(pdf, px + pw / 2, agg_y + 11, glob_y)

    # dashed arrow: global model -> Layer 1 (saved weights loaded)
    import math
    pdf.set_draw_color(*AMBER)
    pdf.set_line_width(0.4)
    gx = cbx
    gy = glob_y + 4.5
    # target: right edge of Layer-1 box on the spine (3rd spine box)
    l1_y = top + 2 * (bh + gap) + bh / 2
    l1_x = spine_x + spine_w
    # draw dashed segments
    steps = 18
    for s in range(steps):
        if s % 2 == 0:
            xa = gx + (l1_x - gx) * (s / steps)
            ya = gy + (l1_y - gy) * (s / steps)
            xb = gx + (l1_x - gx) * ((s + 1) / steps)
            yb = gy + (l1_y - gy) * ((s + 1) / steps)
            pdf.line(xa, ya, xb, yb)
    # arrowhead at l1
    ang = math.atan2(l1_y - gy, l1_x - gx)
    pdf.line(l1_x, l1_y, l1_x - 2.6 * math.cos(ang - 0.5), l1_y - 2.6 * math.sin(ang - 0.5))
    pdf.line(l1_x, l1_y, l1_x - 2.6 * math.cos(ang + 0.5), l1_y - 2.6 * math.sin(ang + 0.5))
    pdf.set_font("Helvetica", "I", 6.6)
    pdf.set_text_color(*AMBER)
    midx = (gx + l1_x) / 2
    midy = (gy + l1_y) / 2
    pdf.set_xy(midx - 11, midy + 1.5)
    pdf.cell(22, 3, "saved weights", align="C")

    pdf.set_y(max(y, py + ph) + 3)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build():
    pdf = DesignPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 16, 15)
    pdf.set_title("NSF system design (high-level and detailed)")
    pdf.add_page()

    # Title band
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, "System Design", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5.5,
                   "Domain-agnostic federated neuro-symbolic framework for reliable, explainable reasoning\n"
                   "over privacy-preserving heterogeneous data (Healthcare risk and Banking fraud)",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Part 1
    heading(pdf, "1. High-Level System Design")
    para(pdf,
         "The big picture. Each participant (client) keeps its own private data. A federated server trains one shared "
         "neural model by exchanging ONLY model weights - never the raw data. The trained global model feeds a "
         "neuro-symbolic inference stage that verifies every prediction against domain rules, a fact-locked explainer "
         "produces a safe explanation, and results are served through a dashboard with separate coordinator and client views.")
    draw_highlevel(pdf)
    para(pdf,
         "Privacy is structural: because only weights move upward, no client's records are ever centralised, yet the "
         "global model still benefits from every client's patterns.")

    # Part 2
    pdf.add_page()
    heading(pdf, "2. Detailed System Design")
    para(pdf,
         "The internals. The left spine shows how a SINGLE record flows through the five layers at inference time, "
         "including the decision logic that resolves conflicts between the neural and symbolic layers. The right-hand "
         "panel shows Layer 4 (federated training), which runs beforehand to produce the model that Layer 1 loads.")
    draw_detailed(pdf)
    para(pdf,
         "Reading the diagram: a record is standardised, classified by the federated MLP (Layer 1), converted into a "
         "typed fact (Layer 2) and scored by the symbolic rules (Layer 3). The decision block keeps the neural class "
         "when both agree, overrides to the symbolic class when a hard safety rule fires or neural confidence is below "
         "0.55, and otherwise keeps the neural class while logging the conflict. The verified explainer (Layer 5) then "
         "writes a fact-locked explanation, and the output bundles the decision, confidence, warnings and a full trace.")

    heading(pdf, "Key design decisions")
    bullets(pdf, [
        "Separation of concerns: neural model for accuracy, symbolic engine for verifiable logic, explainer for safe "
        "natural language - each is independently testable and swappable.",
        "Privacy by construction: federated averaging plus optional differential-privacy noise on client updates; raw "
        "data never leaves a client.",
        "Safety overrides: hard clinical/financial constraints can override an unsafe or low-confidence neural output.",
        "No hallucination: the explainer sees only verified facts and passes a guard; failures fall back to a strict "
        "template, giving a 100% hallucination-free ratio in our runs.",
        "Domain-agnostic: the same pipeline serves healthcare and banking by swapping only the inputs and the rule set.",
        "Role-based access: the dashboard enforces coordinator vs client views; a logged-in client sees only its own data.",
    ])

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
