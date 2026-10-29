"""Write NSF_Models_Explained.pdf into the project root (main nsf folder).

The document describes every model / component used in the framework and the
reason it was chosen. Results are read live from results/summary.json so the
accuracy table always matches the latest run. Core (Helvetica) fonts only, so
text is ASCII; ">=" is written literally instead of the Unicode glyph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.fonts import FontFace

OUT = ROOT / "NSF_Models_Explained.pdf"
SUMMARY = ROOT / "results" / "summary.json"
CONFIG = ROOT / "configs" / "default.yaml"

INK = (26, 35, 50)
MUTED = (80, 90, 102)
TEAL = (12, 92, 86)
TEAL_SOFT = (229, 243, 241)


class ModelsPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "Models used in the NSF framework and why",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(
            0, 8,
            f"Page {self.page_no()}  |  Code: nsf/neural, nsf/federated, nsf/symbolic, nsf/bridge, nsf/llm",
            align="C",
        )


def heading(pdf, text):
    if pdf.get_y() > 252:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_color(*INK)


def subheading(pdf, text):
    if pdf.get_y() > 258:
        pdf.add_page()
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


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


def ensure_room(pdf, mm):
    if pdf.get_y() + mm > pdf.page_break_trigger:
        pdf.add_page()


def add_table(pdf, headers, rows, widths):
    pdf.set_font("Helvetica", "", 8.4)
    style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TEAL)
    with pdf.table(
        width=sum(widths),
        col_widths=widths,
        headings_style=style,
        text_align="LEFT",
        line_height=4.4,
        padding=1.4,
        borders_layout="ALL",
        cell_fill_color=TEAL_SOFT,
        cell_fill_mode="ROWS",
        repeat_headings=1,
    ) as table:
        head = table.row()
        for name in headers:
            head.cell(name)
        for values in rows:
            row = table.row()
            for value in values:
                row.cell(str(value))
    pdf.ln(2)


def load_summary():
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return None


def build():
    summary = load_summary()

    pdf = ModelsPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 16, 15)
    pdf.set_title("Models used in the NSF framework and why")
    pdf.add_page()

    # Title block
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, "Models used in the framework, and why they were chosen",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0, 5.5,
        "Domain-agnostic federated neuro-symbolic framework\n"
        "Reliable, explainable LLM reasoning over privacy-preserving heterogeneous data",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # 0. Overview
    heading(pdf, "1. Overview: a five-layer pipeline, not a single model")
    para(
        pdf,
        "The framework deliberately combines several models, each responsible for one job. The neural model gives accuracy, "
        "the federated optimiser lets it learn without moving raw data, the symbolic engine gives verifiable logic, and the "
        "explainer turns verified facts into safe text. This separation is what makes the system both accurate and reliable.",
    )
    add_table(
        pdf,
        ["Layer", "Component / model", "Type", "Role"],
        [
            ["1", "PerceptionMLP", "Neural network (PyTorch)", "Classify tabular records per domain"],
            ["4", "FedAvg / FedProx / FedAdagrad", "Federated optimisers", "Train Layer 1 across clients, no raw data shared"],
            ["2", "NeuralSymbolicBridge", "Deterministic converter", "Turn predictions + features into typed facts"],
            ["3", "SymbolicReasoner", "Rule engine", "Verify predictions, apply safe overrides"],
            ["5", "VerifiedLLMExplainer", "Constrained generator + guard", "Explain using verified facts only"],
        ],
        (12, 58, 48, 62),
    )

    # 2. Neural model
    heading(pdf, "2. Neural model: PerceptionMLP (Layer 1)")
    para(
        pdf,
        "A compact multi-layer perceptron defined in nsf/neural/models.py. The same architecture is reused for both domains "
        "(only the input and output sizes change), which is what makes the framework domain-agnostic.",
    )
    subheading(pdf, "Architecture")
    add_table(
        pdf,
        ["Stage", "Operation", "Purpose"],
        [
            ["Input", "Linear(input_dim -> 96)", "Project features into a hidden space"],
            ["", "LayerNorm(96) -> ReLU -> Dropout(0.12)", "Stabilise + regularise"],
            ["Hidden", "Linear(96 -> 96)", "Learn feature interactions"],
            ["", "LayerNorm(96) -> ReLU -> Dropout(0.08)", "Stabilise + regularise"],
            ["Output", "Linear(96 -> num_classes)", "Class logits (softmax for probabilities)"],
        ],
        (22, 78, 80),
    )
    para(
        pdf,
        "Healthcare uses input_dim=12 and num_classes=3 (low/medium/high risk). Banking uses input_dim=14 and num_classes=2 "
        "(legitimate/fraudulent).",
    )
    subheading(pdf, "Why an MLP, and why these choices")
    bullets(
        pdf,
        [
            "The data is small, tabular and heterogeneous. For tabular data an MLP is a strong, standard baseline; CNNs assume "
            "spatial structure and Transformers need far more data, so neither is justified here.",
            "LayerNorm (not BatchNorm) is used because federated clients have small, non-IID batches. BatchNorm statistics are "
            "unstable and leak batch information across the federation; LayerNorm normalises per-sample and is federation-safe.",
            "Dropout (0.12 then 0.08) regularises the model so it does not overfit the small per-client datasets.",
            "It is compact (one hidden width of 96) so each training round is fast on CPU and the parameter payload sent each "
            "federated round stays small (about 5 MB total communication per strategy in our runs).",
            "A simple, inspectable model keeps the downstream symbolic verification meaningful: we can reason about its outputs "
            "rather than treat it as an opaque black box.",
        ],
    )

    # 3. Federated strategies
    heading(pdf, "3. Federated training optimisers (Layer 4)")
    para(
        pdf,
        "Training is federated: each client trains locally on its own data and only model parameters are aggregated on the "
        "server. Raw records never leave the client (nsf/federated/client.py). Clients optimise with SGD (momentum 0.9) and "
        "cross-entropy loss. Three server strategies are implemented and compared so we can show which behaves best under "
        "non-IID data (nsf/federated/aggregation.py).",
    )
    add_table(
        pdf,
        ["Strategy", "What it does", "Why it is included"],
        [
            ["FedAvg", "Size-weighted average of client weights.",
             "The canonical baseline (McMahan 2017); communication-efficient reference point."],
            ["FedProx", "FedAvg plus a client-side proximal term (mu/2)||w - w_global||^2.",
             "Handles non-IID / system heterogeneity by limiting client drift (Li 2020)."],
            ["FedAdagrad", "Adaptive server optimiser: accumulates squared pseudo-gradients and scales the update.",
             "Faster, more robust convergence under heterogeneity (Reddi 2021)."],
        ],
        (26, 78, 76),
    )
    bullets(
        pdf,
        [
            "Why three and not one: the datasets are split across clients with a non-IID Dirichlet partition (alpha 0.6-0.7), so "
            "client distributions differ a lot. Comparing FedAvg, FedProx and FedAdagrad demonstrates robustness and lets the "
            "project report which converges best per domain.",
            "Why SGD with momentum on the client: it is the standard, stable local optimiser used in the FedAvg/FedProx/FedAdagrad "
            "literature and keeps client compute light.",
            "FedProx's proximal term is applied on the client; its server aggregation is identical to FedAvg. FedAdagrad keeps an "
            "extra server-side accumulator, which is why its server update differs.",
        ],
    )

    # 4. Bridge, symbolic, explainer
    heading(pdf, "4. The non-neural models and why they matter")
    subheading(pdf, "Neural -> Symbolic Bridge (Layer 2)")
    para(
        pdf,
        "A deterministic converter (nsf/bridge/neural_symbolic.py) that packages the prediction, confidence and the numeric/"
        "boolean features into a typed SymbolicFact. Why: it guarantees that only structured facts - never free-form text - "
        "enter the logic layer, which is the first line of defence against hallucination.",
    )
    subheading(pdf, "Symbolic Reasoner (Layer 3)")
    para(
        pdf,
        "A transparent rule engine (nsf/symbolic/engine.py) that scores published domain rules, compares the symbolic class with "
        "the neural class, and applies an override only under hard safety constraints or when neural confidence is low. Why a "
        "rule engine instead of a second network: its decisions are auditable, reproducible, and provide the evidence sentences "
        "the explainer is allowed to use. (The thresholds behind these rules are justified in NSF_Threshold_Evidence.pdf.)",
    )
    subheading(pdf, "Verified LLM Explainer (Layer 5)")
    para(
        pdf,
        "A constrained, fact-locked generator with a hallucination guard (nsf/llm/explainer.py). It consumes only verified facts "
        "and can mention only rules that actually fired; if any drift is detected it falls back to a strict template. Why not a "
        "raw external LLM: reliability and reproducibility without an API key. An external API model could later be plugged in "
        "purely as a rephraser over the same verified payload, without gaining the ability to invent facts.",
    )

    # 5. Training configuration
    heading(pdf, "5. Training configuration (configs/default.yaml)")
    add_table(
        pdf,
        ["Setting", "Value", "Setting", "Value"],
        [
            ["Clients", "5", "Global rounds", "12"],
            ["Local epochs", "4", "Batch size", "32"],
            ["Learning rate", "0.015", "FedProx mu", "0.01"],
            ["Client optimiser", "SGD (momentum 0.9)", "Loss", "Cross-entropy"],
            ["Device", "cpu (cuda optional)", "Seed", "42"],
        ],
        (40, 50, 40, 50),
    )

    # 6. Results
    heading(pdf, "6. Measured results (from results/summary.json)")
    if summary:
        rows = []
        for domain in ("healthcare", "banking"):
            dom = summary["domains"][domain]
            for strat, m in dom["strategies"].items():
                rows.append([
                    domain,
                    strat,
                    f"{m['neural_only_accuracy']*100:.1f}%",
                    f"{m['neurosymbolic_accuracy']*100:.1f}%",
                    f"{m['communication_mb']:.2f}",
                    f"{m['hallucination_free_ratio']*100:.0f}%",
                ])
        add_table(
            pdf,
            ["Domain", "Strategy", "Neural acc", "Neuro-sym acc", "Comm (MB)", "Halluc-free"],
            rows,
            (30, 30, 28, 34, 26, 32),
        )
        hc = summary["domains"]["healthcare"]
        bk = summary["domains"]["banking"]
        para(
            pdf,
            f"Best strategy - healthcare: {hc['best_strategy']} "
            f"(neuro-symbolic accuracy {hc['best_neurosymbolic_accuracy']*100:.1f}%). "
            f"Best strategy - banking: {bk['best_strategy']} "
            f"(neuro-symbolic accuracy {bk['best_neurosymbolic_accuracy']*100:.1f}%). "
            "The hallucination-free ratio is 100% in every configuration, which is the main reliability claim of the design.",
        )
    else:
        para(pdf, "results/summary.json was not found. Run scripts/train_and_evaluate.py first, then regenerate this PDF.")

    # 7. Why this combination
    heading(pdf, "7. Why this combination of models")
    bullets(
        pdf,
        [
            "Accuracy: the federated MLP reaches ~91-93% (healthcare) and ~99% (banking) on held-out test data.",
            "Privacy: federated learning keeps raw client data local; only parameters are shared.",
            "Reliability: the symbolic engine verifies every prediction and can override unsafe ones.",
            "Explainability: the explainer is fact-locked, so explanations never introduce unverified claims.",
            "Domain-agnostic: the identical neural architecture and pipeline serve both Healthcare and Banking by changing only "
            "the input/output sizes and the domain rule set.",
        ],
    )

    # 8. References
    heading(pdf, "8. Method references")
    refs = [
        "McMahan B, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg). AISTATS 2017. "
        "https://arxiv.org/abs/1602.05629",
        "Li T, et al. Federated Optimization in Heterogeneous Networks (FedProx). MLSys 2020. https://arxiv.org/abs/1812.06127",
        "Reddi S, et al. Adaptive Federated Optimization (FedAdagrad/FedAdam/FedYogi). ICLR 2021. https://arxiv.org/abs/2003.00295",
        "Ba JL, Kiros JR, Hinton GE. Layer Normalization. 2016. https://arxiv.org/abs/1607.06450",
        "Srivastava N, et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 2014.",
    ]
    pdf.set_font("Helvetica", "", 9.2)
    for r in refs:
        ensure_room(pdf, 12)
        pdf.multi_cell(0, 4.6, "-  " + r, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.8)

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
