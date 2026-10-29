# Domain-Agnostic Federated Neuro-Symbolic Framework

Reliable, explainable LLM reasoning over privacy-preserving heterogeneous data.

Based on the project synopsis: *A Domain-Agnostic Federated Neuro-Symbolic Framework for Reliable and Explainable LLM Reasoning over Privacy-Preserving Heterogeneous Data* (RIT M.Tech CSE AI&ML).

## What this implements

| Layer | Role |
|-------|------|
| **1. Neural Perception** | MLP classifiers for tabular domain signals |
| **2. Neural → Symbolic Bridge** | Converts predictions into typed facts |
| **3. Symbolic Reasoning** | Rule/logic verification (anti-hallucination) |
| **4. Federated Learning** | FedAvg, FedProx, FedAdagrad — raw data stays local |
| **5. Verified LLM Explanation** | Explanations from **verified facts only** |

## Domains (custom datasets)

1. **Healthcare** — disease risk classification (`low_risk` / `medium_risk` / `high_risk`)
2. **Banking** — fraud detection (`legitimate` / `fraudulent`)

Both datasets are **generated** with causal feature → label rules aligned to the symbolic rule base (non-IID Dirichlet client splits).

## Anti-hallucination design

- Symbolic rules verify every neural prediction.
- Conflicts with strong symbolic evidence → **override** neural output.
- The explanation layer receives **only** `verified_facts_only()`.
- A hallucination guard rejects explanations that invent rule names or omit entity grounding.

## Quick start

```powershell
# From project root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# Generate datasets
python scripts/generate_datasets.py

# Train all strategies on both domains + evaluate
python scripts/train_and_evaluate.py

# Demo verified explanations
python scripts/demo_inference.py

# Plots
python scripts/plot_results.py

# Dashboard at http://127.0.0.1:8765
python scripts/serve_dashboard.py
```

The dashboard reads the saved models and result files. It does not retrain. **Results** compares the three federated strategies, **Cases** opens a verified trace, and **Predict** runs a new record through the same neural model, symbolic checker, and explanation guard.

## Project layout

```
configs/default.yaml          # hyperparameters
nsf/
  data/generators.py          # custom Healthcare + Banking datasets
  neural/models.py            # perception MLP
  bridge/neural_symbolic.py   # neural → symbolic facts
  symbolic/rules.py           # domain logical constraints
  symbolic/engine.py          # verification + override policy
  federated/                  # FedAvg / FedProx / FedAdagrad
  llm/explainer.py            # verified explanations + traces
  pipeline.py                 # end-to-end orchestration
scripts/                      # CLI entry points, including the dashboard server
web/                          # dashboard page
data/                         # generated client/test CSVs + facts
models/                       # saved global models
results/                      # metrics, traces, plots
```

## Outputs

- `results/summary.json` — accuracy, communication cost, hallucination-free ratio
- `results/{domain}_{strategy}_results.json` — detailed metrics
- `results/{domain}_{strategy}_traces.json` — reasoning traces
- `results/plots/` — accuracy curves and neural vs neuro-symbolic bars

## Notes

- Default device is `cpu` (set `device: cuda` in config if a GPU is available).
- No raw client data is shared during federated rounds — only model parameters.
- Explanations are constrained/template-grounded so the system remains reliable without an external LLM API key. You can later plug in an API model **only** as a rephraser over the same verified fact payload.
