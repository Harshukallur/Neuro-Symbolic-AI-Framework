# Architecture

```
INPUT (heterogeneous tabular features per domain)
        │
        ▼
┌───────────────────────────┐
│ Layer 1 Neural Perception │  Federated MLP (FedAvg / FedProx / FedAdagrad)
└─────────────┬─────────────┘
              │ probs + features
              ▼
┌───────────────────────────┐
│ Layer 2 Neural→Symbolic   │  Typed SymbolicFact objects
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Layer 3 Symbolic Reasoning│  Domain rules + hard constraints + conflict flags
└─────────────┬─────────────┘
              │ verified facts ONLY
              ▼
┌───────────────────────────┐
│ Layer 5 Verified Explainer│  Reasoning traces, hallucination guard
└─────────────┬─────────────┘
              │
              ▼
 OUTPUT: prediction + explanation + confidence + warnings
```

Layer 4 (Federated Learning) trains Layer 1 across clients without sharing raw data.
