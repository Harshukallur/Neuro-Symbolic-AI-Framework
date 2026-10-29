"""Generate custom Healthcare and Banking datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsf.data.generators import generate_all_datasets
from nsf.utils import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    report = generate_all_datasets(cfg, ROOT / cfg["paths"]["data_dir"])
    print("Datasets generated:")
    for domain, meta in report.items():
        print(
            f"  - {domain}: train={meta['train_size']} test={meta['test_size']} "
            f"clients={meta['num_clients']} alpha={meta['non_iid_alpha']}"
        )


if __name__ == "__main__":
    main()
