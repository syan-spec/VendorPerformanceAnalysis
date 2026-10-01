"""
Run the full pipeline end to end:

    python scripts/run_all.py

ingestion -> vendor summary -> freight model -> approval model -> key findings -> figures
"""
from __future__ import annotations

import json
import time

import analysis
import config as cfg
import get_vendor_summary
import ingestion_db
import make_figures
import train_approval_model
import train_freight_model

log = cfg.get_logger("run_all")


def main() -> None:
    t0 = time.time()
    steps = [
        ("1/6 ingest raw CSVs into SQLite", ingestion_db.main),
        ("2/6 build purchases, SKU and vendor tables", lambda: get_vendor_summary.run(cfg.DATA_DIR, cfg.DB_PATH)),
        ("3/6 train freight cost model", train_freight_model.run),
        ("4/6 train manual-approval model", train_approval_model.run),
        ("5/6 compute key findings", lambda: (cfg.DATA_DIR / "key_findings.json").write_text(
            json.dumps(analysis.key_findings(), indent=2, default=float))),
        ("6/6 create report figures", make_figures.main),
    ]
    for name, fn in steps:
        log.info("=== %s", name)
        fn()
    log.info("Pipeline finished in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
