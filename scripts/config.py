"""Central configuration: paths, file names, thresholds and random seeds."""
from __future__ import annotations

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_NAME = "inventory.db"
DB_PATH = DATA_DIR / DB_NAME          # SQLite database (git-ignored)
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
LOGS_DIR = ROOT / "logs"

# Raw inputs (table name -> csv file in data/)
RAW_FILES = {
    "vendor_invoice": "export_output_vendor_invoice.csv",
    "purchase_prices": "export_output_purchase_prices.csv",
    "begin_inventory": "export_output_begin_inventory.csv",
    "end_inventory": "export_output_end_inventory.csv",
}

# Generated outputs
OUT_PURCHASES = "export_output_purchases.csv"
OUT_SUMMARY = "export_output_vendor_summary.csv"
OUT_SKU = "export_output_sku_margins.csv"
OUT_QUALITY = "data_quality_report.json"
OUT_INVOICE_RISK = "export_output_invoice_risk.csv"
OUT_FREIGHT_METRICS = "freight_model_metrics.json"
OUT_APPROVAL_METRICS = "approval_model_metrics.json"

# Vendor KPI reliability thresholds
MIN_UNITS_AVAILABLE = 100      # begin inventory + purchased units
MIN_PO_COUNT = 3               # purchase orders in the period

# Composite vendor score: metric -> (weight, higher_is_better); weights sum to 1.0
SCORE_WEIGHTS: dict[str, tuple[float, bool]] = {
    "GrossMarginPct": (0.35, True),
    "SellThroughRate": (0.20, True),
    "StockTurnover": (0.20, True),
    "AvgLeadTimeDays": (0.07, False),
    "LeadTimeCV": (0.06, False),
    "FreightPctOfPurchases": (0.07, False),
    "AvgPaymentTermsDays": (0.05, True),
}

# Invoice risk rules (see scripts/risk_rules.py)
HIGH_VALUE_THRESHOLD = 250_000      # observed policy: every invoice >= ~$250.7K was manually approved
FREIGHT_PCT_UPPER = 0.75            # freight % of invoice $ above this = 1.5x the standard 0.5% rate
UNIT_COST_RATIO = 2.0               # unit cost >= 2x or <= 0.5x the vendor median
ROBUST_Z_LIMIT = 4.0                # |robust z| above this = lead-time / payment-terms outlier
MIN_VENDOR_POS = 8                  # invoices needed before vendor-specific norms are trusted
DUP_WINDOW_DAYS = 14                # same vendor + dollars + quantity within this window = repeat
DUP_MIN_DOLLARS = 1_000             # ignore repeats on tiny invoices

# Machine-learning settings
RANDOM_STATE = 42
TEST_SIZE = 0.25
TEMPORAL_CUTOFF = "2024-10-01"      # freight model: train before, test on/after


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Console + optional file logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for command-line scripts."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
