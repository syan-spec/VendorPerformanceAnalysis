"""Sanity tests for the pipeline outputs, risk rules and models.  Run:  pytest -q   (after run_all.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import config as cfg  # noqa: E402
import ml_utils  # noqa: E402
import risk_rules as rr  # noqa: E402


@pytest.fixture(scope="module")
def purchases():
    return pd.read_csv(cfg.DATA_DIR / cfg.OUT_PURCHASES, parse_dates=["PODate", "InvoiceDate", "PayDate"])


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(cfg.DATA_DIR / cfg.OUT_SUMMARY)


@pytest.fixture(scope="module")
def risk():
    return pd.read_csv(cfg.DATA_DIR / cfg.OUT_INVOICE_RISK)


def test_purchases_match_raw_invoice():
    raw = pd.read_csv(cfg.DATA_DIR / cfg.RAW_FILES["vendor_invoice"])
    out = pd.read_csv(cfg.DATA_DIR / cfg.OUT_PURCHASES)
    assert len(out) == len(raw) and out["PONumber"].is_unique
    assert np.isclose(out["Dollars"].sum(), raw["Dollars"].sum())
    assert out["Quantity"].sum() == raw["Quantity"].sum()


def test_purchases_derived_columns(purchases):
    assert (purchases["LeadTimeDays"] == (purchases["InvoiceDate"] - purchases["PODate"]).dt.days).all()
    assert (purchases["PaymentTermsDays"] == (purchases["PayDate"] - purchases["InvoiceDate"]).dt.days).all()
    assert np.allclose(purchases["UnitCost"], purchases["Dollars"] / purchases["Quantity"], atol=1e-3)
    assert np.allclose(purchases["LandedCost"], purchases["Dollars"] + purchases["Freight"], atol=0.01)


def test_summary_reconciles(summary, purchases):
    in_p = purchases[purchases["InPeriod"]]
    assert np.isclose(summary["PurchaseDollars"].sum(), in_p["Dollars"].sum())
    assert summary["VendorNumber"].is_unique
    assert np.isclose(summary["PurchaseSharePct"].sum(), 100)
    assert summary["VendorScore"].dropna().between(0, 100).all()


def test_estimates_hidden_for_unreliable_vendors(summary):
    bad = summary[~summary["EstimateReliable"]]
    assert bad[["EstUnitsSold", "StockTurnover", "EstRevenue"]].isna().all().all()


def test_approval_policy_is_a_dollar_rule(purchases):
    approved = purchases["ApprovedBy"].notna()
    assert purchases.loc[approved, "Dollars"].min() >= cfg.HIGH_VALUE_THRESHOLD
    assert purchases.loc[~approved, "Dollars"].max() < cfg.HIGH_VALUE_THRESHOLD


def test_risk_flags_basic():
    df = pd.DataFrame({
        "VendorNumber": [1, 1, 1], "PONumber": [1, 2, 3],
        "InvoiceDate": pd.to_datetime(["2024-03-01"] * 3), "Quantity": [100, 100, 100],
        "Dollars": [300_000.0, 10_000.0, 10_000.0], "Freight": [1_500.0, 50.0, 500.0],
        "LeadTimeDays": [16] * 3, "PaymentTermsDays": [35] * 3, "ApprovedBy": [None] * 3})
    df["UnitCost"] = df["Dollars"] / df["Quantity"]
    df["FreightPctOfDollars"] = df["Freight"] / df["Dollars"] * 100
    out = rr.add_risk_flags(df)
    assert out["Flag_HighValue"].tolist() == [True, False, False]
    assert out["Flag_FreightAnomaly"].tolist() == [False, False, True]
    assert out["ManualApprovalRequired"].tolist() == [1, 0, 1]


def test_risk_export_consistency(risk):
    flags = [c for c in rr.FLAG_COLUMNS]
    assert (risk[flags].any(axis=1).astype(int) == risk["ManualApprovalRequired"]).all()
    assert risk["RiskProbability"].between(0, 1).all()
    assert (risk["ControlGap"] == ((risk["ManualApprovalRequired"] == 1) & (risk["HistoricalManualApproval"] == 0))).all()


def test_models_score_sensibly():
    freight_art, approval_art = ml_utils.load_artifacts()
    rows = pd.DataFrame([
        dict(VendorNumber=3960, InvoiceDate="2024-11-04", Quantity=40_000, Dollars=400_000, Freight=2_000,
             LeadTimeDays=16, PaymentTermsDays=35, CatalogSKUs=480),
        dict(VendorNumber=3960, InvoiceDate="2024-11-04", Quantity=1_000, Dollars=10_000, Freight=50,
             LeadTimeDays=16, PaymentTermsDays=35, CatalogSKUs=480),
        dict(VendorNumber=3960, InvoiceDate="2024-11-04", Quantity=1_000, Dollars=10_000, Freight=600,
             LeadTimeDays=16, PaymentTermsDays=35, CatalogSKUs=480)])
    s = ml_utils.score_invoices(rows, freight_art, approval_art)
    assert s["ModelFlagged"].tolist() == [1, 0, 1]
    assert abs(s.loc[1, "ExpectedFreight"] - 50) < 5            # about 0.5% of $10,000


def test_unseen_vendor_does_not_crash():
    freight_art, approval_art = ml_utils.load_artifacts()
    rows = pd.DataFrame([dict(VendorNumber=-5, InvoiceDate="2024-11-04", Quantity=10, Dollars=100, Freight=0.5,
                              LeadTimeDays=16, PaymentTermsDays=35, CatalogSKUs=0)])
    s = ml_utils.score_invoices(rows, freight_art, approval_art)
    assert np.isfinite(s["RiskProbability"]).all()
