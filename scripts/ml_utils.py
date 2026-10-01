"""
Shared machine-learning helpers: data loading, feature engineering, regression wrappers and metrics.

Used by train_freight_model.py, train_approval_model.py, make_figures.py, the notebooks and the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_invoices(data_dir=cfg.DATA_DIR) -> pd.DataFrame:
    """PO-level purchases table (export_output_purchases.csv), sorted chronologically."""
    df = pd.read_csv(
        data_dir / cfg.OUT_PURCHASES, parse_dates=["PODate", "InvoiceDate", "PayDate"]
    )
    return df.sort_values(["InvoiceDate", "PONumber"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Freight model: features and estimators
# --------------------------------------------------------------------------- #
FREIGHT_FEATURES = [
    "Dollars", "Quantity", "UnitCost", "LeadTimeDays", "PaymentTermsDays",
    "Month", "DayOfWeek", "CatalogSKUs", "VendorRatePct",
]


def fit_freight_reference(train: pd.DataFrame) -> dict:
    """Typical freight rate per vendor, from training invoices without a freight anomaly."""
    clean = train[train["FreightPctOfDollars"] <= cfg.FREIGHT_PCT_UPPER]
    by_vendor = clean.groupby("VendorNumber")["FreightPctOfDollars"].agg(["median", "size"])
    by_vendor = by_vendor[by_vendor["size"] >= 5]["median"]
    return {
        "vendor_rate_pct": by_vendor.to_dict(),
        "global_rate_pct": float(clean["FreightPctOfDollars"].median()),
    }


def freight_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Features known when the invoice arrives. Freight itself is never used as an input."""
    x = pd.DataFrame(index=df.index)
    x["Dollars"] = df["Dollars"].astype(float)
    x["Quantity"] = df["Quantity"].astype(float)
    x["UnitCost"] = df["Dollars"] / df["Quantity"]
    x["LeadTimeDays"] = df["LeadTimeDays"].astype(float)
    x["PaymentTermsDays"] = df["PaymentTermsDays"].astype(float)
    x["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month.astype(float)
    x["DayOfWeek"] = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek.astype(float)
    x["CatalogSKUs"] = df["CatalogSKUs"].astype(float)
    x["VendorRatePct"] = (
        df["VendorNumber"].map(ref["vendor_rate_pct"]).fillna(ref["global_rate_pct"]).astype(float)
    )
    return x[FREIGHT_FEATURES]


class MedianRateBaseline(BaseEstimator, RegressorMixin):
    """Rule of thumb: freight = median historical freight rate x invoice dollars."""

    def fit(self, X, y):
        self.rate_ = float(np.median(y / X["Dollars"]))
        return self

    def predict(self, X):
        return self.rate_ * X["Dollars"].to_numpy()


class DollarRegressor(BaseEstimator, RegressorMixin):
    """Fit ``base`` on selected columns to predict freight dollars directly."""

    def __init__(self, base, cols):
        self.base, self.cols = base, cols

    def fit(self, X, y):
        self.model_ = clone(self.base).fit(X[self.cols], y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X[self.cols]))


class RateRegressor(BaseEstimator, RegressorMixin):
    """Predict the freight rate (freight / dollars) then scale back to dollars."""

    def __init__(self, base, cols):
        self.base, self.cols = base, cols

    def fit(self, X, y):
        self.model_ = clone(self.base).fit(X[self.cols], y / X["Dollars"])
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X[self.cols])) * X["Dollars"].to_numpy()


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(np.mean(ape) * 100),
        "MedianAPE_pct": float(np.median(ape) * 100),
        "Within10pct": float(np.mean(ape <= 0.10)),
    }


# --------------------------------------------------------------------------- #
# Manual-approval model: features
# --------------------------------------------------------------------------- #
APPROVAL_FEATURES = [
    "LogDollars", "LogQuantity", "LogUnitCost", "LogFreight", "FreightVsExpected",
    "LeadTimeDays", "PaymentTermsDays", "UnitCostRatioVendor", "LeadZ", "TermsZ",
    "DollarsRelVendor", "HasVendorHistory", "Month", "DayOfWeek", "CatalogSKUs",
]


def approval_features(df: pd.DataFrame, expected_freight: np.ndarray, vendor_stats) -> pd.DataFrame:
    """Invoice features for the manual-approval classifier.

    The classifier never sees the rule flags or the historical approval column; it only sees the
    invoice values and how they compare with the vendor's norms (computed on training data only).
    Vendors with too little history get neutral values and ``HasVendorHistory = 0``.
    """
    from risk_rules import robust_z, vendor_columns

    v = vendor_columns(df, vendor_stats)
    x = pd.DataFrame(index=df.index)
    x["LogDollars"] = np.log(df["Dollars"])
    x["LogQuantity"] = np.log(df["Quantity"])
    x["LogUnitCost"] = np.log(df["UnitCost"])
    x["LogFreight"] = np.log1p(df["Freight"])
    x["FreightVsExpected"] = df["Freight"] / np.maximum(np.asarray(expected_freight), 0.01)
    x["LeadTimeDays"] = df["LeadTimeDays"].astype(float)
    x["PaymentTermsDays"] = df["PaymentTermsDays"].astype(float)
    x["UnitCostRatioVendor"] = np.exp(np.log(df["UnitCost"]) - v["med_log_uc"]).fillna(1.0)
    x["LeadZ"] = robust_z(df["LeadTimeDays"], v["med_lead"], v["mad_lead"]).fillna(0.0)
    x["TermsZ"] = robust_z(df["PaymentTermsDays"], v["med_terms"], v["mad_terms"]).fillna(0.0)
    x["DollarsRelVendor"] = np.exp(np.log(df["Dollars"]) - v["med_log_dollars"]).fillna(1.0)
    x["HasVendorHistory"] = v["med_log_uc"].notna().astype(float)
    x["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month.astype(float)
    x["DayOfWeek"] = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek.astype(float)
    x["CatalogSKUs"] = df["CatalogSKUs"].astype(float)
    return x[APPROVAL_FEATURES]


# --------------------------------------------------------------------------- #
# Scoring new invoices (used by the dashboard "check an invoice" tool and tests)
# --------------------------------------------------------------------------- #
def load_artifacts() -> tuple[dict, dict]:
    import joblib

    return (joblib.load(cfg.MODELS_DIR / "freight_model.joblib"),
            joblib.load(cfg.MODELS_DIR / "approval_model.joblib"))


def score_invoices(raw: pd.DataFrame, freight_art: dict, approval_art: dict) -> pd.DataFrame:
    """Score invoices with the trained freight and manual-approval models.

    ``raw`` needs: VendorNumber, InvoiceDate, Quantity, Dollars, Freight, LeadTimeDays,
    PaymentTermsDays, CatalogSKUs. Returns the rule flags, expected freight, model risk probability
    and the model's manual-approval decision.
    """
    import risk_rules as rr

    df = raw.copy().reset_index(drop=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if "PONumber" not in df:
        df["PONumber"] = np.arange(len(df))
    df["UnitCost"] = df["Dollars"] / df["Quantity"]
    df["FreightPctOfDollars"] = df["Freight"] / df["Dollars"] * 100
    expected = freight_art["model"].predict(freight_features(df, freight_art["reference"]))
    vs = approval_art["vendor_stats"]
    flagged = rr.add_risk_flags(df, vs)
    X = approval_features(df, expected, vs)
    proba = approval_art["model"].predict_proba(X)[:, 1]
    flagged["ExpectedFreight"] = expected
    flagged["FreightVsExpected"] = df["Freight"] / np.maximum(expected, 0.01)
    flagged["RiskProbability"] = proba
    flagged["ModelFlagged"] = (proba >= approval_art["threshold"]).astype(int)
    return flagged
