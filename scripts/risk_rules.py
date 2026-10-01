"""
Invoice risk rules - the definition of "misbehaviour" that requires manual approval.

The manual-approval target combines
    (1) the historical approval policy recovered from the data: every invoice of $250K or more was
        manually approved and none below that amount was; and
    (2) anomaly flags on the invoice itself: freight, unit cost, lead time, payment terms, repeats.

    ManualApprovalRequired = 1  if ANY flag below fires

IMPORTANT: these labels are rule-derived ("weak supervision"), not confirmed fraud. The ML model learns
and operationalises this policy; its metrics show how well the policy can be reproduced and scored,
not independent fraud detection. See README - "Limitations".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg

FLAG_COLUMNS = {
    "Flag_HighValue": "High-value invoice (>= $250K)",
    "Flag_FreightAnomaly": "Freight above 1.5x the standard rate",
    "Flag_UnitCostOutlier": "Unit cost far from vendor norm",
    "Flag_LeadTimeOutlier": "Lead time unusual for vendor",
    "Flag_PaymentTermsOutlier": "Payment terms unusual for vendor",
    "Flag_PossibleDuplicate": "Possible duplicate invoice",
}
MAD_TO_SIGMA = 1.4826


# --------------------------------------------------------------------------- #
# Vendor reference statistics
# --------------------------------------------------------------------------- #
@dataclass
class VendorStats:
    """Per-vendor reference statistics plus global fallbacks."""

    table: pd.DataFrame          # index VendorNumber
    global_: dict[str, float]


def _mad(s: pd.Series) -> float:
    return float(np.median(np.abs(s - s.median())) * MAD_TO_SIGMA)


def fit_vendor_stats(df: pd.DataFrame) -> VendorStats:
    """Reference statistics per vendor, computed on ``df`` only (pass the training rows when modelling)."""
    d = df.assign(
        _luc=np.log(df["UnitCost"]),
        _ldol=np.log(df["Dollars"]),
    )
    g = d.groupby("VendorNumber")
    tbl = pd.DataFrame(
        {
            "n_po": g.size(),
            "med_log_uc": g["_luc"].median(),
            "med_log_dollars": g["_ldol"].median(),
            "med_lead": g["LeadTimeDays"].median(),
            "mad_lead": g["LeadTimeDays"].apply(_mad),
            "med_terms": g["PaymentTermsDays"].median(),
            "mad_terms": g["PaymentTermsDays"].apply(_mad),
        }
    )
    # typical freight rate from invoices that are not freight anomalies
    normal = d[d["FreightPctOfDollars"] <= cfg.FREIGHT_PCT_UPPER]
    tbl["freight_rate_pct"] = normal.groupby("VendorNumber")["FreightPctOfDollars"].median()

    glob = {
        "freight_rate_pct": float(normal["FreightPctOfDollars"].median()),
        "med_log_uc": float(d["_luc"].median()),
        "med_log_dollars": float(d["_ldol"].median()),
    }
    # vendor-relative statistics are only trusted with enough history
    thin = tbl["n_po"] < cfg.MIN_VENDOR_POS
    tbl.loc[thin, ["med_log_uc", "med_lead", "mad_lead", "med_terms", "mad_terms", "med_log_dollars"]] = np.nan
    tbl.loc[tbl["n_po"] < 5, "freight_rate_pct"] = np.nan
    return VendorStats(table=tbl, global_=glob)


def robust_z(x: pd.Series, med: pd.Series, mad: pd.Series) -> pd.Series:
    """Vendor-relative robust z-score; NaN when the vendor has no usable spread."""
    return (x - med) / mad.where(mad > 0)


def vendor_columns(df: pd.DataFrame, vs: VendorStats) -> pd.DataFrame:
    """Attach the vendor reference statistics to each invoice row."""
    return df[["VendorNumber"]].join(vs.table, on="VendorNumber").drop(columns="VendorNumber")


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
def repeat_invoice(df: pd.DataFrame) -> pd.Series:
    """True for an invoice repeating an earlier one (same vendor, dollars, quantity) within the window."""
    d = df[["VendorNumber", "Dollars", "Quantity", "InvoiceDate", "PONumber"]].sort_values(
        ["VendorNumber", "Dollars", "Quantity", "InvoiceDate", "PONumber"]
    )
    gap = d.groupby(["VendorNumber", "Dollars", "Quantity"])["InvoiceDate"].diff().dt.days
    rep = (gap <= cfg.DUP_WINDOW_DAYS) & (d["Dollars"] >= cfg.DUP_MIN_DOLLARS)
    return rep.reindex(df.index).fillna(False).astype(bool)


def add_risk_flags(df: pd.DataFrame, vendor_stats: VendorStats | None = None) -> pd.DataFrame:
    """Add the six rule flags, ``ManualApprovalRequired``, ``HistoricalManualApproval`` and ``RiskReasons``.

    ``df`` needs: VendorNumber, PONumber, InvoiceDate (datetime), Quantity, Dollars, Freight, UnitCost,
    FreightPctOfDollars, LeadTimeDays, PaymentTermsDays and (optionally) ApprovedBy.
    With ``vendor_stats=None`` the statistics come from ``df`` itself (full-data policy labelling).
    """
    vs = vendor_stats or fit_vendor_stats(df)
    out = df.copy()
    v = vendor_columns(out, vs)

    out["Flag_HighValue"] = out["Dollars"] >= cfg.HIGH_VALUE_THRESHOLD
    out["Flag_FreightAnomaly"] = out["FreightPctOfDollars"] > cfg.FREIGHT_PCT_UPPER

    uc_ratio = np.exp(np.log(out["UnitCost"]) - v["med_log_uc"])
    out["Flag_UnitCostOutlier"] = (
        (uc_ratio >= cfg.UNIT_COST_RATIO) | (uc_ratio <= 1 / cfg.UNIT_COST_RATIO)
    ).fillna(False)

    lead_z = robust_z(out["LeadTimeDays"], v["med_lead"], v["mad_lead"])
    terms_z = robust_z(out["PaymentTermsDays"], v["med_terms"], v["mad_terms"])
    out["Flag_LeadTimeOutlier"] = (lead_z.abs() > cfg.ROBUST_Z_LIMIT).fillna(False)
    out["Flag_PaymentTermsOutlier"] = (terms_z.abs() > cfg.ROBUST_Z_LIMIT).fillna(False)
    out["Flag_PossibleDuplicate"] = repeat_invoice(out)

    flags = list(FLAG_COLUMNS)
    out[flags] = out[flags].astype(bool)
    out["ManualApprovalRequired"] = out[flags].any(axis=1).astype(int)
    if "ApprovedBy" in out:
        out["HistoricalManualApproval"] = out["ApprovedBy"].notna().astype(int)
    out["RiskReasons"] = out[flags].apply(
        lambda r: "; ".join(FLAG_COLUMNS[c] for c in flags if r[c]), axis=1
    )
    return out


def policy_consistency(df: pd.DataFrame) -> dict:
    """Evidence that the $250K rule explains all historical manual approvals, and where controls are silent."""
    approved = df[df["HistoricalManualApproval"] == 1]
    anomalies = df[df[[c for c in FLAG_COLUMNS if c != "Flag_HighValue"]].any(axis=1) & ~df["Flag_HighValue"]]
    return {
        "historical_approvals": int(len(approved)),
        "approver_names": sorted(approved["ApprovedBy"].dropna().unique().tolist()),
        "min_approved_dollars": round(float(approved["Dollars"].min()), 2),
        "max_unapproved_dollars": round(float(df.loc[df["HistoricalManualApproval"] == 0, "Dollars"].max()), 2),
        "approved_below_threshold": int((approved["Dollars"] < cfg.HIGH_VALUE_THRESHOLD).sum()),
        "high_value_not_approved": int(
            ((df["Dollars"] >= cfg.HIGH_VALUE_THRESHOLD) & (df["HistoricalManualApproval"] == 0)).sum()
        ),
        "anomalous_invoices_below_threshold": int(len(anomalies)),
        "anomalous_below_threshold_never_approved": int((anomalies["HistoricalManualApproval"] == 0).sum()),
    }
