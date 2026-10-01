"""
Shared analysis functions (research questions, hypothesis test, key findings).

    python scripts/analysis.py        # writes data/key_findings.json

Used by the notebooks, the dashboard and the PDF report so every number comes from one place.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    purchases = pd.read_csv(cfg.DATA_DIR / cfg.OUT_PURCHASES, parse_dates=["PODate", "InvoiceDate", "PayDate"])
    summary = pd.read_csv(cfg.DATA_DIR / cfg.OUT_SUMMARY)
    sku = pd.read_csv(cfg.DATA_DIR / cfg.OUT_SKU)
    return purchases, summary, sku


def margin_test(summary: pd.DataFrame) -> dict | None:
    """Welch t-test on vendor gross margin: top-quartile vs bottom-quartile spend vendors."""
    v = summary[summary["ScoreEligible"]]
    hi = v[v["PurchaseDollars"] >= v["PurchaseDollars"].quantile(0.75)]["GrossMarginPct"]
    lo = v[v["PurchaseDollars"] <= v["PurchaseDollars"].quantile(0.25)]["GrossMarginPct"]
    if len(hi) < 3 or len(lo) < 3:
        return None
    res = stats.ttest_ind(hi, lo, equal_var=False)
    ci = res.confidence_interval(0.95)
    return {
        "n_high_spend": int(len(hi)), "n_low_spend": int(len(lo)),
        "mean_margin_high_spend": float(hi.mean()), "mean_margin_low_spend": float(lo.mean()),
        "difference_pp": float(hi.mean() - lo.mean()), "t_statistic": float(res.statistic),
        "p_value": float(res.pvalue), "ci95_low": float(ci.low), "ci95_high": float(ci.high),
        "significant_at_5pct": bool(res.pvalue < 0.05),
    }


def bulk_order_effect(purchases: pd.DataFrame) -> pd.DataFrame:
    """Within each vendor (>= 8 POs), unit cost of each order relative to the vendor average,
    grouped by order-size quartile. An index below 1.00 for large orders would indicate volume discounts."""
    p = purchases[purchases["InPeriod"]].copy()
    p = p[p.groupby("VendorNumber")["PONumber"].transform("count") >= 8].copy()
    vc = p.groupby("VendorNumber")[["Dollars", "Quantity"]].transform("sum")
    p["CostIndex"] = p["UnitCost"] / (vc["Dollars"] / vc["Quantity"])
    labels = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
    p["OrderSizeQuartile"] = p.groupby("VendorNumber")["Quantity"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=labels)
    )
    return (p.groupby("OrderSizeQuartile", observed=True)["CostIndex"]
            .agg(["mean", "median", "count"]).reset_index())


def promotion_candidates(sku: pd.DataFrame) -> pd.DataFrame:
    """SKUs with high margin whose inventory grew and whose ending stock value is high.

    Proxy for 'high margin but slow selling': brand-level sales are not available, so a growing,
    large stock of a high-margin SKU is the best available signal for promotion or repricing.
    """
    s = sku[sku["ValidPricing"] & (sku["EndUnits"] > 0)].copy()
    return s[
        (s["MarginPct"] >= s["MarginPct"].quantile(0.75))
        & (s["EndUnits"] > s["BeginUnits"])
        & (s["EndValueAtCost"] >= s["EndValueAtCost"].quantile(0.75))
    ].sort_values("EndValueAtCost", ascending=False)


def key_findings() -> dict:
    purchases, summary, sku = load_tables()
    quality = json.loads((cfg.DATA_DIR / cfg.OUT_QUALITY).read_text())
    in_p = purchases[purchases["InPeriod"]]
    active = summary[summary["PurchaseOrders"] > 0]
    spend = active["PurchaseDollars"].sum()
    rel = summary[summary["EstimateReliable"]]
    hhi = float(((active["PurchaseDollars"] / spend) ** 2).sum() * 10_000)
    top10 = active.nlargest(10, "PurchaseDollars")
    half = active.nsmallest(len(active) // 2, "PurchaseDollars")
    under = summary[summary["Segment"] == "Underperformer - review or exit"]
    promo = promotion_candidates(sku)
    bulk = bulk_order_effect(purchases)
    scored = summary[summary["ScoreEligible"]].sort_values("VendorScore", ascending=False)
    small = in_p[in_p["Dollars"] < 1_000]
    large = in_p[in_p["Dollars"] >= 50_000]

    out = {
        "period": [quality["period_start"], quality["period_end"]],
        "purchase_orders_in_period": int(len(in_p)),
        "active_vendors": int(len(active)),
        "total_purchases": float(spend),
        "total_freight": float(active["FreightDollars"].sum()),
        "units_purchased": int(active["UnitsPurchased"].sum()),
        "top10_share_pct": float(top10["PurchaseDollars"].sum() / spend * 100),
        "top10_vendors": top10["VendorName"].tolist(),
        "hhi": hhi,
        "bottom_half_vendors": int(len(half)),
        "bottom_half_share_pct": float(half["PurchaseDollars"].sum() / spend * 100),
        "est_gross_margin_pct": float(rel["EstGrossProfit"].sum() / rel["EstRevenue"].sum() * 100),
        "est_revenue": float(rel["EstRevenue"].sum()),
        "est_gross_profit": float(rel["EstGrossProfit"].sum()),
        "segment_counts": summary["Segment"].value_counts().to_dict(),
        "segment_spend_pct": (summary.groupby("Segment")["PurchaseDollars"].sum() / spend * 100).round(2).to_dict(),
        "underperformers": {
            "vendors": int(len(under)),
            "spend_pct": float(under["PurchaseDollars"].sum() / spend * 100),
            "ending_inventory_at_cost": float(under["EndValueAtCost"].sum()),
            "largest_by_inventory": under.nlargest(3, "EndValueAtCost")["VendorName"].tolist(),
        },
        "top_scored_vendors": scored.head(5)["VendorName"].tolist(),
        "bottom_scored_vendors": scored.tail(5)["VendorName"].tolist(),
        "promotion_candidates": {
            "skus": int(len(promo)),
            "ending_inventory_at_cost": float(promo["EndValueAtCost"].sum()),
            "avg_margin_pct": float(promo["MarginPct"].mean()) if len(promo) else None,
            "rule": "margin >= P75, inventory grew over the year, ending stock value >= P75",
        },
        "inventory": {
            "begin_value_at_cost": float(sku["BeginValueAtCost"].sum()),
            "end_value_at_cost": float(sku["EndValueAtCost"].sum()),
            "growth_pct": float((sku["EndValueAtCost"].sum() / sku["BeginValueAtCost"].sum() - 1) * 100),
        },
        "margin_hypothesis_test": margin_test(summary),
        "bulk_order_median_cost_index": {r["OrderSizeQuartile"]: float(r["median"]) for _, r in bulk.iterrows()},
        "bulk_order_mean_cost_index": {r["OrderSizeQuartile"]: float(r["mean"]) for _, r in bulk.iterrows()},
        "freight_pct_small_orders_under_1k": float(small["Freight"].sum() / small["Dollars"].sum() * 100),
        "freight_pct_large_orders_50k_plus": float(large["Freight"].sum() / large["Dollars"].sum() * 100),
        "negative_margin_vendors": summary[summary["GrossMarginPct"] < 0]["VendorName"].tolist(),
    }
    fm, am = cfg.MODELS_DIR / cfg.OUT_FREIGHT_METRICS, cfg.MODELS_DIR / cfg.OUT_APPROVAL_METRICS
    if fm.exists():
        out["freight_model"] = json.loads(fm.read_text())
    if am.exists():
        out["approval_model"] = json.loads(am.read_text())
    return out


if __name__ == "__main__":
    kf = key_findings()
    (cfg.DATA_DIR / "key_findings.json").write_text(json.dumps(kf, indent=2, default=float))
    print(json.dumps({k: v for k, v in kf.items() if k not in ("freight_model", "approval_model")}, indent=1, default=float)[:3500])
