"""
Step 2 - Build the analytical tables from the SQLite database.

    python scripts/get_vendor_summary.py

SQL (CTEs, joins, window functions) does the extraction and aggregation; pandas adds the derived KPIs,
the composite vendor score and the segments. Outputs (CSV files in ``data/`` and tables in the DB):

    export_output_purchases.csv        PO-level purchases fact table (cleaned + enriched)
    export_output_vendor_summary.csv   one row per vendor: KPIs, composite score, segment
    export_output_sku_margins.csv      one row per SKU (brand): catalog margin + inventory
    data_quality_report.json           reconciliation checks and assumptions

DATA LIMITATIONS
    No sales file and no PO line items are available, so units sold are ESTIMATED per vendor with
    the inventory balance equation:  units_sold = begin_inventory + units_purchased - end_inventory.
    Revenue and gross profit are estimates built from that figure and inventory-weighted prices.
    Invoice date is used as the receiving date.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

LOG = logging.getLogger("get_vendor_summary")

# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
PERIOD_SQL = """
SELECT (SELECT MIN(startDate) FROM begin_inventory) AS period_start,
       (SELECT MAX(startDate) FROM begin_inventory) AS period_start_max,
       (SELECT MIN(endDate)   FROM end_inventory)   AS period_end,
       (SELECT MAX(endDate)   FROM end_inventory)   AS period_end_max
"""

PURCHASES_SQL = """
WITH canonical_name AS (            -- vendor name on the most recent invoice (2 vendors were renamed)
    SELECT VendorNumber, TRIM(VendorName) AS VendorName
    FROM (
        SELECT VendorNumber, VendorName,
               ROW_NUMBER() OVER (PARTITION BY VendorNumber
                                  ORDER BY InvoiceDate DESC, PONumber DESC) AS rn
        FROM vendor_invoice
    )
    WHERE rn = 1
),
catalog AS (                        -- SKUs each vendor supplies
    SELECT VendorNumber, COUNT(DISTINCT Brand) AS CatalogSKUs
    FROM purchase_prices
    GROUP BY VendorNumber
)
SELECT  i.VendorNumber,
        n.VendorName,
        i.PONumber,
        i.PODate,
        i.InvoiceDate,
        i.PayDate,
        SUBSTR(i.InvoiceDate, 1, 7)                                        AS InvoiceMonth,
        i.Quantity,
        i.Dollars,
        i.Freight,
        ROUND(i.Dollars + i.Freight, 2)                                    AS LandedCost,
        ROUND(1.0 * i.Dollars / i.Quantity, 4)                             AS UnitCost,
        ROUND(1.0 * i.Freight / i.Quantity, 4)                             AS FreightPerUnit,
        ROUND(100.0 * i.Freight / i.Dollars, 4)                            AS FreightPctOfDollars,
        CAST(julianday(i.InvoiceDate) - julianday(i.PODate) AS INTEGER)    AS LeadTimeDays,
        CAST(julianday(i.PayDate) - julianday(i.InvoiceDate) AS INTEGER)   AS PaymentTermsDays,
        NULLIF(TRIM(i.Approval), '')                                       AS ApprovedBy,
        COALESCE(c.CatalogSKUs, 0)                                         AS CatalogSKUs,
        CASE WHEN i.InvoiceDate BETWEEN :start AND :end THEN 1 ELSE 0 END  AS InPeriod
FROM vendor_invoice i
JOIN canonical_name n ON n.VendorNumber = i.VendorNumber
LEFT JOIN catalog c   ON c.VendorNumber = i.VendorNumber
ORDER BY i.InvoiceDate, i.PONumber
"""

SKU_SQL = """
WITH begin_units AS (
    SELECT Brand, SUM(onHand) AS BeginUnits FROM begin_inventory GROUP BY Brand
),
end_units AS (
    SELECT Brand, SUM(onHand) AS EndUnits FROM end_inventory GROUP BY Brand
),
snapshots AS (
    SELECT Brand, onHand, Price FROM begin_inventory
    UNION ALL
    SELECT Brand, onHand, Price FROM end_inventory
),
shelf AS (                          -- on-hand weighted store shelf price
    SELECT Brand, SUM(1.0 * onHand * Price) / SUM(onHand) AS ShelfPrice
    FROM snapshots
    GROUP BY Brand
    HAVING SUM(onHand) > 0
)
SELECT  p.Brand,
        p.Description,
        p.Size,
        p.Classification,
        p.VendorNumber,
        TRIM(p.VendorName)            AS VendorName,
        p.PurchasePrice,
        p.Price                       AS CatalogRetailPrice,
        COALESCE(b.BeginUnits, 0)     AS BeginUnits,
        COALESCE(e.EndUnits, 0)       AS EndUnits,
        s.ShelfPrice
FROM purchase_prices p
LEFT JOIN begin_units b ON b.Brand = p.Brand
LEFT JOIN end_units   e ON e.Brand = p.Brand
LEFT JOIN shelf       s ON s.Brand = p.Brand
"""

VENDOR_PURCHASE_SQL = """
SELECT  VendorNumber,
        COUNT(*)               AS PurchaseOrders,
        SUM(Quantity)          AS UnitsPurchased,
        SUM(Dollars)           AS PurchaseDollars,
        SUM(Freight)           AS FreightDollars,
        AVG(LeadTimeDays)      AS AvgLeadTimeDays,
        AVG(PaymentTermsDays)  AS AvgPaymentTermsDays
FROM purchases
WHERE InPeriod = 1
GROUP BY VendorNumber
"""

VENDOR_INVENTORY_SQL = """
SELECT  VendorNumber,
        COUNT(DISTINCT Brand)  AS CatalogSKUs,
        SUM(BeginUnits)        AS BeginUnits,
        SUM(EndUnits)          AS EndUnits,
        SUM(BeginValueAtCost)  AS BeginValueAtCost,
        SUM(EndValueAtCost)    AS EndValueAtCost
FROM sku_margins
GROUP BY VendorNumber
"""


# --------------------------------------------------------------------------- #
# Extract / transform
# --------------------------------------------------------------------------- #
def get_period(conn: sqlite3.Connection) -> tuple[pd.Timestamp, pd.Timestamp]:
    r = conn.execute(PERIOD_SQL).fetchone()
    if r[0] != r[1] or r[2] != r[3]:
        raise ValueError("Each inventory snapshot must have exactly one date")
    return pd.Timestamp(r[0]), pd.Timestamp(r[2])


def build_purchases(conn: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_sql_query(
        PURCHASES_SQL, conn, params={"start": str(start.date()), "end": str(end.date())}
    )
    if df["PONumber"].duplicated().any():
        raise ValueError("PONumber is expected to be unique in the invoice file")
    # fail loudly rather than produce wrong numbers
    assert (df["Quantity"] > 0).all(), "Non-positive quantity found"
    assert (df["Dollars"] > 0).all(), "Non-positive dollars found"
    assert (df["Freight"] >= 0).all(), "Negative freight found"
    assert (df["LeadTimeDays"] >= 0).all(), "Invoice before PO date found"
    assert (df["PaymentTermsDays"] >= 0).all(), "Pay date before invoice date found"
    df["InPeriod"] = df["InPeriod"].astype(bool)
    return df


def build_sku_table(conn: sqlite3.Connection) -> pd.DataFrame:
    sku = pd.read_sql_query(SKU_SQL, conn)
    sku["AvgUnits"] = (sku["BeginUnits"] + sku["EndUnits"]) / 2
    # fall back to the catalog price when a brand never had stock on hand
    sku["RetailPrice"] = sku["ShelfPrice"].fillna(sku["CatalogRetailPrice"])
    valid = (sku["RetailPrice"] > 0) & (sku["PurchasePrice"] > 0)
    sku["ValidPricing"] = valid
    sku["UnitMargin"] = np.where(valid, sku["RetailPrice"] - sku["PurchasePrice"], np.nan)
    sku["MarginPct"] = np.where(valid, sku["UnitMargin"] / sku["RetailPrice"] * 100, np.nan)
    sku["BeginValueAtCost"] = sku["BeginUnits"] * sku["PurchasePrice"]
    sku["EndValueAtCost"] = sku["EndUnits"] * sku["PurchasePrice"]
    cols = [
        "Brand", "Description", "Size", "Classification", "VendorNumber", "VendorName",
        "PurchasePrice", "CatalogRetailPrice", "RetailPrice", "UnitMargin", "MarginPct",
        "BeginUnits", "EndUnits", "AvgUnits", "BeginValueAtCost", "EndValueAtCost", "ValidPricing",
    ]
    return sku[cols]


def _weighted_mix(sku: pd.DataFrame) -> pd.DataFrame:
    """Per vendor: inventory-weighted retail and purchase price (valid pricing only)."""
    s = sku[sku["ValidPricing"]].copy()
    s["w"] = s["AvgUnits"]
    has_weight = s.groupby("VendorNumber")["w"].transform("sum") > 0
    s.loc[~has_weight, "w"] = 1.0          # vendors whose SKUs never had stock -> equal weights
    s["wr"] = s["w"] * s["RetailPrice"]
    s["wc"] = s["w"] * s["PurchasePrice"]
    g = s.groupby("VendorNumber").agg(w=("w", "sum"), wr=("wr", "sum"), wc=("wc", "sum"))
    return pd.DataFrame({"WtdRetailPrice": g["wr"] / g["w"], "WtdCatalogCost": g["wc"] / g["w"]})


def build_vendor_summary(
    conn: sqlite3.Connection,
    purchases: pd.DataFrame,
    sku: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    period_days = (end - start).days + 1

    pg = pd.read_sql_query(VENDOR_PURCHASE_SQL, conn, index_col="VendorNumber")
    in_p = purchases[purchases["InPeriod"]]
    pg["LeadTimeStdDays"] = in_p.groupby("VendorNumber")["LeadTimeDays"].std()   # SQLite has no STDDEV
    pg["LandedCost"] = pg["PurchaseDollars"] + pg["FreightDollars"]
    pg["AvgUnitCost"] = pg["PurchaseDollars"] / pg["UnitsPurchased"]
    pg["AvgPOValue"] = pg["PurchaseDollars"] / pg["PurchaseOrders"]
    pg["FreightPctOfPurchases"] = pg["FreightDollars"] / pg["PurchaseDollars"] * 100
    pg["LeadTimeCV"] = pg["LeadTimeStdDays"] / pg["AvgLeadTimeDays"]

    ig = pd.read_sql_query(VENDOR_INVENTORY_SQL, conn, index_col="VendorNumber")
    mix = _weighted_mix(sku)
    names = (
        pd.concat([purchases[["VendorNumber", "VendorName"]], sku[["VendorNumber", "VendorName"]]])
        .drop_duplicates("VendorNumber")
        .set_index("VendorNumber")
    )

    df = names.join(pg, how="left").join(ig, how="left").join(mix, how="left")
    zero_fill = ["PurchaseOrders", "UnitsPurchased", "PurchaseDollars", "FreightDollars", "LandedCost",
                 "CatalogSKUs", "BeginUnits", "EndUnits", "BeginValueAtCost", "EndValueAtCost"]
    df[zero_fill] = df[zero_fill].fillna(0)
    df["HasPurchases"] = df["PurchaseOrders"] > 0

    # --- estimated sales via the inventory balance equation
    df["UnitsAvailable"] = df["BeginUnits"] + df["UnitsPurchased"]
    df["EstUnitsSold"] = df["UnitsAvailable"] - df["EndUnits"]
    df["SellThroughRate"] = df["EstUnitsSold"] / df["UnitsAvailable"].where(df["UnitsAvailable"] > 0)
    df["AvgInventoryUnits"] = (df["BeginUnits"] + df["EndUnits"]) / 2
    df["StockTurnover"] = df["EstUnitsSold"] / df["AvgInventoryUnits"].where(df["AvgInventoryUnits"] > 0)
    df["DaysOfInventory"] = period_days / df["StockTurnover"].where(df["StockTurnover"] > 0)
    df["EstRevenue"] = df["EstUnitsSold"] * df["WtdRetailPrice"]
    df["EstCOGS"] = df["EstUnitsSold"] * df["WtdCatalogCost"]
    df["EstGrossProfit"] = df["EstRevenue"] - df["EstCOGS"]
    df["GrossMarginPct"] = (df["WtdRetailPrice"] - df["WtdCatalogCost"]) / df["WtdRetailPrice"] * 100
    df["EstGrossProfitAfterFreight"] = df["EstGrossProfit"] - df["FreightDollars"]
    df["InventoryGrowthPct"] = (df["EndValueAtCost"] - df["BeginValueAtCost"]) / df[
        "BeginValueAtCost"
    ].where(df["BeginValueAtCost"] > 0) * 100
    df["UnsoldCapitalAtCost"] = df["EndValueAtCost"]

    # --- reliability flag; sales-based KPIs are blanked for unreliable vendors
    df["EstimateReliable"] = (
        df["HasPurchases"]
        & (df["UnitsAvailable"] >= cfg.MIN_UNITS_AVAILABLE)
        & (df["PurchaseOrders"] >= cfg.MIN_PO_COUNT)
        & (df["EstUnitsSold"] > 0)
        & df["SellThroughRate"].between(0, 1)
        & df["GrossMarginPct"].notna()
    )
    sales_cols = ["EstUnitsSold", "SellThroughRate", "StockTurnover", "DaysOfInventory",
                  "EstRevenue", "EstCOGS", "EstGrossProfit", "EstGrossProfitAfterFreight"]
    df.loc[~df["EstimateReliable"], sales_cols] = np.nan

    # --- spend share / concentration
    total = df["PurchaseDollars"].sum()
    df["PurchaseSharePct"] = df["PurchaseDollars"] / total * 100
    df = df.sort_values("PurchaseDollars", ascending=False)
    df["CumulativeSharePct"] = df["PurchaseSharePct"].cumsum()

    df = _add_scores(df)
    df = _add_segments(df)
    return df.reset_index().rename(columns={"index": "VendorNumber"})


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-100 score from percentile ranks across reliable vendors."""
    metrics = list(cfg.SCORE_WEIGHTS)
    elig = df["EstimateReliable"] & df[metrics].notna().all(axis=1)
    df["ScoreEligible"] = elig
    score = pd.Series(0.0, index=df.index)
    for metric, (weight, higher_better) in cfg.SCORE_WEIGHTS.items():
        pct = df.loc[elig, metric].rank(pct=True, method="average")
        if not higher_better:
            pct = 1 - pct + 1 / elig.sum()          # keeps the best vendor at 1.0
        score.loc[elig] += weight * pct
    df["VendorScore"] = (score * 100).where(elig).round(1)
    df["ScoreRank"] = df["VendorScore"].rank(ascending=False, method="min")
    return df


def _add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Margin x turnover quadrant (medians of eligible vendors) for purchasing strategy."""
    rel = df[df["ScoreEligible"]]
    m_med, t_med = rel["GrossMarginPct"].median(), rel["StockTurnover"].median()

    def seg(r: pd.Series) -> str:
        if not r["ScoreEligible"]:
            return "Insufficient data"
        hi_m, hi_t = r["GrossMarginPct"] >= m_med, r["StockTurnover"] >= t_med
        if hi_m and hi_t:
            return "Star - expand"
        if hi_m:
            return "Margin builder - tighten stock levels"
        if hi_t:
            return "Volume driver - renegotiate cost"
        return "Underperformer - review or exit"

    df["Segment"] = df.apply(seg, axis=1)
    return df


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #
def build_quality_report(
    conn: sqlite3.Connection,
    purchases: pd.DataFrame,
    sku: pd.DataFrame,
    summary: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    invoiced = set(purchases["VendorNumber"])
    cataloged = set(sku["VendorNumber"])
    after = purchases[~purchases["InPeriod"]]
    in_p = purchases[purchases["InPeriod"]]

    begin_units = conn.execute("SELECT SUM(onHand) FROM begin_inventory").fetchone()[0]
    end_units = conn.execute("SELECT SUM(onHand) FROM end_inventory").fetchone()[0]
    missing_city = conn.execute("SELECT COUNT(*) FROM end_inventory WHERE City IS NULL").fetchone()[0]
    missing_desc = conn.execute(
        "SELECT COUNT(*) FROM purchase_prices WHERE Description IS NULL"
    ).fetchone()[0]
    renamed = conn.execute(
        """SELECT VendorNumber, GROUP_CONCAT(DISTINCT TRIM(VendorName))
           FROM vendor_invoice GROUP BY VendorNumber HAVING COUNT(DISTINCT TRIM(VendorName)) > 1"""
    ).fetchall()

    # reconciliation: summary totals must tie back to the source tables
    assert np.isclose(summary["PurchaseDollars"].sum(), in_p["Dollars"].sum())
    assert int(summary["UnitsPurchased"].sum()) == int(in_p["Quantity"].sum())
    assert int(summary["BeginUnits"].sum()) == int(begin_units)
    assert int(summary["EndUnits"].sum()) == int(end_units)

    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "purchase_orders_total": int(len(purchases)),
        "purchase_orders_in_period": int(purchases["InPeriod"].sum()),
        "purchase_orders_after_period_end": int(len(after)),
        "dollars_after_period_end_excluded": round(float(after["Dollars"].sum()), 2),
        "vendors_with_invoices": len(invoiced),
        "vendors_in_catalog": len(cataloged),
        "vendors_invoiced_but_not_in_catalog": sorted(int(x) for x in invoiced - cataloged),
        "vendors_in_catalog_without_invoices": sorted(int(x) for x in cataloged - invoiced),
        "vendors_renamed_during_period": {int(k): v.split(",") for k, v in renamed},
        "catalog_rows_with_invalid_pricing": int((~sku["ValidPricing"]).sum()),
        "catalog_rows_missing_description": int(missing_desc),
        "end_inventory_rows_missing_city": int(missing_city),
        "vendors_reliable_for_sales_estimate": int(summary["EstimateReliable"].sum()),
        "vendors_score_eligible": int(summary["ScoreEligible"].sum()),
        "reconciliation": "purchase dollars/units and begin/end inventory units tie to the source tables",
        "assumptions": [
            "Units sold are estimated: begin inventory + purchased units - end inventory (per vendor).",
            "Invoice date is used as the receiving date.",
            "Retail price = on-hand weighted store price; cost = catalog PurchasePrice.",
            "Estimated revenue/profit use inventory-weighted brand mix as a proxy for the sales mix.",
            f"Sales KPIs require >= {cfg.MIN_PO_COUNT} POs and >= {cfg.MIN_UNITS_AVAILABLE} units available.",
        ],
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(data_dir: Path, db_path: Path) -> dict[str, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found - run scripts/ingestion_db.py first")

    with sqlite3.connect(db_path) as conn:
        start, end = get_period(conn)
        LOG.info("Analysis period: %s -> %s", start.date(), end.date())

        purchases = build_purchases(conn, start, end)
        purchases.to_sql("purchases", conn, if_exists="replace", index=False)

        sku = build_sku_table(conn)
        sku.to_sql("sku_margins", conn, if_exists="replace", index=False)

        summary = build_vendor_summary(conn, purchases, sku, start, end)
        summary.to_sql("vendor_summary", conn, if_exists="replace", index=False)

        quality = build_quality_report(conn, purchases, sku, summary, start, end)

    # CSV exports (dates as ISO text, InPeriod as True/False)
    purchases.to_csv(data_dir / cfg.OUT_PURCHASES, index=False)
    summary.to_csv(data_dir / cfg.OUT_SUMMARY, index=False)
    sku.to_csv(data_dir / cfg.OUT_SKU, index=False)
    (data_dir / cfg.OUT_QUALITY).write_text(json.dumps(quality, indent=2))

    LOG.info("Wrote %s (%d rows)", cfg.OUT_PURCHASES, len(purchases))
    LOG.info("Wrote %s (%d rows)", cfg.OUT_SUMMARY, len(summary))
    LOG.info("Wrote %s (%d rows)", cfg.OUT_SKU, len(sku))
    LOG.info("Wrote %s", cfg.OUT_QUALITY)
    return {"purchases": purchases, "summary": summary, "sku": sku}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vendor performance tables from SQLite")
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--db", type=Path, default=None, help="default: <data-dir>/inventory.db")
    args = parser.parse_args()
    cfg.setup_logging()
    run(args.data_dir, args.db or args.data_dir / cfg.DB_NAME)


if __name__ == "__main__":
    main()
