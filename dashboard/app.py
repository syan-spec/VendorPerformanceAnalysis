"""
Vendor Performance Dashboard (Streamlit)
========================================

Run from the project root:
    streamlit run dashboard/app.py

Tabs: executive overview, vendor scorecard, profitability & inventory, procurement operations,
freight cost model, invoice risk (manual approval) model, insights & statistics, methodology.

If the generated files are missing, the app builds them (ingestion -> summary -> models).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analysis  # noqa: E402
import config as cfg  # noqa: E402
from ml_utils import load_artifacts, score_invoices  # noqa: E402

st.set_page_config(page_title="Vendor Performance Analysis", page_icon="📦", layout="wide")

_chart_counter = {"n": 0}


def show(fig: go.Figure) -> None:
    """Render a Plotly chart full-width with a unique key (filters can produce identical figures)."""
    _chart_counter["n"] += 1
    st.plotly_chart(fig, width="stretch", key=f"chart_{_chart_counter['n']}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
REQUIRED = [
    cfg.DATA_DIR / cfg.OUT_PURCHASES, cfg.DATA_DIR / cfg.OUT_SUMMARY, cfg.DATA_DIR / cfg.OUT_SKU,
    cfg.DATA_DIR / cfg.OUT_QUALITY, cfg.DATA_DIR / cfg.OUT_INVOICE_RISK,
    cfg.MODELS_DIR / "freight_model.joblib", cfg.MODELS_DIR / "approval_model.joblib",
]


def build_everything() -> None:
    import get_vendor_summary
    import ingestion_db
    import train_approval_model
    import train_freight_model

    if not cfg.DB_PATH.exists():
        ingestion_db.main()
    get_vendor_summary.run(cfg.DATA_DIR, cfg.DB_PATH)
    train_freight_model.run()
    train_approval_model.run()


@st.cache_data(show_spinner="Loading data...")
def load_data():
    if not all(p.exists() for p in REQUIRED):
        with st.spinner("First run: building tables and training models (about a minute)..."):
            build_everything()
    purchases, summary, sku = analysis.load_tables()
    quality = json.loads((cfg.DATA_DIR / cfg.OUT_QUALITY).read_text())
    risk = pd.read_csv(cfg.DATA_DIR / cfg.OUT_INVOICE_RISK, parse_dates=["InvoiceDate"])
    fm = json.loads((cfg.MODELS_DIR / cfg.OUT_FREIGHT_METRICS).read_text())
    am = json.loads((cfg.MODELS_DIR / cfg.OUT_APPROVAL_METRICS).read_text())
    return purchases, summary, sku, quality, risk, fm, am


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def cached_margin_test(summary: pd.DataFrame):
    return analysis.margin_test(summary)


@st.cache_data(show_spinner=False)
def cached_bulk(purchases: pd.DataFrame):
    return analysis.bulk_order_effect(purchases)


@st.cache_data(show_spinner=False)
def cached_promo(sku: pd.DataFrame):
    return analysis.promotion_candidates(sku)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.0f}"


def short(name: str, n: int = 26) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


SEGMENT_COLORS = {
    "Star - expand": "#2e7d32",
    "Margin builder - tighten stock levels": "#1565c0",
    "Volume driver - renegotiate cost": "#ef6c00",
    "Underperformer - review or exit": "#c62828",
    "Insufficient data": "#9e9e9e",
}
FLAG_LABELS = {
    "Flag_HighValue": "High value (>= $250K)",
    "Flag_FreightAnomaly": "Freight overcharge",
    "Flag_UnitCostOutlier": "Unit cost outlier",
    "Flag_LeadTimeOutlier": "Lead-time outlier",
    "Flag_PaymentTermsOutlier": "Payment-terms outlier",
    "Flag_PossibleDuplicate": "Possible duplicate",
}

purchases, summary, sku, quality, risk, fm, am = load_data()

st.title("📦 Vendor Performance Analysis")
st.caption(
    "Retail inventory & sales · Vendor efficiency and profitability for strategic purchasing and "
    f"inventory decisions · Period {quality['period_start']} → {quality['period_end']}"
)

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
segments = sorted(summary["Segment"].unique())
sel_segments = st.sidebar.multiselect("Vendor segment", segments, default=segments)
min_spend = st.sidebar.number_input("Minimum purchase $ per vendor", min_value=0, value=0, step=10_000)
months = sorted(purchases.loc[purchases["InPeriod"], "InvoiceMonth"].unique())
m_from, m_to = st.sidebar.select_slider(
    "Invoice months (purchase charts)", options=months, value=(months[0], months[-1])
)
st.sidebar.info(
    "Vendor KPIs cover the full inventory period. The month range only affects purchase-order level "
    "charts. The model tabs always use all invoices."
)

vf = summary[summary["Segment"].isin(sel_segments) & (summary["PurchaseDollars"] >= min_spend)].copy()
pf = purchases[
    purchases["InPeriod"]
    & purchases["VendorNumber"].isin(vf["VendorNumber"])
    & (purchases["InvoiceMonth"] >= m_from)
    & (purchases["InvoiceMonth"] <= m_to)
].copy()
rel = vf[vf["EstimateReliable"]]

tabs = st.tabs([
    "Executive overview", "Vendor scorecard", "Profitability & inventory", "Procurement operations",
    "Freight cost model", "Invoice risk model", "Insights & statistics", "Methodology",
])

# --------------------------------------------------------------------------- #
# 1. Executive overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    if vf.empty:
        st.warning("No vendors match the current filters.")
    else:
        total_spend = vf["PurchaseDollars"].sum()
        top10 = vf.nlargest(10, "PurchaseDollars")["PurchaseDollars"].sum()
        gp, rev = rel["EstGrossProfit"].sum(), rel["EstRevenue"].sum()
        c = st.columns(6)
        c[0].metric("Purchases", money(total_spend))
        c[1].metric("Freight", money(vf["FreightDollars"].sum()))
        c[2].metric("Units purchased", f"{vf['UnitsPurchased'].sum():,.0f}")
        c[3].metric("Vendors with purchases", f"{int(vf['HasPurchases'].sum())}")
        c[4].metric("Top-10 vendor share", f"{top10 / total_spend * 100:.1f}%" if total_spend else "n/a")
        c[5].metric("Est. gross margin", f"{gp / rev * 100:.1f}%" if rev else "n/a",
                    help="Estimated from reliable vendors only (see Methodology).")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Purchase concentration (Pareto)")
            par = vf.nlargest(15, "PurchaseDollars").copy()
            par["Cum"] = par["PurchaseDollars"].cumsum() / total_spend * 100
            fig = go.Figure()
            fig.add_bar(x=par["VendorName"].map(short), y=par["PurchaseDollars"], name="Purchases $",
                        marker_color="#1565c0")
            fig.add_scatter(x=par["VendorName"].map(short), y=par["Cum"], name="Cumulative % of spend",
                            yaxis="y2", mode="lines+markers", line=dict(color="#ef6c00"))
            fig.update_layout(
                yaxis=dict(title="Purchases ($)"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                legend=dict(orientation="h", y=1.12), margin=dict(t=30, b=10), height=420)
            show(fig)
        with right:
            st.subheader("Spend by segment")
            seg = vf.groupby("Segment", as_index=False)["PurchaseDollars"].sum()
            fig = px.pie(seg, names="Segment", values="PurchaseDollars", hole=0.5,
                         color="Segment", color_discrete_map=SEGMENT_COLORS)
            fig.update_layout(margin=dict(t=30, b=10), height=420, legend=dict(orientation="h", y=-0.15))
            show(fig)

        st.subheader("Monthly purchases")
        monthly = pf.groupby("InvoiceMonth", as_index=False).agg(
            Purchases=("Dollars", "sum"), Freight=("Freight", "sum"), POs=("PONumber", "count"))
        fig = px.bar(monthly, x="InvoiceMonth", y="Purchases", hover_data=["Freight", "POs"],
                     color_discrete_sequence=["#1565c0"])
        fig.update_layout(margin=dict(t=10, b=10), height=320, xaxis_title=None)
        show(fig)

# --------------------------------------------------------------------------- #
# 2. Vendor scorecard
# --------------------------------------------------------------------------- #
with tabs[1]:
    if vf.empty:
        st.warning("No vendors match the current filters.")
    else:
        st.subheader("Composite vendor score")
        st.caption("Percentile-rank score (0-100) across vendors with reliable estimates. Weights: "
                   + ", ".join(f"{k} {w * 100:.0f}%" for k, (w, _) in cfg.SCORE_WEIGHTS.items()))
        scored = vf[vf["ScoreEligible"]].sort_values("VendorScore", ascending=False)
        if scored.empty:
            st.info("No score-eligible vendors under the current filters.")
        else:
            a, b = st.columns(2)
            for col, title, data in ((a, "Top 10 vendors by score", scored.head(10)),
                                     (b, "Bottom 10 vendors by score", scored.tail(10))):
                with col:
                    st.markdown(f"**{title}**")
                    t = data.iloc[::-1]
                    fig = px.bar(t, x="VendorScore", y=t["VendorName"].map(short), orientation="h",
                                 color="Segment", color_discrete_map=SEGMENT_COLORS)
                    fig.update_layout(height=400, yaxis_title=None, margin=dict(t=10, b=10), showlegend=False)
                    show(fig)

        table_cols = [
            "VendorNumber", "VendorName", "Segment", "VendorScore", "PurchaseDollars", "PurchaseSharePct",
            "PurchaseOrders", "GrossMarginPct", "SellThroughRate", "StockTurnover", "DaysOfInventory",
            "AvgLeadTimeDays", "FreightPctOfPurchases", "AvgPaymentTermsDays", "EstGrossProfit",
            "EndValueAtCost", "EstimateReliable"]
        st.dataframe(
            vf.sort_values("PurchaseDollars", ascending=False)[table_cols], hide_index=True, width="stretch",
            column_config={
                "VendorScore": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "PurchaseDollars": st.column_config.NumberColumn("Purchases $", format="dollar"),
                "PurchaseSharePct": st.column_config.NumberColumn("Share %", format="%.2f"),
                "GrossMarginPct": st.column_config.NumberColumn("Gross margin %", format="%.1f"),
                "SellThroughRate": st.column_config.NumberColumn("Sell-through", format="%.2f"),
                "StockTurnover": st.column_config.NumberColumn("Turnover", format="%.2f"),
                "DaysOfInventory": st.column_config.NumberColumn("Days of inv.", format="%.0f"),
                "AvgLeadTimeDays": st.column_config.NumberColumn("Lead time (d)", format="%.1f"),
                "FreightPctOfPurchases": st.column_config.NumberColumn("Freight %", format="%.2f"),
                "AvgPaymentTermsDays": st.column_config.NumberColumn("Terms (d)", format="%.1f"),
                "EstGrossProfit": st.column_config.NumberColumn("Est. gross profit", format="dollar"),
                "EndValueAtCost": st.column_config.NumberColumn("Ending inv. at cost", format="dollar"),
            })
        st.download_button("Download vendor summary (CSV)", vf.to_csv(index=False).encode(),
                           "vendor_summary_filtered.csv", "text/csv")

        st.divider()
        st.subheader("Vendor drill-down")
        pick = st.selectbox("Vendor", vf.sort_values("PurchaseDollars", ascending=False)["VendorName"].tolist())
        v = vf[vf["VendorName"] == pick].iloc[0]
        d = st.columns(5)
        d[0].metric("Purchases", money(v["PurchaseDollars"]))
        d[1].metric("Purchase orders", f"{int(v['PurchaseOrders'])}")
        d[2].metric("Gross margin", "n/a" if pd.isna(v["GrossMarginPct"]) else f"{v['GrossMarginPct']:.1f}%")
        d[3].metric("Sell-through", "n/a" if pd.isna(v["SellThroughRate"]) else f"{v['SellThroughRate']:.0%}")
        d[4].metric("Segment", v["Segment"].split(" - ")[0])
        if not bool(v["EstimateReliable"]):
            st.warning("Sales-based KPIs are hidden for this vendor: too few orders/units for a reliable estimate.")
        vp = purchases[(purchases["VendorNumber"] == v["VendorNumber"]) & purchases["InPeriod"]]
        vr = risk[risk["VendorNumber"] == v["VendorNumber"]]
        if len(vr):
            st.caption(f"Invoices flagged for manual approval by the risk model: "
                       f"{int(vr['ModelFlagged'].sum())} of {len(vr)}")
        e1, e2 = st.columns(2)
        with e1:
            if not vp.empty:
                m = vp.groupby("InvoiceMonth", as_index=False)["Dollars"].sum()
                fig = px.bar(m, x="InvoiceMonth", y="Dollars", title="Monthly purchases",
                             color_discrete_sequence=["#1565c0"]).update_layout(height=320)
                show(fig)
        with e2:
            vs = sku[sku["VendorNumber"] == v["VendorNumber"]].nlargest(10, "EndValueAtCost")
            st.markdown("**Top SKUs by ending inventory value (at cost)**")
            st.dataframe(
                vs[["Brand", "Description", "Size", "PurchasePrice", "RetailPrice", "MarginPct", "EndUnits",
                    "EndValueAtCost"]], hide_index=True, width="stretch",
                column_config={
                    "MarginPct": st.column_config.NumberColumn("Margin %", format="%.1f"),
                    "EndValueAtCost": st.column_config.NumberColumn("Ending value", format="dollar"),
                    "PurchasePrice": st.column_config.NumberColumn("Cost", format="dollar"),
                    "RetailPrice": st.column_config.NumberColumn("Retail", format="dollar"),
                })

# --------------------------------------------------------------------------- #
# 3. Profitability & inventory
# --------------------------------------------------------------------------- #
with tabs[2]:
    if vf.empty:
        st.warning("No vendors match the current filters.")
    else:
        st.subheader("Margin vs. stock turnover")
        sc = vf[vf["ScoreEligible"]].copy()
        if sc.empty:
            st.info("No score-eligible vendors under the current filters.")
        else:
            fig = px.scatter(
                sc, x="StockTurnover", y="GrossMarginPct", size="PurchaseDollars", color="Segment",
                color_discrete_map=SEGMENT_COLORS, hover_name="VendorName", size_max=50,
                hover_data={"PurchaseDollars": ":$,.0f", "VendorScore": ":.1f", "StockTurnover": ":.2f",
                            "GrossMarginPct": ":.1f"})
            elig_all = summary[summary["ScoreEligible"]]
            fig.add_hline(y=elig_all["GrossMarginPct"].median(), line_dash="dash", line_color="grey",
                          annotation_text="median margin")
            fig.add_vline(x=elig_all["StockTurnover"].median(), line_dash="dash", line_color="grey",
                          annotation_text="median turnover")
            fig.update_layout(height=520, margin=dict(t=10), xaxis_title="Stock turnover (x per period)",
                              yaxis_title="Gross margin %")
            show(fig)
            st.caption("Bubble size = purchase dollars. Quadrant lines are medians across all score-eligible vendors.")

        a, b = st.columns(2)
        with a:
            st.subheader("Top vendors by estimated gross profit")
            t = rel.nlargest(10, "EstGrossProfit").iloc[::-1]
            if not t.empty:
                fig = px.bar(t, x="EstGrossProfit", y=t["VendorName"].map(short), orientation="h",
                             color_discrete_sequence=["#2e7d32"])
                fig.update_layout(height=400, yaxis_title=None, margin=dict(t=10))
                show(fig)
        with b:
            st.subheader("Capital tied up in ending inventory")
            t = vf.nlargest(10, "EndValueAtCost").iloc[::-1]
            fig = px.bar(t, x="EndValueAtCost", y=t["VendorName"].map(short), orientation="h",
                         color="Segment", color_discrete_map=SEGMENT_COLORS)
            fig.update_layout(height=400, yaxis_title=None, margin=dict(t=10), showlegend=False)
            show(fig)

        st.subheader("Slow movers: highest days of inventory (vendors > $100K purchases)")
        slow = rel[rel["PurchaseDollars"] > 100_000].nlargest(10, "DaysOfInventory")
        st.dataframe(
            slow[["VendorName", "PurchaseDollars", "StockTurnover", "DaysOfInventory", "SellThroughRate",
                  "EndValueAtCost", "InventoryGrowthPct"]], hide_index=True, width="stretch",
            column_config={
                "PurchaseDollars": st.column_config.NumberColumn(format="dollar"),
                "EndValueAtCost": st.column_config.NumberColumn("Ending inv. at cost", format="dollar"),
                "StockTurnover": st.column_config.NumberColumn(format="%.2f"),
                "DaysOfInventory": st.column_config.NumberColumn(format="%.0f"),
                "SellThroughRate": st.column_config.NumberColumn(format="%.2f"),
                "InventoryGrowthPct": st.column_config.NumberColumn("Inv. growth %", format="%.1f"),
            })

        st.subheader("Promotion / repricing candidates (SKU level)")
        promo = cached_promo(sku)
        promo = promo[promo["VendorNumber"].isin(vf["VendorNumber"])]
        st.caption("High-margin SKUs (top quartile) whose inventory grew during the year and whose ending stock "
                   "value is in the top quartile. Brand-level sales are not available, so growing stock is the "
                   "proxy for slow movement.")
        c = st.columns(3)
        c[0].metric("SKUs", f"{len(promo):,}")
        c[1].metric("Ending inventory at cost", money(promo["EndValueAtCost"].sum()))
        c[2].metric("Avg margin", f"{promo['MarginPct'].mean():.1f}%" if len(promo) else "n/a")
        st.dataframe(
            promo.head(50)[["Brand", "Description", "Size", "VendorName", "PurchasePrice", "RetailPrice",
                            "MarginPct", "BeginUnits", "EndUnits", "EndValueAtCost"]],
            hide_index=True, width="stretch",
            column_config={
                "MarginPct": st.column_config.NumberColumn("Margin %", format="%.1f"),
                "EndValueAtCost": st.column_config.NumberColumn("Ending value", format="dollar"),
                "PurchasePrice": st.column_config.NumberColumn("Cost", format="dollar"),
                "RetailPrice": st.column_config.NumberColumn("Retail", format="dollar"),
            })

# --------------------------------------------------------------------------- #
# 4. Procurement operations
# --------------------------------------------------------------------------- #
with tabs[3]:
    if pf.empty:
        st.info("No purchase orders under the current filters.")
    else:
        c = st.columns(4)
        c[0].metric("Avg lead time (PO → invoice)", f"{pf['LeadTimeDays'].mean():.1f} days")
        c[1].metric("Avg payment terms", f"{pf['PaymentTermsDays'].mean():.1f} days")
        c[2].metric("Freight % of purchases", f"{pf['Freight'].sum() / pf['Dollars'].sum() * 100:.2f}%")
        c[3].metric("Purchase orders", f"{len(pf):,}")
        a, b = st.columns(2)
        with a:
            st.subheader("Lead-time distribution")
            fig = px.histogram(pf, x="LeadTimeDays", nbins=16, color_discrete_sequence=["#1565c0"])
            fig.update_layout(height=340, margin=dict(t=10), bargap=0.05)
            show(fig)
        with b:
            st.subheader("Payment terms distribution")
            fig = px.histogram(pf, x="PaymentTermsDays", nbins=26, color_discrete_sequence=["#6a1b9a"])
            fig.update_layout(height=340, margin=dict(t=10), bargap=0.05)
            show(fig)

        st.subheader("Lead time by top-15 vendors (by spend)")
        top_v = vf.nlargest(15, "PurchaseDollars")["VendorName"]
        bx = pf[pf["VendorName"].isin(top_v)].copy()
        bx["Vendor"] = bx["VendorName"].map(short)
        fig = px.box(bx, x="Vendor", y="LeadTimeDays", color_discrete_sequence=["#1565c0"])
        fig.update_layout(height=380, margin=dict(t=10), xaxis_title=None)
        show(fig)

        st.subheader("Freight burden vs. order size")
        fig = px.scatter(pf, x="Dollars", y="FreightPctOfDollars", log_x=True, opacity=0.4,
                         hover_name="VendorName", color_discrete_sequence=["#ef6c00"])
        fig.update_layout(height=380, margin=dict(t=10), xaxis_title="Invoice $ (log scale)",
                          yaxis_title="Freight as % of invoice $")
        show(fig)
        small, large = pf[pf["Dollars"] < 1_000], pf[pf["Dollars"] >= 50_000]
        s_pct = small["Freight"].sum() / small["Dollars"].sum() * 100 if len(small) else float("nan")
        l_pct = large["Freight"].sum() / large["Dollars"].sum() * 100 if len(large) else float("nan")
        st.caption(
            f"Freight is {s_pct:.2f}% of invoice $ on orders under $1K vs {l_pct:.2f}% on orders of $50K+. "
            + ("The burden is materially higher on small orders, which supports consolidating purchases."
               if s_pct > 1.25 * l_pct else
               "The burden is roughly flat across order sizes, so freight is not a strong consolidation lever."))

# --------------------------------------------------------------------------- #
# 5. Freight cost model
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Freight cost prediction")
    st.markdown(
        "**Question:** what freight should an invoice carry? Temporal hold-out: trained on invoices before "
        f"**{fm['split']['cutoff']}**, tested on **{fm['split']['test_rows']:,}** invoices from that date on.")
    sel = fm["selected_test_metrics"]
    c = st.columns(5)
    c[0].metric("Selected model", fm["selected_model"].split(" (")[0])
    c[1].metric("Test R²", f"{sel['R2']:.4f}")
    c[2].metric("Test MAE", f"${sel['MAE']:.2f}")
    c[3].metric("Median % error", f"{sel['MedianAPE_pct']:.1f}%")
    c[4].metric("Within ±10%", f"{sel['Within10pct']:.0%}")
    if fm.get("linear_model_equation"):
        st.info(f"Fitted equation: **{fm['linear_model_equation']}**")
    base = fm["baseline_test_metrics"]
    if fm["ml_beats_baseline"]:
        st.success(f"The ML model improves test MAE by {fm['ml_lift_vs_baseline_test_MAE_pct']:.1f}% over the "
                   "0.5% rule of thumb.")
    else:
        st.warning(
            f"No ML model meaningfully beats the rule 'freight = {fm['typical_freight_rate_pct']:.2f}% of invoice "
            f"dollars' (test MAE ${sel['MAE']:.2f} vs ${base['MAE']:.2f} for the baseline). Freight is almost "
            "purely proportional to order value, so the simplest model is selected.")

    cand = pd.DataFrame(fm["candidates"]).T.reset_index().rename(columns={"index": "Model"})
    a, b = st.columns(2)
    with a:
        fig = px.bar(cand, x="test_MAE", y="Model", orientation="h", title="Hold-out MAE ($, lower is better)",
                     color_discrete_sequence=["#1565c0"])
        fig.update_layout(height=340, yaxis_title=None, margin=dict(t=40))
        show(fig)
    with b:
        te = risk[risk["InvoiceDate"] >= cfg.TEMPORAL_CUTOFF]
        fig = px.scatter(te, x="ExpectedFreight", y="Freight", log_x=True, log_y=True, opacity=0.4,
                         title="Hold-out: actual vs predicted freight", color_discrete_sequence=["#1565c0"])
        lim = [te[["ExpectedFreight", "Freight"]].min().min(), te[["ExpectedFreight", "Freight"]].max().max()]
        fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="black"), name="perfect")
        fig.update_layout(height=340, margin=dict(t=40), showlegend=False)
        show(fig)
    st.dataframe(
        cand[["Model", "cv_MAE", "test_MAE", "test_RMSE", "test_R2", "test_MedianAPE_pct", "test_Within10pct"]],
        hide_index=True, width="stretch",
        column_config={
            "cv_MAE": st.column_config.NumberColumn("CV MAE", format="%.2f"),
            "test_MAE": st.column_config.NumberColumn("Test MAE", format="%.2f"),
            "test_RMSE": st.column_config.NumberColumn("Test RMSE", format="%.2f"),
            "test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
            "test_MedianAPE_pct": st.column_config.NumberColumn("Median % err", format="%.2f"),
            "test_Within10pct": st.column_config.NumberColumn("Within ±10%", format="%.2f"),
        })

    st.subheader("Freight overcharges found by the model")
    fo = fm["freight_overcharge_summary"]
    c = st.columns(4)
    c[0].metric("Overcharged invoices", f"{fo['invoices_flagged_freight_anomaly']}")
    c[1].metric("Estimated excess freight", money(fo["estimated_excess_freight_$"]))
    c[2].metric("Median actual / expected", f"{fo['median_actual_to_expected_ratio']:.1f}x")
    c[3].metric("Dates affected", f"{fo['first_invoice_date']} → {fo['last_invoice_date']}")
    early = risk[risk["InvoiceDate"] < "2024-02-15"].copy()
    early["Freight % of invoice"] = early["Freight"] / early["Dollars"] * 100
    early["Status"] = early["Flag_FreightAnomaly"].map({1: "Overcharge", 0: "Normal"})
    fig = px.scatter(early, x="InvoiceDate", y="Freight % of invoice", color="Status",
                     color_discrete_map={"Overcharge": "#c62828", "Normal": "#1565c0"}, hover_name="VendorName",
                     hover_data={"Dollars": ":$,.0f", "Freight": ":$,.0f", "ExpectedFreight": ":$,.0f"})
    fig.add_hline(y=cfg.FREIGHT_PCT_UPPER, line_dash="dash", line_color="#ef6c00",
                  annotation_text="alert level (1.5x standard)")
    fig.update_layout(height=380, margin=dict(t=10))
    show(fig)
    st.caption(f"All overcharges are on invoices dated within one {fo['share_of_invoices_in_affected_window_pct']:.0f}%-"
               "affected window in early January 2024; none occur later in the year.")
    over = risk[risk["Flag_FreightAnomaly"] == 1].assign(Excess=lambda d: d["Freight"] - d["ExpectedFreight"])
    st.dataframe(
        over.nlargest(15, "Excess")[["PONumber", "VendorName", "InvoiceDate", "Dollars", "Freight",
                                     "ExpectedFreight", "FreightVsExpected", "Excess"]],
        hide_index=True, width="stretch",
        column_config={c_: st.column_config.NumberColumn(format="dollar")
                       for c_ in ("Dollars", "Freight", "ExpectedFreight", "Excess")})

    st.subheader("Estimate freight for an order")
    f1, f2 = st.columns(2)
    est_dollars = f1.number_input("Invoice dollars", min_value=1.0, value=25_000.0, step=1_000.0, key="est_dollars")
    est_qty = f2.number_input("Quantity (units)", min_value=1, value=2_000, step=100, key="est_qty")
    freight_art, _ = get_artifacts()
    est_df = pd.DataFrame([dict(
        VendorNumber=-1, InvoiceDate="2024-11-01", Quantity=est_qty, Dollars=est_dollars, Freight=1.0,
        LeadTimeDays=16, PaymentTermsDays=35, CatalogSKUs=0)])
    from ml_utils import freight_features
    est = float(freight_art["model"].predict(freight_features(
        est_df.assign(InvoiceDate=pd.to_datetime(est_df["InvoiceDate"])), freight_art["reference"]))[0])
    st.metric("Expected freight", f"${est:,.2f}", help="An invoice above ~1.5x this amount is flagged as an overcharge.")

# --------------------------------------------------------------------------- #
# 6. Invoice risk model
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Invoice risk: which invoices need manual approval?")
    st.markdown(
        "**Target:** the $250K approval policy recovered from the data **or** any anomaly flag (freight overcharge, "
        "unit cost, lead time, payment terms, repeat invoice). The classifier sees invoice values and how they "
        "compare with vendor norms, never the flags themselves.")
    st.warning("The labels are rule-derived, not confirmed fraud. Metrics show how well the policy can be learned and "
               "scored; they are not independent fraud detection.")
    lc, tm = am["label_counts_all_invoices"], am["test_metrics"]
    c = st.columns(5)
    c[0].metric("Invoices needing review", f"{lc['ManualApprovalRequired']:,}",
                f"{lc['ManualApprovalRequired'] / lc['total_invoices'] * 100:.1f}% of all", delta_color="off")
    c[1].metric("Model", am["selected_model"])
    c[2].metric("Test precision", f"{tm['precision']:.3f}")
    c[3].metric("Test recall", f"{tm['recall']:.3f}")
    c[4].metric("Test PR-AUC", f"{tm['PR_AUC']:.3f}")

    a, b = st.columns(2)
    with a:
        cm = tm["confusion"]
        fig = px.imshow([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], text_auto=True,
                        x=["No review", "Manual review"], y=["No review", "Manual review"],
                        color_continuous_scale="Blues", title=f"Hold-out confusion matrix (threshold {am['decision_threshold']:.2f})")
        fig.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=40),
                          xaxis_title="Model decision", yaxis_title="Rule-based label")
        show(fig)
    with b:
        imp = pd.Series(am["permutation_importance_PR_AUC"]).sort_values().tail(8)
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Top features (permutation importance)",
                     color_discrete_sequence=["#1565c0"])
        fig.update_layout(height=340, margin=dict(t=40), xaxis_title="Drop in PR-AUC", yaxis_title=None)
        show(fig)

    st.markdown("**How the model compares**")
    bm, iso, bel, tr = (am["benchmark_policy_rule_only_test"], am["isolation_forest_test"],
                        am["test_metrics_invoices_below_250K"], am["temporal_robustness"])
    cmp = pd.DataFrame([
        {"Approach": "$250K rule only (current control)", "Precision": bm["precision"], "Recall": bm["recall"]},
        {"Approach": f"{am['selected_model']} (hold-out)", "Precision": tm["precision"], "Recall": tm["recall"]},
        {"Approach": "Same model, invoices below $250K only", "Precision": bel["precision"], "Recall": bel["recall"]},
        {"Approach": "Same model, time-based split (train < Oct, test ≥ Oct)", "Precision": tr["precision"],
         "Recall": tr["recall"]},
        {"Approach": "Isolation Forest (unsupervised), precision@k", "Precision": iso["precision_at_k"], "Recall": iso["precision_at_k"]},
    ])
    st.dataframe(cmp, hide_index=True, width="stretch",
                 column_config={"Precision": st.column_config.NumberColumn(format="%.3f"),
                                "Recall": st.column_config.NumberColumn(format="%.3f")})
    st.caption("For the Isolation Forest, precision@k equals recall@k because k is the number of positives.")

    rec = pd.DataFrame(am["test_recall_by_flag"]).T.reset_index().rename(columns={"index": "Flag"})
    rec["Flag"] = rec["Flag"].map(FLAG_LABELS)
    st.markdown("**Recall by flag on the hold-out set** (rare flags have very few examples)")
    st.dataframe(rec, hide_index=True, width="stretch",
                 column_config={"recall": st.column_config.NumberColumn("Recall", format="%.2f")})

    st.subheader("Control gap")
    gap = risk[risk["ControlGap"] == 1]
    c = st.columns(3)
    c[0].metric("Anomalous invoices never manually approved", f"{len(gap)}")
    c[1].metric("Value of those invoices", money(gap["Dollars"].sum()))
    c[2].metric("Approvals triggered by", "Dollar amount only")
    st.caption("Every invoice of about $250.7K or more was manually approved (one approver) and none below $250K "
               "was, regardless of freight or unit-cost anomalies.")

    st.subheader("Invoice review queue")
    q1, q2, q3 = st.columns(3)
    mode = q1.selectbox("Show", ["Model flagged, not manually approved (control gap)", "All model-flagged invoices",
                                 "Every invoice"])
    min_p = q2.slider("Minimum risk probability", 0.0, 1.0, 0.0, 0.05)
    flag_sel = q3.multiselect("Rule flag", list(FLAG_LABELS.values()))
    qd = risk.copy()
    if mode.startswith("Model flagged, not"):
        qd = qd[(qd["ModelFlagged"] == 1) & (qd["HistoricalManualApproval"] == 0)]
    elif mode.startswith("All model"):
        qd = qd[qd["ModelFlagged"] == 1]
    qd = qd[qd["RiskProbability"] >= min_p]
    for lbl in flag_sel:
        col = next(k for k, v in FLAG_LABELS.items() if v == lbl)
        qd = qd[qd[col] == 1]
    st.caption(f"{len(qd):,} invoices")
    st.dataframe(
        qd.sort_values("RiskProbability", ascending=False)[
            ["PONumber", "VendorName", "InvoiceDate", "Dollars", "Freight", "FreightVsExpected", "RiskProbability",
             "ModelFlagged", "HistoricalManualApproval", "RiskReasons"]].head(500),
        hide_index=True, width="stretch",
        column_config={
            "Dollars": st.column_config.NumberColumn(format="dollar"),
            "Freight": st.column_config.NumberColumn(format="dollar"),
            "FreightVsExpected": st.column_config.NumberColumn("Freight / expected", format="%.2f"),
            "RiskProbability": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
        })
    st.download_button("Download scored invoices (CSV)", risk.to_csv(index=False).encode(),
                       "export_output_invoice_risk.csv", "text/csv")

    st.subheader("Check a new invoice")
    freight_art, approval_art = get_artifacts()
    vend = summary.sort_values("PurchaseDollars", ascending=False)
    vend = vend[vend["HasPurchases"]]
    v1, v2, v3 = st.columns(3)
    vname = v1.selectbox("Vendor", vend["VendorName"].tolist(), key="chk_vendor")
    vrow = vend[vend["VendorName"] == vname].iloc[0]
    n_dollars = v2.number_input("Invoice dollars", min_value=1.0, value=12_000.0, step=500.0, key="chk_dollars")
    n_qty = v3.number_input("Quantity", min_value=1, value=1_000, step=50, key="chk_qty")
    w1, w2, w3 = st.columns(3)
    n_freight = w1.number_input("Freight $", min_value=0.0, value=round(n_dollars * 0.005, 2), step=10.0,
                                key=f"chk_freight_{int(n_dollars)}")
    n_lead = w2.number_input("Lead time (days)", min_value=0, value=16, key="chk_lead")
    n_terms = w3.number_input("Payment terms (days)", min_value=0, value=35, key="chk_terms")
    new = pd.DataFrame([dict(
        VendorNumber=int(vrow["VendorNumber"]), InvoiceDate="2024-11-04", Quantity=n_qty, Dollars=n_dollars,
        Freight=n_freight, LeadTimeDays=n_lead, PaymentTermsDays=n_terms, CatalogSKUs=int(vrow["CatalogSKUs"]))])
    res = score_invoices(new, freight_art, approval_art).iloc[0]
    r1, r2, r3 = st.columns(3)
    r1.metric("Expected freight", f"${res['ExpectedFreight']:,.2f}", f"actual is {res['FreightVsExpected']:.1f}x expected",
              delta_color="off")
    r2.metric("Risk probability", f"{res['RiskProbability']:.1%}")
    r3.metric("Decision", "Manual approval" if res["ModelFlagged"] == 1 else "Auto-approve")
    if res["RiskReasons"]:
        st.error("Rule flags: " + res["RiskReasons"])
    elif res["ModelFlagged"] == 1:
        st.warning("No single rule fired, but the model scores this invoice above the review threshold.")
    else:
        st.success("No rule flags and the model score is below the review threshold.")

# --------------------------------------------------------------------------- #
# 7. Insights & statistics
# --------------------------------------------------------------------------- #
with tabs[6]:
    kf = analysis.key_findings()
    st.subheader("Key findings (all vendors)")
    t = kf["margin_hypothesis_test"]
    under, promo_kf = kf["underperformers"], kf["promotion_candidates"]
    ap = kf["approval_model"]
    st.markdown(f"""
- **Concentration:** the top 10 vendors account for **{kf['top10_share_pct']:.1f}%** of purchases (HHI = {kf['hhi']:,.0f}; below 1,500 is generally unconcentrated). The bottom {kf['bottom_half_vendors']} vendors supply only {kf['bottom_half_share_pct']:.2f}%, so the tail is a consolidation candidate.
- **Underperformers:** {under['vendors']} vendors in the low-margin / low-turnover quadrant hold **{money(under['ending_inventory_at_cost'])}** of ending inventory on {under['spend_pct']:.1f}% of spend.
- **Inventory:** stock at cost grew **{kf['inventory']['growth_pct']:.1f}%** over the year ({money(kf['inventory']['begin_value_at_cost'])} → {money(kf['inventory']['end_value_at_cost'])}).
- **Promotion candidates:** {promo_kf['skus']} high-margin SKUs with growing stock ({money(promo_kf['ending_inventory_at_cost'])} at cost).
- **Freight:** about {kf['freight_pct_large_orders_50k_plus']:.2f}% of value at every order size; {kf['freight_model']['freight_overcharge_summary']['invoices_flagged_freight_anomaly']} invoices were overcharged (about {money(kf['freight_model']['freight_overcharge_summary']['estimated_excess_freight_$'])} excess), all in early January 2024.
- **Approval control gap:** {ap['control_gap_invoices']} anomalous invoices ({money(ap['control_gap_dollars'])}) were never manually reviewed because approval depends only on invoice size.
- **Negative-margin catalog:** {', '.join(kf['negative_margin_vendors']) or 'none'} (weighted cost above shelf price).
""")
    st.subheader("Hypothesis test: do high-spend vendors earn a different margin?")
    res_t = cached_margin_test(summary)
    if res_t is None:
        st.info("Not enough vendors for the test.")
    else:
        st.markdown("H0: mean gross margin of **top-quartile spend** vendors equals that of **bottom-quartile spend** "
                    "vendors (Welch two-sample t-test, vendor level, reliable vendors only).")
        m = st.columns(5)
        m[0].metric("Top-quartile mean", f"{res_t['mean_margin_high_spend']:.2f}%", f"n={res_t['n_high_spend']}", delta_color="off")
        m[1].metric("Bottom-quartile mean", f"{res_t['mean_margin_low_spend']:.2f}%", f"n={res_t['n_low_spend']}", delta_color="off")
        m[2].metric("Difference", f"{res_t['difference_pp']:.2f} pp")
        m[3].metric("p-value", f"{res_t['p_value']:.4f}")
        m[4].metric("95% CI (pp)", f"[{res_t['ci95_low']:.2f}, {res_t['ci95_high']:.2f}]")
        if res_t["significant_at_5pct"]:
            st.success("Reject H0 at 5%: the margin difference is statistically significant.")
        else:
            st.info("Fail to reject H0 at 5%: no statistically significant margin difference between the groups.")

    st.subheader("Bulk-order effect on unit cost (within vendor)")
    bo = cached_bulk(purchases)
    fig = px.bar(bo, x="OrderSizeQuartile", y="median", color_discrete_sequence=["#2e7d32"],
                 hover_data={"mean": ":.3f", "count": True},
                 labels={"median": "Median unit-cost index (1.00 = vendor average)"})
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_yaxes(range=[0.9, 1.1])
    fig.update_layout(height=340, margin=dict(t=10), xaxis_title="Order-size quartile (within vendor)")
    show(fig)
    q1_, q4_ = bo.iloc[0], bo.iloc[-1]
    st.caption(
        ("Large orders show a lower median unit cost than small orders, consistent with volume discounts."
         if q4_["median"] < 0.97 * q1_["median"] else
         "Median unit cost is essentially the same across order sizes, so there is no evidence of volume discounts. "
         f"Small orders have a higher mean index ({q1_['mean']:.2f} vs {q4_['mean']:.2f}), driven by a few high-cost invoices.")
        + " Indicative only: invoices are not broken out by SKU, so product mix also moves unit cost.")

# --------------------------------------------------------------------------- #
# 8. Methodology
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Important limitations")
    st.warning(
        "No sales file or PO line items were provided. **Units sold, revenue and gross profit are estimates** "
        "derived from the inventory balance equation (begin inventory + purchased units - end inventory) and "
        "inventory-weighted prices.")
    st.subheader("Assumptions")
    for a_ in quality["assumptions"]:
        st.markdown(f"- {a_}")
    st.subheader("KPI definitions")
    st.markdown("""
| KPI | Definition |
|---|---|
| Unit cost | Invoice dollars ÷ invoice quantity |
| Lead time | Invoice date − PO date (days) |
| Payment terms | Pay date − invoice date (days) |
| Freight % | Freight ÷ invoice dollars |
| Est. units sold | Begin inventory + units purchased − end inventory |
| Sell-through | Est. units sold ÷ (begin inventory + units purchased) |
| Stock turnover | Est. units sold ÷ average of begin/end inventory units |
| Days of inventory | Period days ÷ stock turnover |
| Gross margin % | (weighted shelf price − weighted catalog cost) ÷ weighted shelf price |
| Vendor score | Weighted percentile rank across margin, sell-through, turnover, lead time, lead-time consistency, freight %, payment terms |
| Segment | Margin × turnover quadrant versus median of eligible vendors |
| Expected freight | Output of the freight model (0.5% of invoice dollars) |
| Risk probability | Manual-approval classifier's probability that an invoice needs review |
""")
    st.subheader("Risk rules (definition of misbehaviour)")
    st.markdown("""
- **High value:** invoice ≥ $250K (the approval policy observed in the data)
- **Freight overcharge:** freight above 0.75% of invoice dollars (1.5× the standard 0.5% rate)
- **Unit-cost outlier:** unit cost ≥ 2× or ≤ 0.5× the vendor's median
- **Lead-time / payment-terms outlier:** |robust z-score| > 4 versus the vendor's norm
- **Possible duplicate:** same vendor, dollars and quantity within 14 days
""")
    st.subheader("Data quality report")
    st.json(quality, expanded=False)
    st.download_button("Download export_output_purchases.csv", purchases.to_csv(index=False).encode(),
                       "export_output_purchases.csv", "text/csv")
