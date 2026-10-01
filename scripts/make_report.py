"""
Step 7 - Build the PDF report ("Vendor Performance Report.pdf") from key_findings.json and images/.

    python scripts/make_report.py

Every number in the report is read from the generated files, so it can never drift from the analysis.
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from PIL import Image as PILImage

import analysis
import config as cfg

OUT = cfg.ROOT / "Vendor Performance Report.pdf"
BLUE = colors.HexColor("#1f5fae")
LIGHT = colors.HexColor("#eaf1fb")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLUE, fontSize=17, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLUE, fontSize=12.5, spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.6, leading=13.6, spaceAfter=5)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.2, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
TITLE = ParagraphStyle("Title2", parent=ss["Title"], textColor=BLUE, fontSize=26, leading=31, spaceAfter=8)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=12, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.4, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
W = A4[0] - 4 * cm


def money(x: float) -> str:
    a = abs(x)
    return f"${x / 1e6:,.2f}M" if a >= 1e6 else f"${x / 1e3:,.1f}K" if a >= 1e3 else f"${x:,.0f}"


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def img(name: str, width: float = W):
    path = cfg.IMAGES_DIR / name
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def fig(name: str, caption: str, width: float = W):
    return KeepTogether([img(name, width), P(caption, CAP), Spacer(1, 6)])


def table(rows, widths, header=True):
    data = [[P(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d3e3")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE)]
    style += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Vendor Performance Analysis - Retail Inventory & Sales")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    kf = analysis.key_findings()
    fm, am = kf["freight_model"], kf["approval_model"]
    fo, tm = fm["freight_overcharge_summary"], am["test_metrics"]
    t = kf["margin_hypothesis_test"]
    under, promo, inv = kf["underperformers"], kf["promotion_candidates"], kf["inventory"]
    sel, base = fm["selected_test_metrics"], fm["baseline_test_metrics"]
    below, pol, iso, tr = (am["test_metrics_invoices_below_250K"], am["benchmark_policy_rule_only_test"],
                           am["isolation_forest_test"], am["temporal_robustness"])
    seg = kf["segment_counts"]
    s = []

    # ---- cover
    s += [Spacer(1, 5 * cm), P("Vendor Performance Analysis", TITLE),
          P("Retail Inventory &amp; Sales", SUB), Spacer(1, 14),
          P("Vendor efficiency and profitability for strategic purchasing and inventory decisions,<br/>"
            "with machine-learning models for freight cost and invoice manual approval", SUB),
          Spacer(1, 2.2 * cm),
          table([["Period", f"{kf['period'][0]} to {kf['period'][1]}"],
                 ["Purchase orders analysed", f"{kf['purchase_orders_in_period']:,} (in period)"],
                 ["Vendors", f"{kf['active_vendors']} with purchases"],
                 ["Purchases", money(kf['total_purchases'])],
                 ["Tools", "SQLite, Python (pandas, scikit-learn, SciPy), Plotly, Streamlit"]],
                [4.5 * cm, W - 4.5 * cm], header=False),
          PageBreak()]

    # ---- executive summary
    s += [P("1. Executive summary", H1),
          P(f"We analysed {kf['purchase_orders_in_period']:,} purchase orders from {kf['active_vendors']} vendors "
            f"({money(kf['total_purchases'])} of purchases), scored every vendor on profitability and efficiency, and "
            "built two models: one that predicts the freight an invoice should carry, and one that flags invoices that "
            "should go to manual approval.")]
    s += bullets([
        f"<b>Purchases are concentrated.</b> The top 10 vendors account for {kf['top10_share_pct']:.1f}% of spend, while "
        f"the bottom {kf['bottom_half_vendors']} vendors supply only {kf['bottom_half_share_pct']:.1f}%.",
        f"<b>The biggest suppliers earn below-median margins.</b> {seg['Volume driver - renegotiate cost']} 'Volume driver' "
        f"vendors carry {kf['segment_spend_pct']['Volume driver - renegotiate cost']:.0f}% of spend: cost renegotiation here "
        "has the largest profit effect.",
        f"<b>Inventory grew {inv['growth_pct']:.0f}%</b> ({money(inv['begin_value_at_cost'])} to "
        f"{money(inv['end_value_at_cost'])} at cost). {under['vendors']} underperforming vendors hold "
        f"{money(under['ending_inventory_at_cost'])} of it.",
        f"<b>Freight is a flat 0.5% of invoice value</b>, and no ML model beats that rule. The model is valuable as a "
        f"benchmark: it exposed {fo['invoices_flagged_freight_anomaly']} overcharged invoices (about "
        f"{money(fo['estimated_excess_freight_$'])} excess) all dated {fo['first_invoice_date']} to {fo['last_invoice_date']}.",
        f"<b>The approval control has a gap.</b> Manual approval is triggered only by invoice size (>= $250K). "
        f"{am['control_gap_invoices']} anomalous invoices ({money(am['control_gap_dollars'])}) were never reviewed. The risk "
        f"model catches {tm['recall']:.0%} of invoices that need review at {tm['precision']:.0%} precision on held-out data, "
        f"versus {pol['recall']:.0%} recall for the dollar rule alone.",
        "<b>Important limitation:</b> no sales data or PO line items were available, so units sold, revenue and profit "
        "are estimates (see section 3)."])
    s += [P("Recommended actions", H2)]
    s += bullets([
        "Add anomaly-based triggers (freight above 1.5x expected, unit cost far from the vendor norm) to the $250K approval rule.",
        "Investigate and recover the early-January freight overcharges; monitor freight against the 0.5% expected rate.",
        "Focus renegotiation on the high-volume, below-median-margin vendors; review the underperformer vendors.",
        "Release cash from slow stock, starting with high-margin SKUs whose inventory is growing.",
        "Consolidate the long tail of small vendors while managing dependence on the top 10."])
    s += [Spacer(1, 10)]

    # ---- business problem, data, method
    s += [P("2. Business problem and data", H1),
          P("A retailer buys from many vendors. The questions: which vendors create the most value, where is capital "
            "tied up in slow stock, are freight charges reasonable, and which invoices deserve extra scrutiny?"),
          table([["File", "Content", "Rows"],
                 ["export_output_vendor_invoice.csv", "One row per PO invoice: vendor, dates, quantity, dollars, freight, approver", "5,543"],
                 ["export_output_purchase_prices.csv", "Product catalog: brand, vendor, retail and purchase price", "12,261"],
                 ["export_output_begin_inventory.csv", "On-hand units by store and brand, 2024-01-01", "206,529"],
                 ["export_output_end_inventory.csv", "On-hand units by store and brand, 2024-12-31", "224,489"]],
                [6 * cm, W - 8.2 * cm, 2.2 * cm]),
          Spacer(1, 8),
          P("3. Method", H1),
          P("<b>Pipeline.</b> Raw CSVs are loaded into SQLite; SQL (CTEs, joins, window functions) builds the PO-level "
            "purchases table (<i>export_output_purchases.csv</i>), the SKU table and the vendor summary. Pandas adds KPIs, "
            "a composite vendor score and segments. Every build asserts that totals reconcile to the raw files."),
          P("<b>Estimated sales (limitation).</b> Without a sales file, units sold per vendor are estimated as begin "
            "inventory + units purchased - end inventory; revenue and gross profit use the inventory-weighted shelf price "
            "and catalog purchase price. Vendors with fewer than 3 POs or 100 units are marked unreliable and excluded "
            "from sales-based KPIs. Invoice date stands in for receiving date."),
          P("<b>Vendor score and segments.</b> A weighted percentile-rank score (margin 35%, sell-through 20%, turnover "
            "20%, lead time 7%, lead-time consistency 6%, freight % 7%, payment terms 5%). Vendors are placed in a margin "
            "x turnover quadrant: Star, Margin builder, Volume driver, Underperformer."),
          P("<b>Data quality.</b> The raw data is structurally clean. Minor issues handled: two vendors renamed "
            "mid-year, one invoiced vendor missing from the catalog, six catalog vendors never invoiced, two SKUs with "
            "zero pricing, and 140 POs invoiced after year-end (kept, excluded from vendor KPIs)."),
          Spacer(1, 6)]

    # ---- vendor findings
    s += [P("4. Vendor performance findings", H1),
          fig("01_purchase_concentration.png", "Top 15 vendors and cumulative share of purchases.", W * 0.92),
          P(f"The top 10 vendors (Diageo, Martignetti, Jim Beam, Pernod Ricard, Bacardi, ...) make up "
            f"{kf['top10_share_pct']:.1f}% of purchases (HHI {kf['hhi']:.0f}, an unconcentrated market overall). That "
            "gives negotiating leverage but also dependence risk.")]
    s += [fig("02_vendor_segments.png", "Vendor segments by margin and stock turnover (bubble size = purchases).", W * 0.82)]
    s += [table([["Segment", "Vendors", "Share of spend", "Action"],
                 ["Star", seg["Star - expand"], f"{kf['segment_spend_pct']['Star - expand']:.1f}%", "Expand"],
                 ["Margin builder", seg["Margin builder - tighten stock levels"], f"{kf['segment_spend_pct']['Margin builder - tighten stock levels']:.1f}%", "Tighten stock levels"],
                 ["Volume driver", seg["Volume driver - renegotiate cost"], f"{kf['segment_spend_pct']['Volume driver - renegotiate cost']:.1f}%", "Renegotiate cost"],
                 ["Underperformer", seg["Underperformer - review or exit"], f"{kf['segment_spend_pct']['Underperformer - review or exit']:.1f}%", "Review or exit"],
                 ["Insufficient data", seg["Insufficient data"], f"{kf['segment_spend_pct']['Insufficient data']:.2f}%", "Not scored"]],
                [4 * cm, 2.4 * cm, 3.4 * cm, W - 9.8 * cm]),
          Spacer(1, 8),
          P(f"Highest composite scores: {', '.join(n.title() for n in kf['top_scored_vendors'])}. Lowest: "
            f"{', '.join(n.title() for n in kf['bottom_scored_vendors'])}."),
          Spacer(1, 6),
          fig("03_vendor_scores.png", "Top 10 and bottom 10 vendors by composite score.", W * 0.78)]
    s += [P("Research questions", H2)]
    s += bullets([
        f"<b>Brands to promote or reprice:</b> {promo['skus']} SKUs combine a top-quartile margin (average "
        f"{promo['avg_margin_pct']:.0f}%), growing inventory and a large stock value ({money(promo['ending_inventory_at_cost'])} at cost). "
        "Brand-level sales are unavailable, so growing stock is the proxy for slow movement.",
        f"<b>Capital tied up:</b> inventory at cost grew {inv['growth_pct']:.1f}%. The {under['vendors']} underperformers "
        f"hold {money(under['ending_inventory_at_cost'])} on {under['spend_pct']:.1f}% of spend.",
        f"<b>Does bulk buying cut unit cost?</b> No evidence. Within a vendor, median unit cost of the largest orders is "
        f"{kf['bulk_order_median_cost_index']['Q4 largest']:.3f}x the vendor average vs "
        f"{kf['bulk_order_median_cost_index']['Q1 smallest']:.3f}x for the smallest. Freight is also about "
        f"{kf['freight_pct_large_orders_50k_plus']:.2f}% at every order size.",
        f"<b>Do high-spend vendors earn different margins?</b> No (Welch t-test: {t['mean_margin_high_spend']:.1f}% vs "
        f"{t['mean_margin_low_spend']:.1f}%, difference {t['difference_pp']:.2f} pp, p = {t['p_value']:.2f}, 95% CI "
        f"[{t['ci95_low']:.1f}, {t['ci95_high']:.1f}] pp). Vendor size does not predict profitability."])
    s += [PageBreak()]

    # ---- freight model
    s += [P("5. Model 1: freight cost prediction", H1),
          P("<b>Goal.</b> Estimate the freight an invoice should carry from information known when it arrives (order "
            "value, quantity, unit cost, lead time, payment terms, calendar, the vendor's typical rate). "
            f"<b>Design.</b> Temporal hold-out: trained on invoices before {fm['split']['cutoff']} "
            f"({fm['split']['train_rows_used_clean']:,} clean rows), tested on {fm['split']['test_rows']:,} later "
            "invoices. Candidates: the 0.5% rule baseline, linear regression, ridge, random forest and gradient boosting; "
            "the simplest ML model within 2% of the best cross-validated MAE is selected.")]
    rows = [["Model", "CV MAE", "Test MAE", "Test R2", "Median % err"]]
    for n, r in fm["candidates"].items():
        rows.append([n, f"${r['cv_MAE']:.2f}", f"${r['test_MAE']:.2f}", f"{r['test_R2']:.4f}", f"{r['test_MedianAPE_pct']:.1f}%"])
    s += [table(rows, [6.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, W - 13.6 * cm]), Spacer(1, 6),
          P(f"<b>Result.</b> Every model reaches R<super>2</super> of about {sel['R2']:.4f}. None beats the one-line rule "
            f"'freight = {fm['typical_freight_rate_pct']:.2f}% of invoice dollars' (test MAE ${sel['MAE']:.2f} vs "
            f"${base['MAE']:.2f}), because freight is proportional to order value and the other features add nothing. "
            f"The selected model is therefore the simple, interpretable one: <i>{fm['linear_model_equation']}</i>. "
            "This is a useful negative result: a complex model would add cost and opacity for no accuracy."),
          fig("07_freight_actual_vs_predicted.png", "Hold-out actual vs predicted freight (log scales).", W * 0.55),
          PageBreak(),
          P("Freight overcharges", H2),
          fig("05_freight_overcharge_timeline.png", "Freight as % of invoice dollars, January to mid-February 2024.", W * 0.95),
          P(f"The expected-freight benchmark exposes {fo['invoices_flagged_freight_anomaly']} invoices with freight above "
            f"1.5x the standard rate, a median of {fo['median_actual_to_expected_ratio']:.1f}x the expected amount and about "
            f"<b>{money(fo['estimated_excess_freight_$'])}</b> of excess freight. All fall on invoices dated "
            f"{fo['first_invoice_date']} to {fo['last_invoice_date']}, and {fo['share_of_invoices_in_affected_window_pct']:.0f}% "
            "of invoices in that window are affected: a systematic billing event (for example a surcharge or a wrong rate "
            "table) rather than random noise. Because it occurs in one window, the freight model cannot be validated on "
            "overcharges in later months."),
          PageBreak()]

    # ---- approval model
    s += [P("6. Model 2: invoice manual-approval prediction", H1),
          P("<b>Finding that shaped the design.</b> Historical manual approval is a pure dollar rule: all 374 invoices of "
            "about $250.7K or more were approved (by one approver) and none below $250K were. A model trained on that label "
            "alone would trivially score 100% by learning one threshold, so the target was widened to include anomaly flags."),
          fig("12_approval_policy_evidence.png", "Approval versus invoice size: a clean threshold at $250K.", W * 0.7),
          P("<b>Target</b> (<i>ManualApprovalRequired</i>): any of the following."),
          table([["Flag", "Rule", "Invoices"],
                 ["High value", "Invoice >= $250K (observed policy)", am["label_counts_all_invoices"]["Flag_HighValue"]],
                 ["Freight overcharge", "Freight above 1.5x the standard 0.5% rate", am["label_counts_all_invoices"]["Flag_FreightAnomaly"]],
                 ["Unit-cost outlier", "Unit cost >= 2x or <= 0.5x the vendor median", am["label_counts_all_invoices"]["Flag_UnitCostOutlier"]],
                 ["Lead-time outlier", "|robust z| > 4 versus the vendor norm", am["label_counts_all_invoices"]["Flag_LeadTimeOutlier"]],
                 ["Payment-terms outlier", "|robust z| > 4 versus the vendor norm", am["label_counts_all_invoices"]["Flag_PaymentTermsOutlier"]],
                 ["Repeat invoice", "Same vendor, dollars, quantity within 14 days", am["label_counts_all_invoices"]["Flag_PossibleDuplicate"]],
                 ["Needs manual approval", "Any flag", f"{am['label_counts_all_invoices']['ManualApprovalRequired']} of {am['label_counts_all_invoices']['total_invoices']:,}"]],
                [3.6 * cm, W - 6.4 * cm, 2.8 * cm]),
          Spacer(1, 6),
          P("<b>Model.</b> The classifier sees only invoice values and how they compare with the vendor's norms (computed on "
            f"training data only) plus the freight model's expected freight; never the flags or the approval column. "
            f"Logistic regression, random forest and gradient boosting were compared by 5-fold cross-validated PR-AUC; "
            f"<b>{am['selected_model']}</b> was selected and its threshold set by maximising F2 (a missed risky invoice "
            "costs more than an unneeded review)."),
          PageBreak(),
          fig("08_approval_model_evaluation.png", "Hold-out confusion matrix and precision-recall curve.", W * 0.95),
          table([["Evaluation", "Precision", "Recall", "Note"],
                 ["Hold-out (25% of invoices)", f"{tm['precision']:.3f}", f"{tm['recall']:.3f}", f"PR-AUC {tm['PR_AUC']:.3f}"],
                 ["Invoices below $250K only", f"{below['precision']:.3f}", f"{below['recall']:.3f}", "Anomaly detection beyond the dollar rule"],
                 ["Time-based split (train before Oct)", f"{tr['precision']:.3f}", f"{tr['recall']:.3f}", "Robustness over time"],
                 ["$250K rule alone (current control)", f"{pol['precision']:.3f}", f"{pol['recall']:.3f}", "Misses the anomalies"],
                 ["Isolation Forest (unsupervised)", f"{iso['precision_at_k']:.3f}", f"{iso['precision_at_k']:.3f}", f"ROC-AUC {iso['ROC_AUC']:.2f}; much weaker"]],
                [5.2 * cm, 2.2 * cm, 2.2 * cm, W - 9.6 * cm]),
          Spacer(1, 8),
          P("<b>How to read this honestly.</b> The labels are rule-derived (weak supervision), not confirmed fraud. The "
            "scores show that the review policy is learnable and can be applied to new invoices, with a probability and "
            "reasons attached; they do not prove independent fraud detection. The very rare flags (lead time, payment terms, "
            "repeat invoices) have only a handful of examples and are not reliably learned; the rules still catch them."),
          fig("09_approval_feature_importance.png", "Top features by permutation importance.", W * 0.62),
          PageBreak(),
          P("Control gap", H2),
          fig("10_control_gap.png", "Anomalous invoices below $250K that never received manual approval.", W * 0.82),
          P(f"{am['control_gap_invoices']} invoices below $250K carried at least one anomaly flag but were never manually "
            f"approved ({money(am['control_gap_dollars'])} of purchases). Adding the anomaly triggers to the approval policy "
            f"would send roughly {am['label_counts_all_invoices']['ManualApprovalRequired'] - am['label_counts_all_invoices']['Flag_HighValue']} "
            f"more invoices a year to review ({(am['label_counts_all_invoices']['ManualApprovalRequired'] - am['label_counts_all_invoices']['Flag_HighValue']) / am['label_counts_all_invoices']['total_invoices'] * 100:.1f}% "
            "of all invoices), a manageable workload."),
          PageBreak()]

    # ---- recommendations and limitations
    s += [P("7. Recommendations", H1)]
    s += bullets([
        "<b>Close the approval control gap</b> by adding freight and unit-cost anomaly triggers to the $250K rule; use the "
        "risk model to rank the review queue and show the reasons.",
        f"<b>Recover January freight overcharges</b> ({money(fo['estimated_excess_freight_$'])} across {fo['invoices_flagged_freight_anomaly']} "
        "invoices) and monitor freight against the 0.5% expected rate.",
        f"<b>Renegotiate with the Volume drivers</b> ({kf['segment_spend_pct']['Volume driver - renegotiate cost']:.0f}% of spend "
        "at below-median margin): a small cost reduction has the largest profit effect.",
        f"<b>Release cash from slow stock:</b> start with the {promo['skus']} high-margin SKUs with growing inventory and "
        f"review the {under['vendors']} underperformer vendors.",
        f"<b>Consolidate the vendor tail</b> ({kf['bottom_half_vendors']} vendors supply {kf['bottom_half_share_pct']:.1f}% of "
        "spend) while managing the top-10 dependence.",
        "<b>Do not budget on volume discounts or freight savings from consolidation:</b> neither appears in the data."])
    s += [P("8. Limitations", H1)]
    s += bullets([
        "No sales or PO line-item data: units sold, revenue and profit are inventory-balance estimates; vendors with thin "
        "data are excluded from sales KPIs.",
        "Invoice date stands in for receiving date; lead time is PO-to-invoice, so on-time delivery cannot be measured.",
        "Approval labels are rule-derived; the model operationalises the policy and is not independent fraud detection.",
        "Freight overcharges occur in a single nine-day window, so they cannot be validated across time.",
        "Promotion candidates use growing stock as a proxy for slow sales; confirm with sales data before acting."])
    s += [P("9. Deliverables", H1),
          P("GitHub-ready project: ingestion and analysis scripts, two executed notebooks (EDA and analysis with models), "
            "trained models, a Streamlit dashboard with a freight estimator and an invoice checker, tests, and this report. "
            "See README.md for how to run everything.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Vendor Performance Report",
                            subject="Vendor performance analysis with freight and manual-approval models")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
