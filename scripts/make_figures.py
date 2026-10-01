"""
Step 5 - Generate the static charts used in the README and the PDF report (saved to images/).

    python scripts/make_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

import analysis
import config as cfg

sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE, GREEN, RED, GREY = "#1f5fae", "#ef6c00", "#2e7d32", "#c62828", "#9e9e9e"
SEG_COLORS = {
    "Star - expand": GREEN, "Margin builder - tighten stock levels": BLUE,
    "Volume driver - renegotiate cost": ORANGE, "Underperformer - review or exit": RED,
    "Insufficient data": GREY,
}
log = cfg.get_logger("make_figures")


def save(fig, name: str) -> None:
    cfg.IMAGES_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.IMAGES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved images/%s", name)


def short(s: str, n: int = 24) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    purchases, summary, sku = analysis.load_tables()
    risk = pd.read_csv(cfg.DATA_DIR / cfg.OUT_INVOICE_RISK, parse_dates=["InvoiceDate"])
    fm = json.loads((cfg.MODELS_DIR / cfg.OUT_FREIGHT_METRICS).read_text())
    am = json.loads((cfg.MODELS_DIR / cfg.OUT_APPROVAL_METRICS).read_text())
    active = summary[summary["PurchaseOrders"] > 0].sort_values("PurchaseDollars", ascending=False)

    # 1. Pareto of purchases
    top = active.head(15).copy()
    top["cum"] = top["PurchaseDollars"].cumsum() / active["PurchaseDollars"].sum() * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(top)), top["PurchaseDollars"] / 1e6, color=BLUE)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([short(n, 18) for n in top["VendorName"]], rotation=60, ha="right")
    ax.set_ylabel("Purchases ($M)")
    ax2 = ax.twinx()
    ax2.plot(range(len(top)), top["cum"], color=ORANGE, marker="o")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative % of spend")
    ax2.grid(False)
    ax.set_title("Top 15 vendors: purchase concentration (top 10 = %.1f%% of spend)" %
                 (active.head(10)["PurchaseDollars"].sum() / active["PurchaseDollars"].sum() * 100))
    save(fig, "01_purchase_concentration.png")

    # 2. margin vs turnover quadrants
    el = summary[summary["ScoreEligible"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for seg, g in el.groupby("Segment"):
        ax.scatter(g["StockTurnover"], g["GrossMarginPct"], s=np.sqrt(g["PurchaseDollars"]) / 8 + 15,
                   alpha=0.65, color=SEG_COLORS[seg], label=seg, edgecolor="white")
    ax.axhline(el["GrossMarginPct"].median(), color=GREY, ls="--")
    ax.axvline(el["StockTurnover"].median(), color=GREY, ls="--")
    ax.set_xlabel("Stock turnover (x per year)")
    ax.set_ylabel("Estimated gross margin (%)")
    ax.set_title("Vendor segments: margin vs stock turnover (bubble = purchase $)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "02_vendor_segments.png")

    # 3. vendor score top/bottom
    sc = el.sort_values("VendorScore", ascending=False)
    pick = pd.concat([sc.head(10), sc.tail(10)]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([short(n, 30) for n in pick["VendorName"]], pick["VendorScore"],
            color=[SEG_COLORS[s] for s in pick["Segment"]])
    ax.set_xlabel("Composite vendor score (0-100)")
    ax.set_title("Top 10 and bottom 10 vendors by composite score")
    save(fig, "03_vendor_scores.png")

    # 4. freight vs dollars (log-log) with the standard 0.5% line
    fig, ax = plt.subplots(figsize=(9, 6))
    flag = risk["Flag_FreightAnomaly"] == 1
    ax.scatter(risk.loc[~flag, "Dollars"], risk.loc[~flag, "Freight"], s=8, alpha=0.35, color=BLUE, label="Normal")
    ax.scatter(risk.loc[flag, "Dollars"], risk.loc[flag, "Freight"], s=18, alpha=0.8, color=RED,
               label=f"Freight overcharge ({int(flag.sum())})")
    xs = np.logspace(np.log10(risk["Dollars"].min()), np.log10(risk["Dollars"].max()), 100)
    ax.plot(xs, xs * fm["typical_freight_rate_pct"] / 100, color="black", lw=1, label="Standard rate (~0.5%)")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("Invoice dollars (log)"), ax.set_ylabel("Freight $ (log)")
    ax.set_title("Freight is ~0.5% of invoice value, except for a cluster of overcharges")
    ax.legend()
    save(fig, "04_freight_vs_dollars.png")

    # 5. freight overcharge timeline
    daily = risk[risk["InvoiceDate"] < "2024-02-15"].copy()
    daily["rate"] = daily["Freight"] / daily["Dollars"] * 100
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(daily["InvoiceDate"], daily["rate"], c=np.where(daily["Flag_FreightAnomaly"] == 1, RED, BLUE),
               s=14, alpha=0.7)
    ax.axhline(cfg.FREIGHT_PCT_UPPER, color=ORANGE, ls="--", label=f"Alert level ({cfg.FREIGHT_PCT_UPPER}% of invoice $)")
    ax.set_ylabel("Freight as % of invoice $")
    ax.set_title("Freight overcharges are confined to invoices dated Jan 4-12, 2024 (Jan 1 - Feb 14 shown)")
    ax.legend()
    fig.autofmt_xdate()
    save(fig, "05_freight_overcharge_timeline.png")

    # 6. freight model comparison
    cand = pd.DataFrame(fm["candidates"]).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cols = [ORANGE if "Baseline" in n else BLUE for n in cand.index]
    axes[0].barh(cand.index, cand["test_MAE"], color=cols)
    axes[0].set_xlabel("Test MAE ($) - lower is better")
    axes[0].set_title("Hold-out MAE (Oct 2024 onwards)")
    axes[1].barh(cand.index, cand["test_MedianAPE_pct"], color=cols)
    axes[1].set_xlabel("Median absolute % error")
    axes[1].set_title("Median % error")
    axes[1].set_yticklabels([])
    save(fig, "06_freight_model_comparison.png")

    # 7. actual vs predicted (test period)
    te = risk[risk["InvoiceDate"] >= cfg.TEMPORAL_CUTOFF]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(te["ExpectedFreight"], te["Freight"], s=10, alpha=0.4, color=BLUE)
    lim = [te[["ExpectedFreight", "Freight"]].min().min(), te[["ExpectedFreight", "Freight"]].max().max()]
    ax.plot(lim, lim, color="black", lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("Predicted freight ($)"), ax.set_ylabel("Actual freight ($)")
    r2 = fm["selected_test_metrics"]["R2"]
    ax.set_title(f"{fm['selected_model']}: actual vs predicted (R2 = {r2:.4f})")
    save(fig, "07_freight_actual_vs_predicted.png")

    # 8. approval model: confusion matrix + PR curve (held-out rows)
    t = risk[risk["Split"] == "test"]
    cm = am["test_metrics"]["confusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["No review", "Manual review"], yticklabels=["No review", "Manual review"])
    axes[0].set_xlabel("Model decision"), axes[0].set_ylabel("Rule-based label")
    axes[0].set_title(f"Hold-out confusion matrix (threshold {am['decision_threshold']:.2f})")
    p, r, _ = precision_recall_curve(t["ManualApprovalRequired"], t["RiskProbability"])
    axes[1].plot(r, p, color=BLUE)
    axes[1].set_xlabel("Recall"), axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall curve (PR-AUC {am['test_metrics']['PR_AUC']:.3f})")
    axes[1].set_ylim(0, 1.02)
    save(fig, "08_approval_model_evaluation.png")

    # 9. permutation importance
    imp = pd.Series(am["permutation_importance_PR_AUC"]).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Manual-approval model: top features (permutation importance)")
    save(fig, "09_approval_feature_importance.png")

    # 10. control gap
    gap = risk[risk["ControlGap"] == 1]
    reasons = gap["RiskReasons"].str.replace("Freight above 1.5x the standard rate", "Freight overcharge").value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(reasons.index[::-1], reasons.values[::-1], color=RED)
    ax.set_xlabel("Invoices")
    ax.set_title(f"Control gap: {len(gap)} anomalous invoices (${gap['Dollars'].sum() / 1e6:.2f}M) never manually approved")
    save(fig, "10_control_gap.png")

    # 11. monthly purchases
    m = purchases[purchases["InPeriod"]].groupby("InvoiceMonth")["Dollars"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(m.index, m.values, color=BLUE)
    ax.set_ylabel("Purchases ($M)")
    ax.set_title("Monthly purchases (2024)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "11_monthly_purchases.png")

    # 12. approval policy evidence
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(risk["Dollars"], risk["HistoricalManualApproval"] + np.random.default_rng(0).uniform(-0.04, 0.04, len(risk)),
               s=8, alpha=0.4, color=BLUE)
    ax.axvline(cfg.HIGH_VALUE_THRESHOLD, color=RED, ls="--", label="$250K")
    ax.set_xscale("log")
    ax.set_yticks([0, 1]), ax.set_yticklabels(["Not approved", "Manually approved"])
    ax.set_xlabel("Invoice dollars (log)")
    ax.set_title("Historical manual approval is a pure dollar rule: every invoice >= $250K, none below")
    ax.legend()
    save(fig, "12_approval_policy_evidence.png")


if __name__ == "__main__":
    main()
