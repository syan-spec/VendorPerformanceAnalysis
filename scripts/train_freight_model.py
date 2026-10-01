"""
Step 3 - Freight cost prediction model.

    python scripts/train_freight_model.py

Goal: estimate the freight an invoice *should* carry from information known when the invoice arrives
(order value, quantity, unit cost, lead time, payment terms, calendar, vendor's typical freight rate).
Invoices whose actual freight is far above this expectation are freight overcharges and feed the
manual-approval model (train_approval_model.py).

Design
  * Temporal hold-out: train on invoices before ``cfg.TEMPORAL_CUTOFF``, test on invoices on/after it.
  * The model learns *standard* freight, so training uses invoices without a freight anomaly
    (freight <= ``cfg.FREIGHT_PCT_UPPER`` % of invoice dollars).
  * A rule-of-thumb baseline (median freight rate x dollars) is always reported as the benchmark.
    Among the ML candidates, the simplest model whose cross-validated MAE is within 2% of the best ML
    model is selected, so complexity must earn its place.

Outputs: models/freight_model.joblib, models/freight_model_metrics.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

import config as cfg
from ml_utils import (
    DollarRegressor, FREIGHT_FEATURES, MedianRateBaseline, RateRegressor,
    fit_freight_reference, freight_features, load_invoices, regression_metrics,
)

log = cfg.get_logger("train_freight_model", "train_freight_model.log")
SELECTION_TOLERANCE = 1.02


def candidate_models() -> dict:
    """Ordered from simplest to most complex."""
    rs = cfg.RANDOM_STATE
    return {
        "Baseline (median rate x dollars)": MedianRateBaseline(),
        "Linear regression (dollars)": DollarRegressor(LinearRegression(fit_intercept=False), ["Dollars"]),
        "Ridge (dollars, quantity)": DollarRegressor(Ridge(alpha=1.0, fit_intercept=False), ["Dollars", "Quantity"]),
        "Random forest (rate)": RateRegressor(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=rs),
            FREIGHT_FEATURES,
        ),
        "Gradient boosting (rate)": RateRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=300, random_state=rs
            ),
            FREIGHT_FEATURES,
        ),
    }


def cross_validated_mae(model, X: pd.DataFrame, y: pd.Series) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return float(np.mean(np.abs(oof - y.to_numpy())))


def linear_equation(model) -> str | None:
    """Human-readable equation when the selected model is a plain linear regression."""
    inner = getattr(model, "model_", None)
    if isinstance(inner, LinearRegression) and len(model.cols) == 1:
        return f"Freight = {inner.coef_[0]:.5f} x {model.cols[0]} (no intercept)"
    return None


def run() -> dict:
    df = load_invoices()
    cut = pd.Timestamp(cfg.TEMPORAL_CUTOFF)
    train, test = df[df["InvoiceDate"] < cut], df[df["InvoiceDate"] >= cut]
    train_clean = train[train["FreightPctOfDollars"] <= cfg.FREIGHT_PCT_UPPER]
    log.info("Invoices: train=%d (clean %d) test=%d, cutoff %s",
             len(train), len(train_clean), len(test), cfg.TEMPORAL_CUTOFF)

    # reference statistics come from training invoices only
    ref = fit_freight_reference(train)
    X_tr, y_tr = freight_features(train_clean, ref), train_clean["Freight"]
    X_te, y_te = freight_features(test, ref), test["Freight"]

    rows = {}
    for name, model in candidate_models().items():
        cv_mae = cross_validated_mae(model, X_tr, y_tr)
        fitted = clone(model).fit(X_tr, y_tr)
        rows[name] = {"cv_MAE": cv_mae, **{f"test_{k}": v for k, v in
                      regression_metrics(y_te, fitted.predict(X_te)).items()}}
        log.info("%-34s cvMAE %.2f | test MAE %.2f R2 %.4f medAPE %.2f%%", name, cv_mae,
                 rows[name]["test_MAE"], rows[name]["test_R2"], rows[name]["test_MedianAPE_pct"])

    baseline_name = "Baseline (median rate x dollars)"
    ml_rows = {n: r for n, r in rows.items() if n != baseline_name}
    best = min(r["cv_MAE"] for r in ml_rows.values())
    chosen = next(n for n, r in ml_rows.items() if r["cv_MAE"] <= best * SELECTION_TOLERANCE)
    log.info("Selected ML model (simplest within %.0f%% of best ML CV MAE): %s",
             (SELECTION_TOLERANCE - 1) * 100, chosen)
    base_mae, chosen_mae = rows[baseline_name]["test_MAE"], rows[chosen]["test_MAE"]
    lift = (base_mae - chosen_mae) / base_mae * 100
    log.info("Test MAE: ML model %.2f vs rule baseline %.2f (lift %+.1f%%)", chosen_mae, base_mae, lift)

    final = clone(candidate_models()[chosen]).fit(X_tr, y_tr)

    # permutation importance on the hold-out set (skipped for the feature-free baseline)
    importance = {}
    if True:
        pi = permutation_importance(final, X_te, y_te, scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=cfg.RANDOM_STATE)
        importance = dict(sorted(zip(FREIGHT_FEATURES, pi.importances_mean.round(4).tolist()),
                                 key=lambda kv: -kv[1]))

    # expected freight for every invoice -> overcharge summary
    X_all = freight_features(df, ref)
    expected = final.predict(X_all)
    over = df["FreightPctOfDollars"] > cfg.FREIGHT_PCT_UPPER
    excess = float((df.loc[over, "Freight"] - expected[over.to_numpy()]).sum())
    anomaly_summary = {
        "invoices_flagged_freight_anomaly": int(over.sum()),
        "share_of_all_invoices_pct": round(float(over.mean() * 100), 2),
        "first_invoice_date": str(df.loc[over, "InvoiceDate"].min().date()),
        "last_invoice_date": str(df.loc[over, "InvoiceDate"].max().date()),
        "months_affected": sorted(df.loc[over, "InvoiceMonth"].unique().tolist()),
        "actual_freight_on_flagged_$": round(float(df.loc[over, "Freight"].sum()), 2),
        "expected_freight_on_flagged_$": round(float(expected[over.to_numpy()].sum()), 2),
        "estimated_excess_freight_$": round(excess, 2),
        "median_actual_to_expected_ratio": round(
            float((df.loc[over, "Freight"] / expected[over.to_numpy()]).median()), 2),
        "share_of_invoices_in_affected_window_pct": round(float(
            over[(df["InvoiceDate"] >= df.loc[over, "InvoiceDate"].min())
                 & (df["InvoiceDate"] <= df.loc[over, "InvoiceDate"].max())].mean() * 100), 1),
    }

    metrics = {
        "split": {"cutoff": cfg.TEMPORAL_CUTOFF, "train_rows": int(len(train)),
                  "train_rows_used_clean": int(len(train_clean)), "test_rows": int(len(test)),
                  "test_freight_anomalies": int((test["FreightPctOfDollars"] > cfg.FREIGHT_PCT_UPPER).sum())},
        "candidates": rows,
        "selected_model": chosen,
        "selection_rule": "simplest ML model within 2% of the best ML 5-fold CV MAE (baseline excluded)",
        "ml_lift_vs_baseline_test_MAE_pct": round(lift, 2),
        "ml_beats_baseline": bool(lift > 2.0),
        "linear_model_equation": linear_equation(final),
        "selected_test_metrics": {k.replace("test_", ""): v for k, v in rows[chosen].items()
                                  if k.startswith("test_")},
        "baseline_test_metrics": {k.replace("test_", ""): v for k, v in
                                  rows["Baseline (median rate x dollars)"].items()
                                  if k.startswith("test_")},
        "permutation_importance_MAE": importance,
        "typical_freight_rate_pct": round(ref["global_rate_pct"], 4),
        "freight_overcharge_summary": anomaly_summary,
    }

    cfg.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": chosen, "reference": ref, "features": FREIGHT_FEATURES,
                 "cutoff": cfg.TEMPORAL_CUTOFF}, cfg.MODELS_DIR / "freight_model.joblib")
    (cfg.MODELS_DIR / cfg.OUT_FREIGHT_METRICS).write_text(json.dumps(metrics, indent=2))
    log.info("Saved models/freight_model.joblib and models/%s", cfg.OUT_FREIGHT_METRICS)
    return metrics


if __name__ == "__main__":
    run()
