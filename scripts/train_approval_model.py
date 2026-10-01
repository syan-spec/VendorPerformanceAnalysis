"""
Step 4 - Invoice manual-approval (risk) model.

    python scripts/train_approval_model.py      # run train_freight_model.py first

Target (``ManualApprovalRequired``) = the $250K approval policy recovered from the data OR any anomaly
flag from risk_rules.py (freight overcharge, unit-cost outlier, lead-time / payment-terms outlier,
repeat invoice). The classifier sees only invoice values and how they compare with the vendor's norms
(plus the freight model's expected freight) - never the flags or the approval column.

HONEST FRAMING: the labels are rule-derived (weak supervision), so high scores mean the policy is
learnable and can be scored on new invoices, not that fraud was independently detected. The model
is therefore also checked on the anomaly-only subset (invoices below $250K) and against an
unsupervised Isolation Forest.

Outputs: models/approval_model.joblib, models/approval_model_metrics.json,
         data/export_output_invoice_risk.csv
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
import risk_rules as rr
from ml_utils import APPROVAL_FEATURES, approval_features, freight_features, load_invoices

log = cfg.get_logger("train_approval_model", "train_approval_model.log")
SELECTION_TOLERANCE = 0.005     # PR-AUC points within which the simpler model wins
MODEL_FLAGS = [c for c in rr.FLAG_COLUMNS]


def candidate_models() -> dict:
    """Ordered from simplest to most complex."""
    rs = cfg.RANDOM_STATE
    return {
        "Logistic regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]),
        "Random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=rs),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=rs),
    }


def f_beta(p: float, r: float, beta: float) -> float:
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)


def best_threshold(y: np.ndarray, proba: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on out-of-fold probabilities (recall weighted: a missed risky
    invoice costs more than reviewing a clean one)."""
    p, r, t = precision_recall_curve(y, proba)
    scores = [f_beta(pi, ri, beta) for pi, ri in zip(p[:-1], r[:-1])]
    return float(t[int(np.argmax(scores))])


def classification_metrics(y, proba, thr: float) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": round(float(thr), 4), "rows": int(len(y)), "positives": int(y.sum()),
        "precision": prec, "recall": rec, "F1": f_beta(prec, rec, 1.0), "F2": f_beta(prec, rec, 2.0),
        "accuracy": float((tp + tn) / len(y)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    if 0 < y.sum() < len(y):
        out["ROC_AUC"] = float(roc_auc_score(y, proba))
        out["PR_AUC"] = float(average_precision_score(y, proba))
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def run() -> dict:
    df = load_invoices()
    fart = joblib.load(cfg.MODELS_DIR / "freight_model.joblib")
    expected = fart["model"].predict(freight_features(df, fart["reference"]))

    # ---- split: stratify on the two policy flags that do not depend on vendor statistics
    key = (df["Dollars"] >= cfg.HIGH_VALUE_THRESHOLD).astype(int) + 2 * (
        df["FreightPctOfDollars"] > cfg.FREIGHT_PCT_UPPER).astype(int)
    tr_idx, te_idx = train_test_split(
        df.index, test_size=cfg.TEST_SIZE, stratify=key, random_state=cfg.RANDOM_STATE)

    # ---- labels and features use vendor statistics from the TRAINING rows only
    vs = rr.fit_vendor_stats(df.loc[tr_idx])
    flagged = rr.add_risk_flags(df, vs)
    y = flagged["ManualApprovalRequired"].to_numpy()
    X = approval_features(df, expected, vs)
    X_tr, X_te, y_tr, y_te = X.loc[tr_idx], X.loc[te_idx], y[tr_idx], y[te_idx]
    log.info("Rows train=%d test=%d | positives train=%d (%.1f%%) test=%d (%.1f%%)", len(tr_idx),
             len(te_idx), y_tr.sum(), y_tr.mean() * 100, y_te.sum(), y_te.mean() * 100)

    # ---- model selection by 5-fold CV PR-AUC on the training rows
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    cv, oof = {}, {}
    for name, model in candidate_models().items():
        p = cross_val_predict(clone(model), X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        oof[name] = p
        cv[name] = {"cv_PR_AUC": float(average_precision_score(y_tr, p)),
                    "cv_ROC_AUC": float(roc_auc_score(y_tr, p))}
        log.info("%-20s CV PR-AUC %.4f ROC-AUC %.4f", name, cv[name]["cv_PR_AUC"], cv[name]["cv_ROC_AUC"])
    best = max(v["cv_PR_AUC"] for v in cv.values())
    chosen = next(n for n, v in cv.items() if v["cv_PR_AUC"] >= best - SELECTION_TOLERANCE)
    thr = best_threshold(y_tr, oof[chosen])
    log.info("Selected: %s | decision threshold (max F2 on OOF) = %.4f", chosen, thr)

    # ---- hold-out evaluation
    model = clone(candidate_models()[chosen]).fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]
    test_metrics = classification_metrics(y_te, proba_te, thr)
    log.info("TEST precision %.3f recall %.3f F1 %.3f PR-AUC %.4f", test_metrics["precision"],
             test_metrics["recall"], test_metrics["F1"], test_metrics.get("PR_AUC", float("nan")))

    ft = flagged.loc[te_idx]
    recall_by_flag = {}
    for c in MODEL_FLAGS:
        m = ft[c].to_numpy()
        recall_by_flag[c] = {"test_invoices": int(m.sum()),
                             "caught": int(((proba_te >= thr) & m).sum()),
                             "recall": float(((proba_te >= thr) & m).sum() / m.sum()) if m.sum() else None}
    below = ~ft["Flag_HighValue"].to_numpy()
    anomaly_only = classification_metrics(y_te[below], proba_te[below], thr)

    # ---- benchmarks
    hv_pred = ft["Flag_HighValue"].astype(int).to_numpy()
    policy_only = {"precision": float((hv_pred & y_te).sum() / max(hv_pred.sum(), 1)),
                   "recall": float((hv_pred & y_te).sum() / y_te.sum())}
    iso = IsolationForest(n_estimators=300, contamination=float(y_tr.mean()), random_state=cfg.RANDOM_STATE)
    iso.fit(X_tr)
    iso_te = -iso.decision_function(X_te)
    k = int(y_te.sum())
    top_k = np.argsort(-iso_te)[:k]
    iso_metrics = {
        "ROC_AUC": float(roc_auc_score(y_te, iso_te)), "PR_AUC": float(average_precision_score(y_te, iso_te)),
        "precision_at_k": float(y_te[top_k].mean()), "k": k,
        "ROC_AUC_anomaly_only_subset": float(roc_auc_score(y_te[below], iso_te[below])),
    }

    # ---- permutation importance (hold-out)
    pi = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=5,
                                random_state=cfg.RANDOM_STATE)
    importance = dict(sorted(zip(APPROVAL_FEATURES, pi.importances_mean.round(4).tolist()),
                             key=lambda kv: -kv[1]))

    # ---- temporal robustness: train before the cutoff, test after it
    cut = pd.Timestamp(cfg.TEMPORAL_CUTOFF)
    t_tr, t_te = df.index[df["InvoiceDate"] < cut], df.index[df["InvoiceDate"] >= cut]
    vs_t = rr.fit_vendor_stats(df.loc[t_tr])
    fl_t = rr.add_risk_flags(df, vs_t)
    y_t = fl_t["ManualApprovalRequired"].to_numpy()
    X_t = approval_features(df, expected, vs_t)
    m_t = clone(candidate_models()[chosen]).fit(X_t.loc[t_tr], y_t[t_tr])
    temporal = classification_metrics(y_t[t_te], m_t.predict_proba(X_t.loc[t_te])[:, 1], thr)
    temporal["note"] = (f"{int(fl_t.loc[t_te, 'Flag_FreightAnomaly'].sum())} freight anomalies fall in "
                        "the test period (all occur in January 2024)")

    # ---- score every invoice with the final model; export
    proba_all = model.predict_proba(X)[:, 1]
    iso_all = pd.Series(-iso.decision_function(X)).rank(pct=True).to_numpy()
    out = flagged[["PONumber", "VendorNumber", "VendorName", "InvoiceDate", "Quantity", "Dollars",
                   "Freight", "FreightPctOfDollars"]].copy()
    out["ExpectedFreight"] = expected.round(2)
    out["FreightVsExpected"] = (df["Freight"] / np.maximum(expected, 0.01)).round(2)
    for c in MODEL_FLAGS:
        out[c] = flagged[c].astype(int)
    out["RiskReasons"] = flagged["RiskReasons"]
    out["HistoricalManualApproval"] = flagged["HistoricalManualApproval"]
    out["ManualApprovalRequired"] = flagged["ManualApprovalRequired"]
    out["RiskProbability"] = proba_all.round(4)
    out["ModelFlagged"] = (proba_all >= thr).astype(int)
    out["IsolationAnomalyPct"] = iso_all.round(4)
    out["ControlGap"] = ((out["ManualApprovalRequired"] == 1) & (out["HistoricalManualApproval"] == 0)).astype(int)
    out["Split"] = np.where(df.index.isin(te_idx), "test", "train")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"]).dt.strftime("%Y-%m-%d")
    out.to_csv(cfg.DATA_DIR / cfg.OUT_INVOICE_RISK, index=False)

    metrics = {
        "label_definition": "ManualApprovalRequired = any of: " + "; ".join(rr.FLAG_COLUMNS.values()),
        "label_counts_all_invoices": {c: int(flagged[c].sum()) for c in MODEL_FLAGS}
        | {"ManualApprovalRequired": int(y.sum()), "total_invoices": int(len(df))},
        "policy_consistency": rr.policy_consistency(flagged),
        "split": {"train_rows": int(len(tr_idx)), "test_rows": int(len(te_idx)),
                  "train_positive_rate_pct": round(float(y_tr.mean() * 100), 2),
                  "test_positive_rate_pct": round(float(y_te.mean() * 100), 2)},
        "candidates_cv": cv, "selected_model": chosen, "decision_threshold": round(thr, 4),
        "threshold_rule": "maximise F2 on 5-fold out-of-fold training predictions",
        "test_metrics": test_metrics,
        "test_recall_by_flag": recall_by_flag,
        "test_metrics_invoices_below_250K": anomaly_only,
        "benchmark_policy_rule_only_test": policy_only,
        "isolation_forest_test": iso_metrics,
        "temporal_robustness": temporal,
        "permutation_importance_PR_AUC": importance,
        "features": APPROVAL_FEATURES,
        "control_gap_invoices": int(out["ControlGap"].sum()),
        "control_gap_dollars": round(float(out.loc[out["ControlGap"] == 1, "Dollars"].sum()), 2),
    }
    joblib.dump({"model": model, "name": chosen, "threshold": thr, "features": APPROVAL_FEATURES,
                 "vendor_stats": vs}, cfg.MODELS_DIR / "approval_model.joblib")
    (cfg.MODELS_DIR / cfg.OUT_APPROVAL_METRICS).write_text(json.dumps(metrics, indent=2, default=_jsonable))
    log.info("Saved model, metrics and %s (%d rows)", cfg.OUT_INVOICE_RISK, len(out))
    return metrics


if __name__ == "__main__":
    run()
