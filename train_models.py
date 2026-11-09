"""
Train and evaluate the failure classifiers (Section IV-B and V-A, Table III, Fig. 2).

Compares Logistic Regression (LR), Random Forest (RF), and histogram-based
Gradient Boosting (GB) on
  * raw sensor features, and
  * physics-informed features derived from the digital twin state:
        dT       = process - air temperature                    (Eq. 1)
        P        = torque * 2*pi*n/60                            (Eq. 2)
        S, M     = tool wear * torque, and margin to the limit   (Eq. 3)

Protocol: stratified 5-fold cross-validation; the F1-optimal decision threshold
is chosen inside each training fold with an inner 3-fold split, so test folds
are never used for tuning. Permutation importance is computed on a 30 % hold-out.

Usage:
    python src/train_models.py          # reads data/process_data.csv
Writes results/pm_results.json (runtime about 2-3 minutes).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def build_features(df):
    """Return (raw, physics-informed) feature matrices."""
    tcode = pd.get_dummies(df["type"], prefix="type").astype(float)
    raw = pd.concat([df[["air_K", "process_K", "speed_rpm", "torque_Nm", "tool_wear_min"]], tcode], axis=1)
    phys = raw.copy()
    omega = df["speed_rpm"].to_numpy() * 2 * np.pi / 60
    strain = df["tool_wear_min"].to_numpy() * df["torque_Nm"].to_numpy()
    osf_lim = df["type"].map(OSF_LIMIT).to_numpy()
    phys["dT_K"] = df["process_K"].to_numpy() - df["air_K"].to_numpy()
    phys["power_W"] = df["torque_Nm"].to_numpy() * omega
    phys["strain_minNm"] = strain
    phys["strain_margin"] = osf_lim - strain
    return raw, phys


MODELS = {
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "RF": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                         random_state=1, n_jobs=-1),
    "GB": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=1),
}


def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return th[np.nanargmax(f[:-1])]


def main():
    df = pd.read_csv(ROOT / "data" / "process_data.csv")
    y = df["failure"].to_numpy()
    raw, phys = build_features(df)
    n = len(df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    results, oof_store = {}, {}
    for fs_name, X in [("raw", raw), ("phys", phys)]:
        for mname, make in MODELS.items():
            oof, rows = np.zeros(n), []
            for tr, te in skf.split(X, y):
                # threshold chosen on inner out-of-fold predictions of the training fold only
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)
                pin = np.zeros(len(tr))
                for itr, ite in inner.split(X.iloc[tr], y[tr]):
                    m = make(); m.fit(X.iloc[tr].iloc[itr], y[tr][itr])
                    pin[ite] = m.predict_proba(X.iloc[tr].iloc[ite])[:, 1]
                th = best_f1_threshold(y[tr], pin)
                m = make(); m.fit(X.iloc[tr], y[tr])
                p = m.predict_proba(X.iloc[te])[:, 1]
                oof[te] = p
                yp = (p >= th).astype(int)
                rows.append([precision_score(y[te], yp, zero_division=0), recall_score(y[te], yp),
                             f1_score(y[te], yp), roc_auc_score(y[te], p), average_precision_score(y[te], p)])
            a = np.array(rows)
            key = f"{mname}-{fs_name}"
            results[key] = {k: [a[:, j].mean(), a[:, j].std()] for j, k in
                            enumerate(["precision", "recall", "f1", "roc_auc", "pr_auc"])}
            oof_store[key] = oof
            print(f"{key:8s}", {k: round(v[0], 3) for k, v in results[key].items()})

    # best model: per-mode recall and accuracy from out-of-fold predictions
    best = max(results, key=lambda k: results[k]["f1"][0])
    pb = oof_store[best]
    pred = pb >= best_f1_threshold(y, pb)
    mode_recall = {k: float(pred[df[k] == 1].mean()) for k in MODES}
    accuracy = float((pred == y).mean())
    majority_accuracy = float(1 - y.mean())

    # permutation importance of the best model on a 30 % hold-out
    Xtr, Xte, ytr, yte = train_test_split(phys, y, test_size=0.3, stratify=y, random_state=11)
    mb = MODELS[best.split("-")[0]](); mb.fit(Xtr, ytr)
    pi = permutation_importance(mb, Xte, yte, scoring="average_precision", n_repeats=15, random_state=0, n_jobs=-1)
    imp = sorted(zip(phys.columns, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])

    curves = {}
    for k, v in oof_store.items():
        pr, rc, _ = precision_recall_curve(y, v)
        idx = np.linspace(0, len(pr) - 1, 300).astype(int)
        curves[k] = [rc[idx].tolist(), pr[idx].tolist()]

    stats = {"n": n, "failures": int(y.sum()), "failure_rate": float(y.mean()),
             "modes": {k: int(df[k].sum()) for k in MODES},
             "desc": df[["air_K", "process_K", "speed_rpm", "torque_Nm", "tool_wear_min"]]
                     .describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()}
    out = dict(stats=stats, results=results, best=best,
               best_oof=dict(precision=float(precision_score(y, pred)), recall=float(recall_score(y, pred)),
                             f1=float(f1_score(y, pred)), accuracy=accuracy),
               majority_class_accuracy=majority_accuracy, mode_recall=mode_recall,
               importance=[(a, float(b), float(c)) for a, b, c in imp], curves=curves)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "pm_results.json", "w"), indent=1)
    print("best:", best, out["best_oof"])
    print(f"accuracy of a model that never predicts failure: {100 * majority_accuracy:.1f}%")
    print("per-mode recall:", {k: round(v, 3) for k, v in mode_recall.items()})


if __name__ == "__main__":
    main()
