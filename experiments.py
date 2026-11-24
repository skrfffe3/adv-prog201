"""
ADRIPS - extended model experiments

Run from the ADRIPS folder with the venv active:
    python experiments.py            (full run, roughly 10-20 minutes)
    python experiments.py --quick    (fast smoke test, ~2 minutes)

Optional extra models (installed automatically if present):
    pip install lightgbm

Part A  Ten-plus classifiers + soft voting + stacking (3 classes)
Part B  Binary framing: Dropout vs Not dropout (+ threshold for 85% recall)
Part C  Early warning: how well can we predict with LESS data
        (1st semester only / enrollment data only)

Rules (same as train_final.py):
- Same 80/20 stratified split (random_state=42).
- Every model is compared with the SAME 5-fold CV on the training set.
- Every choice (which models to combine, thresholds) uses training data
  only. The test set is scored once at the end, for reporting.
"""

import os
import warnings

# Harmless warning from nested parallelism (CV folds in parallel, each model
# also multi-threaded). Silenced here and in the parallel worker processes.
_W = "ignore:`sklearn.utils.parallel.delayed` should be used:UserWarning"
os.environ["PYTHONWARNINGS"] = _W
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from adrips_core import (CATEGORICAL_FEATURES, CLASS_NAMES, DROPOUT_CLASS,
                         NUMERIC_FEATURES, RAW_INPUT_COLUMNS, TARGET_COL,
                         TARGET_MAPPING, BalancedXGB, FeatureEngineer)

QUICK = "--quick" in sys.argv
DATA_PATH = Path("dataset/predict-student-dropout-and-academic-success.csv")
REPORT_PATH = Path("reports/experiments_report.txt")
TABLE_PATH = Path("reports/experiments_results.csv")
RS = 42
FOLDS = 3 if QUICK else 5
SIZE = 0.3 if QUICK else 1.0   # shrinks tree counts in quick mode
TARGET_RECALL = 0.85           # Part B: catch at least 85% of dropouts

LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def n(k):
    return max(20, int(k * SIZE))


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def pipe(model, scale=False, numeric=NUMERIC_FEATURES,
         categorical=CATEGORICAL_FEATURES):
    steps = [("features", FeatureEngineer()),
             ("preprocess", make_preprocessor(numeric, categorical))]
    if scale:  # distance/gradient-based models need standardized inputs
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def rf_tuned():
    # Best settings found by train_final.py
    return RandomForestClassifier(n_estimators=n(200), min_samples_split=5,
                                  min_samples_leaf=1, max_features="log2",
                                  max_depth=30, class_weight="balanced",
                                  random_state=RS, n_jobs=-1)


def xgb_tuned():
    # Best settings found by train_final.py
    return BalancedXGB(n_estimators=n(500), max_depth=3, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                       random_state=RS)


def current_ensemble(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES):
    """The model currently used by the app: 0.4 RF + 0.6 XGBoost."""
    return VotingClassifier(
        [("rf", pipe(rf_tuned(), numeric=numeric, categorical=categorical)),
         ("xgb", pipe(xgb_tuned(), numeric=numeric, categorical=categorical))],
        voting="soft", weights=[0.4, 0.6])


def model_zoo():
    zoo = {
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=3000, class_weight="balanced"), scale=True),
        # SVC(probability=True) is deprecated in scikit-learn 1.9
        "SVM (RBF kernel)": pipe(CalibratedClassifierCV(SVC(
            C=1.0, gamma="scale", class_weight="balanced", random_state=RS),
            ensemble=False), scale=True),
        "k-Nearest Neighbors": pipe(KNeighborsClassifier(
            n_neighbors=25, weights="distance"), scale=True),
        "Neural Network (MLP)": pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
            max_iter=500, random_state=RS), scale=True),
        "Extra Trees": pipe(ExtraTreesClassifier(
            n_estimators=n(500), min_samples_leaf=2, class_weight="balanced",
            random_state=RS, n_jobs=-1)),
        "Random Forest (tuned)": pipe(rf_tuned()),
        "HistGradientBoosting": pipe(HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=n(300), max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced", random_state=RS)),
        "XGBoost (tuned)": pipe(xgb_tuned()),
    }
    try:
        from lightgbm import LGBMClassifier
        zoo["LightGBM"] = pipe(LGBMClassifier(
            n_estimators=n(400), learning_rate=0.03, num_leaves=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight="balanced", random_state=RS, verbose=-1))
    except ImportError:
        log("(LightGBM not installed - skipped. `pip install lightgbm` to include it.)")
    return zoo


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
MULTI_SCORERS = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "dropout_f1": make_scorer(f1_score, labels=[DROPOUT_CLASS], average="macro"),
    "dropout_recall": make_scorer(recall_score, labels=[DROPOUT_CLASS], average="macro"),
    "dropout_precision": make_scorer(precision_score, labels=[DROPOUT_CLASS],
                                     average="macro", zero_division=0),
}
BINARY_SCORERS = {
    "roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1",
    "recall": "recall", "precision": "precision",
}


def cv_eval(model, X, y, cv, scorers):
    t0 = time.time()
    r = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
    out = {f"cv_{k}": r[f"test_{k}"].mean() for k in scorers}
    out["cv_main_std"] = r[f"test_{list(scorers)[1 if 'roc_auc' not in scorers else 0]}"].std()
    out["seconds"] = time.time() - t0
    return out


def test_multi(y, pred):
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_macro_f1": f1_score(y, pred, average="macro"),
        "test_dropout_f1": f1_score(y, pred, labels=[DROPOUT_CLASS], average="macro"),
        "test_dropout_recall": recall_score(y, pred, labels=[DROPOUT_CLASS], average="macro"),
        "test_dropout_precision": precision_score(y, pred, labels=[DROPOUT_CLASS],
                                                  average="macro", zero_division=0),
    }


def test_binary(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "test_roc_auc": roc_auc_score(y, proba),
        "test_accuracy": accuracy_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_recall": recall_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
    }


def mcnemar(y_true, pred_a, pred_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b, c = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)
    return b, c, min(p, 1.0)


def show(df, cols):
    log(df[cols].round(4).to_string())


# --------------------------------------------------------------------------
# Part A - many models, 3 classes
# --------------------------------------------------------------------------
def part_a(X_tr, X_te, y_tr, y_te, cv):
    log("\n" + "=" * 78)
    log("PART A - Model comparison (Graduate / Enrolled / Dropout)")
    log("=" * 78)
    models = model_zoo()
    models["Current ADRIPS ensemble (0.4 RF + 0.6 XGB)"] = current_ensemble()

    rows = {}
    for name, m in models.items():
        log(f"  cross-validating {name} ...")
        rows[name] = cv_eval(m, X_tr, y_tr, cv, MULTI_SCORERS)

    # Pick ensemble members using CV only
    singles = pd.DataFrame(rows).T.drop(index=[k for k in rows if k.startswith("Current")])
    ranked = singles.sort_values("cv_macro_f1", ascending=False).index.tolist()
    top3, top5 = ranked[:3], ranked[:5]
    log(f"\n  Top 3 by CV macro F1 -> soft voting: {top3}")
    log(f"  Top 5 by CV macro F1 -> stacking:    {top5}")

    zoo = model_zoo()
    models["Soft Voting (top 3)"] = VotingClassifier(
        [(f"m{i}", zoo[k]) for i, k in enumerate(top3)], voting="soft")
    models["Stacking (top 5 + LR meta-learner)"] = StackingClassifier(
        [(f"m{i}", zoo[k]) for i, k in enumerate(top5)],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=StratifiedKFold(FOLDS, shuffle=True, random_state=RS),
        stack_method="predict_proba", n_jobs=1)
    for name in ["Soft Voting (top 3)", "Stacking (top 5 + LR meta-learner)"]:
        log(f"  cross-validating {name} ...")
        rows[name] = cv_eval(models[name], X_tr, y_tr, cv, MULTI_SCORERS)

    # One-time test evaluation
    preds = {}
    for name, m in models.items():
        log(f"  fitting {name} on full training set ...")
        m.fit(X_tr, y_tr)
        preds[name] = m.predict(X_te)
        rows[name].update(test_multi(y_te, preds[name]))

    res = pd.DataFrame(rows).T.sort_values("cv_macro_f1", ascending=False)
    log("\nCross-validation on the training set (used for choosing):")
    show(res, ["cv_macro_f1", "cv_main_std", "cv_accuracy", "cv_dropout_f1",
               "cv_dropout_recall", "cv_dropout_precision", "seconds"])
    log("\nTest set (885 students, reported once):")
    show(res, ["test_accuracy", "test_macro_f1", "test_dropout_f1",
               "test_dropout_recall", "test_dropout_precision"])

    base = "Current ADRIPS ensemble (0.4 RF + 0.6 XGB)"
    log(f"\nMcNemar test vs {base} (p < 0.05 = real difference):")
    for name in res.index:
        if name == base:
            continue
        b, c, p = mcnemar(y_te.values, preds[name], preds[base])
        log(f"  {name:<40} only-this-correct={b:>3}  only-current-correct={c:>3}  p={p:.4f}")

    best = res.index[0]
    gap = res.loc[best, "cv_macro_f1"] - res.loc[base, "cv_macro_f1"]
    log(f"\nBest by CV: {best} (CV macro F1 {res.loc[best, 'cv_macro_f1']:.4f}, "
        f"{gap:+.4f} vs current ensemble).")
    if best == base or gap < res.loc[base, "cv_main_std"]:
        log("-> The gain is within one CV standard deviation: no model is "
            "meaningfully better than the current ensemble.")
    else:
        log("-> Larger than one CV standard deviation: worth considering.")
    res.insert(0, "part", "A: 3-class")
    return res


# --------------------------------------------------------------------------
# Part B - binary: Dropout vs Not dropout
# --------------------------------------------------------------------------
def part_b(X_tr, X_te, y_tr, y_te, cv):
    log("\n" + "=" * 78)
    log("PART B - Binary framing: Dropout vs Not dropout")
    log("=" * 78)
    yb_tr = (y_tr == DROPOUT_CLASS).astype(int)
    yb_te = (y_te == DROPOUT_CLASS).astype(int)
    log(f"Dropout share: {yb_tr.mean():.1%} of training students")

    models = {
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=3000, class_weight="balanced"), scale=True),
        "Random Forest (tuned)": pipe(rf_tuned()),
        "XGBoost (tuned)": pipe(xgb_tuned()),
        "Ensemble (0.4 RF + 0.6 XGB)": current_ensemble(),
    }
    rows, probas = {}, {}
    for name, m in models.items():
        log(f"  cross-validating {name} ...")
        rows[name] = cv_eval(m, X_tr, yb_tr, cv, BINARY_SCORERS)
        m.fit(X_tr, yb_tr)
        probas[name] = m.predict_proba(X_te)[:, 1]
        rows[name].update(test_binary(yb_te, probas[name]))
    res = pd.DataFrame(rows).T.sort_values("cv_roc_auc", ascending=False)
    log("\nCross-validation (training set):")
    show(res, ["cv_roc_auc", "cv_main_std", "cv_accuracy", "cv_f1", "cv_recall", "cv_precision"])
    log("\nTest set at threshold 0.50:")
    show(res, ["test_roc_auc", "test_accuracy", "test_f1", "test_recall", "test_precision"])

    # Threshold for target recall, chosen on out-of-fold TRAINING predictions
    name = "Ensemble (0.4 RF + 0.6 XGB)"
    oof = cross_val_predict(current_ensemble(), X_tr, yb_tr, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    cands = np.round(np.arange(0.05, 0.951, 0.01), 2)
    ok = [t for t in cands if recall_score(yb_tr, (oof >= t).astype(int)) >= TARGET_RECALL]
    th = max(ok) if ok else 0.5
    log(f"\nThreshold for >= {TARGET_RECALL:.0%} dropout recall (chosen on training "
        f"out-of-fold predictions): {th:.2f}")
    t = test_binary(yb_te, probas[name], th)
    log(f"Test set at {th:.2f}: recall {t['test_recall']:.3f}, precision "
        f"{t['test_precision']:.3f}, F1 {t['test_f1']:.3f}, accuracy {t['test_accuracy']:.3f}")
    flagged = int((probas[name] >= th).sum())
    caught = int(((probas[name] >= th) & (yb_te == 1)).sum())
    log(f"-> Flags {flagged} of {len(yb_te)} students; catches {caught} of "
        f"{int(yb_te.sum())} actual dropouts.")
    res.insert(0, "part", "B: binary")
    return res


# --------------------------------------------------------------------------
# Part C - early warning with less data
# --------------------------------------------------------------------------
def part_c(X_tr, X_te, y_tr, y_te, cv):
    log("\n" + "=" * 78)
    log("PART C - Early warning: predicting with less data (current ensemble)")
    log("=" * 78)
    second_sem = {"Approval_Rate_2nd", "Overall_Approval_Rate",
                  "Failed_Units_2nd", "Average_Grade"}
    sets = {
        "End of year 1 (all data, current)": NUMERIC_FEATURES,
        "End of semester 1 (no 2nd-sem data)": [
            f for f in NUMERIC_FEATURES if "2nd sem" not in f and f not in second_sem],
        "At enrollment (no grades at all)": [
            f for f in NUMERIC_FEATURES if "Curricular" not in f
            and f not in second_sem | {"Approval_Rate_1st", "Failed_Units_1st"}],
    }
    yb_tr = (y_tr == DROPOUT_CLASS).astype(int)
    yb_te = (y_te == DROPOUT_CLASS).astype(int)
    rows = {}
    for label, numeric in sets.items():
        log(f"  {label}: {len(numeric)} numeric + {len(CATEGORICAL_FEATURES)} categorical features")
        m3 = current_ensemble(numeric=numeric)
        r = cv_eval(m3, X_tr, y_tr, cv, MULTI_SCORERS)
        m3.fit(X_tr, y_tr)
        r.update(test_multi(y_te, m3.predict(X_te)))
        mb = current_ensemble(numeric=numeric)
        rb = cross_validate(mb, X_tr, yb_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        mb.fit(X_tr, yb_tr)
        r["cv_binary_roc_auc"] = rb["test_score"].mean()
        r["test_binary_roc_auc"] = roc_auc_score(yb_te, mb.predict_proba(X_te)[:, 1])
        rows[label] = r
    res = pd.DataFrame(rows).T
    show(res, ["cv_macro_f1", "cv_dropout_f1", "cv_binary_roc_auc",
               "test_macro_f1", "test_dropout_f1", "test_dropout_recall",
               "test_binary_roc_auc"])
    res.insert(0, "part", "C: early warning")
    return res


def main():
    start = time.time()
    log(f"ADRIPS experiments - {datetime.now():%Y-%m-%d %H:%M}"
        + ("  [QUICK MODE - not for reporting]" if QUICK else ""))
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    X, y = df[RAW_INPUT_COLUMNS], df[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=RS, stratify=y)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=RS)
    log(f"Train {len(X_tr)} / Test {len(X_te)} / {FOLDS}-fold CV")

    a = part_a(X_tr, X_te, y_tr, y_te, cv)
    b = part_b(X_tr, X_te, y_tr, y_te, cv)
    c = part_c(X_tr, X_te, y_tr, y_te, cv)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    pd.concat([a, b, c]).to_csv(TABLE_PATH)
    log(f"\nTotal time: {(time.time() - start) / 60:.1f} min")
    REPORT_PATH.write_text("\n".join(LINES), encoding="utf-8")
    print(f"Saved {REPORT_PATH} and {TABLE_PATH}")


if __name__ == "__main__":
    main()
