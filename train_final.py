"""
ADRIPS - final model training (multi-stage early warning)

Run from the ADRIPS folder with the venv active:
    python train_final.py            (full run, roughly 10-15 minutes)
    python train_final.py --quick    (fast smoke test - NOT for reporting)

Design (based on experiments.py results)
- Task: Dropout (1) vs Not dropout (0). Graduate and Enrolled = 0.
- Three stage models, one per point in time:
      Stage 1  at enrollment       background, program, socio-economic data
      Stage 2  after 1st semester  + 1st-semester results
      Stage 3  after 1st year      + 2nd-semester results
- Each stage: 0.4/0.6-style ensemble of Random Forest + XGBoost, both tuned.

Leakage safeguards
1. One 80/20 split (same as before). The test set is touched ONCE, at the end.
2. All preprocessing (feature engineering, imputing, encoding) is inside the
   pipeline, so it is re-fitted on the training part of every CV fold.
3. Hyperparameters are tuned on one set of CV folds; the ensemble weight
   and risk thresholds are chosen on a DIFFERENT set of folds, so the same
   folds are not used twice for decisions.
4. Each stage only sees information that exists at that point in time
   (attribute groups from Realinho et al., 2022, Table 1).
5. Missing academic records (0 enrolled units) are treated as missing,
   not as "0% approval / grade 0".
6. An automatic audit checks for train/test overlap and for any single
   feature that predicts dropout suspiciously well on its own.
7. Stage 2 and 3 are also evaluated on "active" students only (those who
   took at least one evaluation), because students who enrolled but took no
   exams have practically already left; including them inflates the scores.
"""

import os
import warnings

# Harmless warning from nested parallelism (CV folds in parallel, each model
# also multi-threaded). Silenced here and in the parallel worker processes.
_W = "ignore:`sklearn.utils.parallel.delayed` should be used:UserWarning"
os.environ["PYTHONWARNINGS"] = _W
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adrips_core import (CATEGORICAL_FEATURES, RAW_INPUT_COLUMNS, STAGE_ORDER,
                         STAGES, TARGET_COL, BalancedXGB, FeatureEngineer,
                         disengagement_alert, risk_level, stage_inputs)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
QUICK = "--quick" in sys.argv
DATA_PATH = Path("dataset/predict-student-dropout-and-academic-success.csv")
MODEL_PATH = Path("models/adrips_model.joblib")
REPORT_PATH = Path("reports/training_report.txt")

RANDOM_STATE = 42
SELECT_STATE = 7               # different folds for weight/threshold choice
N_ITER = 4 if QUICK else 20
CV_FOLDS = 3 if QUICK else 5
SCORING = "roc_auc"            # ranking quality; thresholds are set separately
# Share of dropouts that Medium + High must catch, per stage. Stage 1 has
# little information, so asking for 85% there would flag most students.
TARGET_RECALL = {"enrollment": 0.70, "semester1": 0.85, "year1": 0.85}
HIGH_PRECISION = 0.90          # High Risk must be right >= 90% of the time
SUSPICIOUS_AUC = 0.90          # single-feature AUC that triggers a warning

RF_PARAM_GRID = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [10, 20, 30, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}
XGB_PARAM_GRID = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 4, 6, 8],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
}

LINES = []


def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def make_pipeline(model, stage_key, scale=False):
    st = STAGES[stage_key]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), st["numeric"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # dense: with sparse input XGBoost would read every 0 as missing
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), st["categorical"]),
    ])
    steps = [("features", FeatureEngineer()), ("preprocess", pre)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def rf_default():
    return RandomForestClassifier(n_estimators=100 if QUICK else 300,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)


def xgb_default():
    return BalancedXGB(n_estimators=100 if QUICK else 300, max_depth=6,
                       learning_rate=0.05, subsample=1.0, colsample_bytree=1.0,
                       random_state=RANDOM_STATE)


def tune(name, pipeline, grid, X, y, cv):
    """Default vs best of N_ITER random settings on the same folds."""
    default = cross_val_score(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    search = RandomizedSearchCV(pipeline, grid, n_iter=N_ITER, cv=cv,
                                scoring=SCORING, n_jobs=-1,
                                random_state=RANDOM_STATE, refit=False)
    search.fit(X, y)
    i = search.best_index_
    best, best_sd = (search.cv_results_["mean_test_score"][i],
                     search.cv_results_["std_test_score"][i])
    log(f"  {name:<14} default {default.mean():.4f} ± {default.std():.4f} | "
        f"tuned {best:.4f} ± {best_sd:.4f}")
    if best > default.mean():
        log(f"  {'':<14} -> tuned kept: {search.best_params_}")
        return pipeline.set_params(**search.best_params_), "tuned", search.best_params_, best
    log(f"  {'':<14} -> default kept (tuning did not help)")
    return pipeline, "default", {}, default.mean()


def feature_groups(fitted_pipeline, stage_key):
    """Map transformed columns back to original features."""
    st = STAGES[stage_key]
    groups, pos = [], 0
    for f in st["numeric"]:
        groups.append((f, [pos]))
        pos += 1
    enc = fitted_pipeline.named_steps["preprocess"].named_transformers_["cat"] \
        .named_steps["encoder"]
    for f, cats in zip(st["categorical"], enc.categories_):
        groups.append((f, list(range(pos, pos + len(cats)))))
        pos += len(cats)
    return groups


def pick_thresholds(y, p, target_recall):
    """Medium: highest cut-off that still catches >= target_recall of
    dropouts. High: lowest cut-off above Medium whose precision is
    >= HIGH_PRECISION. Chosen on out-of-fold TRAINING predictions."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    ok = [t for t in grid if recall_score(y, p >= t) >= target_recall]
    medium = float(max(ok)) if ok else 0.5
    hi = [t for t in grid if t > medium and (p >= t).sum() > 0
          and precision_score(y, p >= t) >= HIGH_PRECISION]
    high = float(min(hi)) if hi else round(medium + (1 - medium) / 2, 2)
    return {"medium": medium, "high": high}


def binary_metrics(y, p, threshold):
    pred = p >= threshold
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "flagged": int(pred.sum()),
        "caught": int((pred & (y == 1)).sum()),
        "dropouts": int((y == 1).sum()),
        "n": int(len(y)),
    }


INT_KEYS = {"flagged", "caught", "dropouts", "n"}


def band_table(y, p, th):
    df = pd.DataFrame({"Risk_Level": [risk_level(v, th) for v in p],
                       "Dropout": np.asarray(y)})
    t = df.groupby("Risk_Level").agg(Students=("Dropout", "size"),
                                     Actual_Dropouts=("Dropout", "sum"))
    t = t.reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0).astype(int)
    t["Dropout_Rate"] = (t.Actual_Dropouts / t.Students.replace(0, np.nan)).round(3)
    return t


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def audit(X_tr, X_te, y_tr):
    section("LEAKAGE AUDIT")
    overlap = pd.merge(X_tr.drop_duplicates(), X_te.drop_duplicates(),
                       how="inner", on=RAW_INPUT_COLUMNS)
    log(f"[1] Identical student rows in both train and test: {len(overlap)}"
        + ("  OK" if len(overlap) == 0 else "  WARNING"))

    log("[2] Preprocessing is inside each pipeline -> re-fitted per CV fold.  OK")

    for key in STAGE_ORDER:
        cols = set(stage_inputs(key))
        if key == "enrollment":
            bad = [c for c in cols if "Curricular" in c]
        elif key == "semester1":
            bad = [c for c in cols if "2nd sem" in c]
        else:
            bad = []
        log(f"[3] {STAGES[key]['short']:<20} uses {len(cols)} raw inputs; "
            f"future information: {bad if bad else 'none'}"
            + ("  OK" if not bad else "  WARNING"))

    log(f"[4] Single-feature check (training set): features that alone give "
        f"AUC >= {SUSPICIOUS_AUC} would suggest a target proxy.")
    eng = FeatureEngineer().transform(X_tr)
    all_num = sorted(set().union(*[STAGES[k]["numeric"] for k in STAGE_ORDER])
                     | {"Debtor", "Tuition fees up to date", "Scholarship holder"})
    rows = []
    for c in all_num:
        v = eng[c]
        m = v.notna()
        if v[m].nunique() < 2:
            continue
        auc = roc_auc_score(y_tr[m], v[m])
        rows.append((c, max(auc, 1 - auc)))
    rows.sort(key=lambda r: -r[1])
    for c, a in rows[:6]:
        log(f"      {c:<45} AUC {a:.3f}" + ("  WARNING" if a >= SUSPICIOUS_AUC else ""))
    if all(a < SUSPICIOUS_AUC for _, a in rows):
        log("      No single feature is a near-perfect predictor.  OK")

    n1 = sum(disengagement_alert(r, "semester1") for _, r in X_tr.iterrows())
    n2 = sum(disengagement_alert(r, "year1") for _, r in X_tr.iterrows())
    log(f"[5] Training students enrolled but with 0 evaluations: "
        f"{n1} in 1st sem, {n2} in 2nd sem.")
    log("    These have practically already left, so stage 2/3 results are "
        "also reported for 'active' students only.")
    nr = int((X_tr["Curricular units 1st sem (enrolled)"] == 0).sum())
    log(f"[6] Training students with no academic record (0 enrolled units): {nr} "
        f"-> treated as missing + flagged, not as failing.")
    log("[7] Note: 'Enrolled' students are labelled Not dropout, but some may "
        "still drop out later (label noise that can only lower the scores).")
    log("[8] Note: the dataset authors cleaned it before release (records "
        "removed, outliers and missing values handled). Their cleaning cannot "
        "be audited here; real school data will be messier.")
    log("[9] Note: Unemployment/Inflation/GDP have one value per enrollment "
        "year, so they also identify the student's cohort (2008/09-2018/19).")


# --------------------------------------------------------------------------
# One stage
# --------------------------------------------------------------------------
def train_stage(key, X_tr, X_te, y_tr, y_te):
    st = STAGES[key]
    section(f"{st['label'].upper()}")
    log(st["description"])
    log(f"Features: {len(st['numeric'])} numeric + {len(st['categorical'])} categorical")
    cv_tune = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_sel = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SELECT_STATE)

    log(f"\nTuning ({CV_FOLDS}-fold CV, {SCORING}):")
    rf, rf_mode, rf_params, rf_cv = tune("Random Forest", make_pipeline(rf_default(), key),
                                         RF_PARAM_GRID, X_tr, y_tr, cv_tune)
    xgb, xgb_mode, xgb_params, xgb_cv = tune("XGBoost", make_pipeline(xgb_default(), key),
                                             XGB_PARAM_GRID, X_tr, y_tr, cv_tune)

    log("\nEnsemble weight and thresholds (separate CV folds, training set only):")
    oof_rf = cross_val_predict(rf, X_tr, y_tr, cv=cv_sel, method="predict_proba", n_jobs=-1)[:, 1]
    oof_xgb = cross_val_predict(xgb, X_tr, y_tr, cv=cv_sel, method="predict_proba", n_jobs=-1)[:, 1]
    scores = [(w, roc_auc_score(y_tr, w * oof_rf + (1 - w) * oof_xgb))
              for w in np.round(np.arange(0, 1.01, 0.1), 1)]
    w, w_auc = max(scores, key=lambda r: (round(r[1], 4), -abs(r[0] - 0.5)))
    log("  " + "  ".join(f"{a:.1f}:{s:.4f}" for a, s in scores))
    log(f"  -> RF weight {w:.1f} / XGBoost weight {1 - w:.1f} (OOF AUC {w_auc:.4f})")
    oof = w * oof_rf + (1 - w) * oof_xgb
    th = pick_thresholds(y_tr.values, oof, TARGET_RECALL[key])
    log(f"  -> Thresholds: Medium >= {th['medium']:.2f} (catch >= {TARGET_RECALL[key]:.0%} "
        f"of dropouts), High >= {th['high']:.2f} (precision >= {HIGH_PRECISION:.0%})")

    # Final fit on all training data, then ONE look at the test set
    rf.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)
    lr = make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"),
                       key, scale=True).fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]
    p_xgb = xgb.predict_proba(X_te)[:, 1]
    p_ens = w * p_rf + (1 - w) * p_xgb
    p_lr = lr.predict_proba(X_te)[:, 1]

    log("\nTest set (reported once):")
    yv = y_te.values
    comp = pd.DataFrame({
        "Logistic Regression (baseline)": binary_metrics(yv, p_lr, th["medium"]),
        f"Random Forest ({rf_mode})": binary_metrics(yv, p_rf, th["medium"]),
        f"XGBoost ({xgb_mode})": binary_metrics(yv, p_xgb, th["medium"]),
        f"Ensemble ({w:.1f}/{1 - w:.1f})": binary_metrics(yv, p_ens, th["medium"]),
    }).T
    log(f"(recall/precision/F1/accuracy at the Medium threshold {th['medium']:.2f})")
    log(comp[["roc_auc", "recall", "precision", "f1", "accuracy"]].astype(float)
        .round(4).to_string())

    ens = binary_metrics(yv, p_ens, th["medium"])
    log(f"\nEnsemble flags {ens['flagged']} of {ens['n']} students (Medium + High) "
        f"and catches {ens['caught']} of {ens['dropouts']} actual dropouts.")
    bands_all = band_table(yv, p_ens, th)
    log("\nRisk levels - all test students:")
    log(bands_all.to_string())

    active = None
    if st["activity_col"]:
        mask = (X_te[st["activity_col"]] > 0).values
        active = binary_metrics(yv[mask], p_ens[mask], th["medium"])
        log(f"\nActive students only ({mask.sum()} took >= 1 evaluation in the "
            f"latest semester; {(~mask).sum()} excluded):")
        log(f"  AUC {active['roc_auc']:.4f} | recall {active['recall']:.3f} | "
            f"precision {active['precision']:.3f} | catches {active['caught']} of "
            f"{active['dropouts']} dropouts")
        log(band_table(yv[mask], p_ens[mask], th).to_string())

    return {
        "label": st["label"], "short": st["short"],
        "description": st["description"], "inputs": stage_inputs(key),
        "rf_pipeline": rf, "xgb_pipeline": xgb, "rf_weight": float(w),
        "thresholds": th, "feature_groups": feature_groups(xgb, key),
        "test_comparison": {k: {m: (int(v) if m in INT_KEYS else round(float(v), 4))
                                for m, v in r.items()} for k, r in comp.iterrows()},
        "test_active": active,
        "risk_bands": bands_all.reset_index().to_dict("records"),
        "info": {"rf_mode": rf_mode, "rf_params": rf_params, "rf_cv_auc": round(float(rf_cv), 4),
                 "xgb_mode": xgb_mode, "xgb_params": xgb_params,
                 "xgb_cv_auc": round(float(xgb_cv), 4), "oof_ensemble_auc": round(float(w_auc), 4)},
    }


def sensitivity_financial(X_tr, y_tr):
    """How much of Stage 1 depends on financial status (Debtor / Tuition
    fees up to date)? The dataset paper lists these as socio-economic data
    known at enrollment, so they are included; this shows their weight."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    st = STAGES["enrollment"]
    withfin = cross_val_score(make_pipeline(rf_default(), "enrollment"), X_tr, y_tr,
                              cv=cv, scoring=SCORING, n_jobs=-1).mean()
    saved = (st["numeric"], st["categorical"])
    st["numeric"] = [c for c in saved[0] if c != "Financial_Risk"]
    st["categorical"] = [c for c in saved[1]
                         if c not in ("Debtor", "Tuition fees up to date")]
    try:
        without = cross_val_score(make_pipeline(rf_default(), "enrollment"), X_tr, y_tr,
                                  cv=cv, scoring=SCORING, n_jobs=-1).mean()
    finally:
        st["numeric"], st["categorical"] = saved
    log(f"\nSensitivity check, Stage 1 (Random Forest, CV AUC): with financial "
        f"status {withfin:.4f} | without it {without:.4f} ({without - withfin:+.4f}).")
    log("Financial status is included at enrollment because Realinho et al. "
        "(2022) group it with socio-economic data known at enrollment.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    start = time.time()
    log(f"ADRIPS multi-stage training - {datetime.now():%Y-%m-%d %H:%M}"
        + ("  [QUICK MODE - NOT FOR REPORTING]" if QUICK else ""))
    log(f"scikit-learn {sklearn.__version__}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    X = df[RAW_INPUT_COLUMNS]
    y3 = df[TARGET_COL]
    y = (y3 == "Dropout").astype(int)
    log(f"Dataset: {len(df)} records | Dropout {y.sum()} ({y.mean():.1%}) | "
        f"Not dropout {len(y) - y.sum()} (Graduate {int((y3 == 'Graduate').sum())}, "
        f"Enrolled {int((y3 == 'Enrolled').sum())})")

    # stratify on the 3 original classes -> identical split to earlier runs
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y3)
    log(f"Train {len(X_tr)} / Test {len(X_te)}")

    audit(X_tr, X_te, y_tr)

    stages = {}
    for key in STAGE_ORDER:
        stages[key] = train_stage(key, X_tr, X_te, y_tr, y_te)
    sensitivity_financial(X_tr, y_tr)

    section("SUMMARY - one dropout model per stage (test set)")
    rows = []
    for key in STAGE_ORDER:
        s = stages[key]
        e = list(s["test_comparison"].values())[-1]
        a = s["test_active"]
        rows.append({"Stage": s["short"], "AUC": e["roc_auc"], "Recall": e["recall"],
                     "Precision": e["precision"], "Flagged": e["flagged"],
                     "Caught": f"{e['caught']}/{e['dropouts']}",
                     "AUC (active only)": a["roc_auc"] if a else None,
                     "Medium/High": f"{s['thresholds']['medium']:.2f}/{s['thresholds']['high']:.2f}"})
    log(pd.DataFrame(rows).set_index("Stage").to_string())

    defaults, choices, ranges = {}, {}, {}
    for c in RAW_INPUT_COLUMNS:
        if c in CATEGORICAL_FEATURES:
            defaults[c] = int(X_tr[c].mode()[0])
            choices[c] = sorted(int(v) for v in X_tr[c].unique())
        else:
            defaults[c] = float(X_tr[c].median())
            ranges[c] = (float(X[c].min()), float(X[c].max()))

    bundle = {
        "version": 2,
        "stage_order": STAGE_ORDER,
        "stages": stages,
        "input_defaults": defaults,
        "category_choices": choices,
        "numeric_ranges": ranges,
        "info": {"trained_at": datetime.now().isoformat(timespec="minutes"),
                 "sklearn_version": sklearn.__version__, "quick_mode": QUICK,
                 "scoring": SCORING, "target_recall": TARGET_RECALL,
                 "high_precision": HIGH_PRECISION,
                 "n_train": len(X_tr), "n_test": len(X_te)},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    log(f"\nSaved model bundle -> {MODEL_PATH}")
    log(f"Info: {json.dumps(bundle['info'])}")
    log(f"Total time: {(time.time() - start) / 60:.1f} min")
    REPORT_PATH.write_text("\n".join(LINES), encoding="utf-8")
    print(f"Saved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
