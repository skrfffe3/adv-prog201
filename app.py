"""
HOPE - Streamlit app

Run from the ADRIPS folder with the venv active:
    streamlit run app.py

Needs models/adrips_model.joblib (created by train_final.py).
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from adrips_core import (CATEGORICAL_FEATURES, CODE_LABELS, RAW_INPUT_COLUMNS,
                         field_help, field_label, ordered_options,
                         disengagement_alert, dropout_contributions,
                         explain_student, friendly, predict_dropout, recommend,
                         risk_level, score_frame, stage_inputs)

MODEL_PATH = Path("models/adrips_model.joblib")

st.set_page_config(page_title="HOPE - Dropout Risk Prediction", page_icon="🎓",
                   layout="wide")

RISK_COLORS = {"Low Risk": "#2e7d32", "Medium Risk": "#ef6c00",
               "High Risk": "#c62828"}

# Binary columns shown as readable options instead of 0/1
BINARY_LABELS = {
    "Daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    "Gender": {1: "Male", 0: "Female"},
}
YES_NO = {1: "Yes", 0: "No"}
BINARY_COLS = ["Displaced", "Educational special needs", "Debtor",
               "Tuition fees up to date", "Scholarship holder", "International",
               "Daytime/evening attendance", "Gender"]

FORM_SECTIONS = {
    "Enrollment & program": [
        "Course", "Application mode", "Application order",
        "Daytime/evening attendance", "Previous qualification",
    ],
    "1st semester": [
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (without evaluations)",
    ],
    "2nd semester": [
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "Financial": ["Tuition fees up to date", "Debtor", "Scholarship holder"],
    "Personal": [
        "Age at enrollment", "Gender", "Marital status", "Nacionality",
        "International", "Displaced", "Educational special needs",
    ],
    "Family background": [
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ],
    "Economic context (year of enrollment)": [
        "Unemployment rate", "Inflation rate", "GDP",
    ],
}
SECTION_NOTES = {
    "1st semester": "Counts are numbers of subjects (curricular units).",
    "2nd semester": "Counts are numbers of subjects (curricular units).",
    "Family background": "Parents' schooling and work. Pick the closest "
                         "match, or 'Unknown' / 'Not specified'. Tip: click a "
                         "list and type a word (e.g. 'teacher') to search.",
    "Economic context (year of enrollment)": "National figures for the year "
                                             "the student enrolled.",
}
INTEGER_COLS = {c for c in RAW_INPUT_COLUMNS
                if c not in CATEGORICAL_FEATURES and "grade" not in c
                and c not in ("Unemployment rate", "Inflation rate", "GDP")}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def risk_badge(level, prob):
    color = RISK_COLORS[level]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:8px;background:{color};"
        f"color:white;font-size:1.4rem;font-weight:600'>{level} "
        f"<span style='font-weight:400;font-size:1rem'>· dropout probability "
        f"{prob:.0%}</span></div>",
        unsafe_allow_html=True,
    )


def input_widget(col, bundle):
    d = bundle["input_defaults"][col]
    key = f"in_{col}"
    if col in BINARY_COLS:
        labels = BINARY_LABELS.get(col, YES_NO)
        opts = [1, 0]
        return st.selectbox(field_label(col), opts, index=opts.index(int(d)),
                            format_func=lambda v: labels[v], key=key,
                            help=field_help(col))
    if col in CATEGORICAL_FEATURES:
        opts = ordered_options(col, bundle["category_choices"][col])
        idx = opts.index(int(d)) if int(d) in opts else 0
        labels = CODE_LABELS.get(col)
        # names only (no codes); the list is searchable - just start typing
        return st.selectbox(field_label(col), opts, index=idx, key=key,
                            format_func=lambda v: labels.get(v, f"Code {v}")
                            if labels else str(v),
                            help=field_help(col))
    lo, hi = bundle["numeric_ranges"][col]
    if col in INTEGER_COLS:
        return st.number_input(field_label(col), min_value=0,
                               max_value=int(hi * 2) + 1, value=int(round(d)),
                               step=1, key=key, help=field_help(col))
    return st.number_input(field_label(col), min_value=float(min(lo, 0) * 2),
                           max_value=float(hi * 2 + 1), value=float(d),
                           step=0.1, format="%.2f", key=key,
                           help=field_help(col))


def contribution_chart(up, down):
    data = pd.concat([up, down]).rename("contribution").reset_index()
    data.columns = ["factor", "contribution"]
    data["factor"] = data["factor"].map(friendly)
    data["effect"] = data["contribution"].apply(
        lambda v: "Raises dropout risk" if v > 0 else "Lowers dropout risk")
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution to dropout score"),
            y=alt.Y("factor:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=320)),  # show full factor names
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(domain=["Raises dropout risk", "Lowers dropout risk"],
                                range=["#c62828", "#2e7d32"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["factor", alt.Tooltip("contribution:Q", format=".3f")],
        )
        .properties(height=40 * len(data) + 40)
    )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def stage_picker(bundle, where):
    order = bundle["stage_order"]
    return where.selectbox(
        "Assessment stage", order, index=len(order) - 1,
        format_func=lambda k: bundle["stages"][k]["label"], key="stage")


def page_single(bundle, stage):
    model = bundle["stages"][stage]
    st.header("Assess a student")
    st.caption(f"**{model['label']}** - {model['description']} "
               "Fields are pre-filled with typical values; change them to "
               "match the student.")
    needed = set(model["inputs"])

    with st.form("student_form"):
        values = {}
        for section, cols in FORM_SECTIONS.items():
            cols = [c for c in cols if c in needed]
            if not cols:
                continue
            with st.expander(section, expanded=section in ("1st semester",
                                                          "2nd semester",
                                                          "Financial")):
                if SECTION_NOTES.get(section):
                    st.caption(SECTION_NOTES[section])
                n = 2 if section == "Family background" else 3
                grid = st.columns(n)
                for i, col in enumerate(cols):
                    with grid[i % n]:
                        values[col] = input_widget(col, bundle)
        submitted = st.form_submit_button("Assess dropout risk", type="primary")

    if not submitted:
        return

    outside = []
    for col, v in values.items():
        rng = bundle["numeric_ranges"].get(col)
        if rng and not (rng[0] <= v <= rng[1]):
            outside.append(f"{field_label(col)} = {v:g} (training data: "
                           f"{rng[0]:g} to {rng[1]:g})")
    if outside:
        st.warning("Some values are outside the range seen in the training "
                   "data, so the prediction for them is less reliable: "
                   + "; ".join(outside))

    for sem in ("1st", "2nd"):
        e = values.get(f"Curricular units {sem} sem (enrolled)")
        a = values.get(f"Curricular units {sem} sem (approved)")
        if e is not None and a is not None and a > e:
            st.warning(f"{sem} semester: approved units are higher than "
                       "enrolled units. Please check the inputs.")

    X = pd.DataFrame([values])
    p_drop = float(predict_dropout(model, X)[0])
    level = risk_level(p_drop, model["thresholds"])
    alert = disengagement_alert(values, stage)

    left, right = st.columns([1, 1])
    with left:
        risk_badge(level, p_drop)
        if alert:
            st.error("**Disengagement alert:** enrolled in units but took no "
                     "evaluations this semester. In the data, almost every "
                     "such student had already left.")
        # risk at earlier stages, using the same student's data
        order = bundle["stage_order"]
        earlier = order[:order.index(stage) + 1]
        if len(earlier) > 1:
            st.subheader("Risk across stages")
            rows = []
            for k in earlier:
                m = bundle["stages"][k]
                p = float(predict_dropout(m, X)[0])
                rows.append({"Stage": m["short"], "Dropout probability": f"{p:.0%}",
                             "Risk level": risk_level(p, m["thresholds"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        th = model["thresholds"]
        st.caption(f"Thresholds for this stage: Medium ≥ {th['medium']:.0%}, "
                   f"High ≥ {th['high']:.0%} dropout probability.")

    contrib = dropout_contributions(model, X).iloc[0]
    up, down = explain_student(contrib)
    with right:
        st.subheader("Why this risk level?")
        if len(up) + len(down):
            st.altair_chart(contribution_chart(up, down), width="stretch")
        st.caption("Per-student factor contributions from the XGBoost model "
                   "(TreeSHAP). Red bars push toward dropout, green bars away.")

    st.subheader("Recommended interventions")
    for title, text in recommend(level, up, alert):
        st.markdown(f"**{title}:** {text}")
    st.caption("Recommendations support, not replace, the judgment of the "
               "guidance counselor.")


def page_batch(bundle, stage):
    model = bundle["stages"][stage]
    st.header("Batch screening")
    st.write(f"**{model['label']}.** Upload a CSV with the student records. "
             f"This stage needs these {len(model['inputs'])} columns "
             "(extra columns such as `Target` are kept but ignored).")
    with st.expander("Required columns"):
        st.write(", ".join(model["inputs"]))
    file = st.file_uploader("Student records (CSV)", type="csv")
    if file is None:
        return
    # sep=None also accepts the semicolon-separated original UCI file
    df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    missing = [c for c in model["inputs"] if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    with st.spinner(f"Scoring {len(df)} students..."):
        res = score_frame(model, stage, df[model["inputs"]])
    out = pd.concat([res.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    counts = res["Risk_Level"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", len(res))
    c2.metric("High Risk", int(counts.get("High Risk", 0)))
    c3.metric("Medium Risk", int(counts.get("Medium Risk", 0)))
    c4.metric("Low Risk", int(counts.get("Low Risk", 0)))
    c5.metric("Disengagement alerts", int(res["Disengagement_Alert"].sum()))

    level_filter = st.multiselect("Show risk levels",
                                  ["High Risk", "Medium Risk", "Low Risk"],
                                  default=["High Risk", "Medium Risk"])
    view = out[out["Risk_Level"].isin(level_filter) | out["Disengagement_Alert"]]
    view = view.sort_values(["Disengagement_Alert", "Dropout_Probability"],
                            ascending=False)
    st.dataframe(view, width="stretch")
    st.download_button("Download results (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       f"adrips_risk_{stage}.csv", "text/csv")


def page_model(bundle, stage):
    info = bundle["info"]
    st.header("Model information")
    st.markdown(
        "ADRIPS predicts **Dropout vs Not dropout** at three points in time. "
        "Each stage uses a weighted ensemble of Random Forest and XGBoost, "
        "tuned with cross-validation; weights and risk thresholds were chosen "
        "on separate cross-validation folds of the training set, and the test "
        f"set ({info['n_test']} students) was used once for the results below."
    )
    rows = []
    for k in bundle["stage_order"]:
        m = bundle["stages"][k]
        e = list(m["test_comparison"].values())[-1]
        a = m.get("test_active")
        rows.append({
            "Stage": m["short"], "ROC AUC": e["roc_auc"],
            "Dropouts caught": f"{e['caught']} / {e['dropouts']}",
            "Recall": e["recall"], "Precision": e["precision"],
            "Students flagged": e["flagged"],
            "AUC (active students)": a["roc_auc"] if a else None,
            "RF / XGB weight": f"{m['rf_weight']:.1f} / {1 - m['rf_weight']:.1f}",
            "Medium / High threshold": f"{m['thresholds']['medium']:.2f} / "
                                       f"{m['thresholds']['high']:.2f}",
        })
    st.subheader("Test set results per stage")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    m = bundle["stages"][stage]
    st.subheader(f"{m['label']}: model comparison")
    st.dataframe(pd.DataFrame(m["test_comparison"]).T[
        ["roc_auc", "recall", "precision", "f1", "accuracy"]],
        width="stretch")
    st.subheader("Actual dropout rate per risk level (test set)")
    st.dataframe(pd.DataFrame(m["risk_bands"]), hide_index=True, width="stretch")
    st.info("Notes: 'Enrolled' students count as Not dropout, though some may "
            "still leave later. Students enrolled in units but with no "
            "evaluations are flagged by a separate disengagement rule.")
    st.caption(f"Trained {info['trained_at']} · scikit-learn {info['sklearn_version']}"
               + (" · QUICK MODE - retrain before reporting" if info.get("quick_mode") else ""))


def main():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}. "
                 "Run `python train_final.py` first.")
        st.stop()
    bundle = load_bundle()
    if bundle.get("version") != 2:
        st.error("This model file is from the older version. "
                 "Run `python train_final.py` again.")
        st.stop()

    st.sidebar.title("🎓 HOPE")
    st.sidebar.caption("Student Dropout Risk Prediction and Intervention Support")
    page = st.sidebar.radio("Go to", ["Assess a student", "Batch screening",
                                      "Model information"])
    stage = stage_picker(bundle, st.sidebar)
    {"Assess a student": page_single,
     "Batch screening": page_batch,
     "Model information": page_model}[page](bundle, stage)


main()
