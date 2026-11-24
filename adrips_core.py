"""
ADRIPS core logic, shared by train_final.py and app.py.

Keep this file next to app.py: the saved model file refers to the
FeatureEngineer class below, so Python must be able to import it
when the model is loaded.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# Column definitions (same as in main.ipynb)
# --------------------------------------------------------------------------
TARGET_COL = "Target"
TARGET_MAPPING = {"Graduate": 0, "Enrolled": 1, "Dropout": 2}
CLASS_NAMES = ["Graduate", "Enrolled", "Dropout"]
DROPOUT_CLASS = 2

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

RAW_NUMERIC_FEATURES = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

ENGINEERED_FEATURES = [
    "Approval_Rate_1st",
    "Approval_Rate_2nd",
    "Overall_Approval_Rate",
    "Failed_Units_1st",
    "Failed_Units_2nd",
    "Average_Grade",
    "Financial_Risk",
]

# The 34 columns a user must supply (everything in the CSV except Target)
RAW_INPUT_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES


# --------------------------------------------------------------------------
# Feature engineering as a pipeline step
# --------------------------------------------------------------------------
FLAG_FEATURES = ["No_Units_1st", "No_Units_2nd"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the engineered features from main.ipynb, with two leakage /
    data-quality fixes:

    1. A student with 0 enrolled units has NO academic record (180 students,
       all in Course 2, 75 of whom graduated). The old formulas turned that
       into "0% approval / grade 0", i.e. it looked like total failure.
       Now rates and grades are left missing (filled by the imputer) and a
       No_Units flag tells the model the record is absent.
    2. Columns a stage does not use may be absent from the input; they are
       added as missing so the same code works for every stage.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for c in RAW_INPUT_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df[RAW_INPUT_COLUMNS] = df[RAW_INPUT_COLUMNS].astype(float)

        e1 = df["Curricular units 1st sem (enrolled)"]
        a1 = df["Curricular units 1st sem (approved)"]
        g1 = df["Curricular units 1st sem (grade)"]
        e2 = df["Curricular units 2nd sem (enrolled)"]
        a2 = df["Curricular units 2nd sem (approved)"]
        g2 = df["Curricular units 2nd sem (grade)"]

        has1, has2 = e1 > 0, e2 > 0
        df["No_Units_1st"] = (e1 == 0).astype(float).where(e1.notna())
        df["No_Units_2nd"] = (e2 == 0).astype(float).where(e2.notna())

        df["Approval_Rate_1st"] = (a1 / e1).where(has1)
        df["Approval_Rate_2nd"] = (a2 / e2).where(has2)
        tot = e1.fillna(0) + e2.fillna(0)
        df["Overall_Approval_Rate"] = (
            (a1.fillna(0) + a2.fillna(0)) / tot).where(tot > 0)
        df["Failed_Units_1st"] = np.maximum(e1 - a1, 0)
        df["Failed_Units_2nd"] = np.maximum(e2 - a2, 0)

        # grade 0 with no enrolled units = no record, not a failing grade
        df["Curricular units 1st sem (grade)"] = g1.where(has1)
        df["Curricular units 2nd sem (grade)"] = g2.where(has2)
        df["Average_Grade"] = pd.concat(
            [g1.where(has1), g2.where(has2)], axis=1).mean(axis=1)

        debt, fees = df["Debtor"], df["Tuition fees up to date"]
        risk = ((debt == 1) | (fees == 0)).astype(float)
        df["Financial_Risk"] = risk.where(debt.notna() | fees.notna())
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_INPUT_COLUMNS + ENGINEERED_FEATURES + FLAG_FEATURES,
                        dtype=object)


# --------------------------------------------------------------------------
# Code labels. The dataset copy used here renumbers the official UCI codes
# 1..N in ascending order (e.g. Course 171 -> 2). Verified for Course: the
# number of students per code matches the official course list exactly.
# Source of labels: UCI dataset 697 / Realinho et al. (2022), Appendix A.
# --------------------------------------------------------------------------
_COURSES = [
    "Biofuel Production Technologies", "Animation and Multimedia Design",
    "Social Service (evening)", "Agronomy", "Communication Design",
    "Veterinary Nursing", "Informatics Engineering", "Equinculture",
    "Management", "Social Service", "Tourism", "Nursing", "Oral Hygiene",
    "Advertising and Marketing Management", "Journalism and Communication",
    "Basic Education", "Management (evening)",
]
_APP_MODES = [
    "1st phase - general contingent", "Ordinance No. 612/93",
    "1st phase - special contingent (Azores)", "Holders of other higher courses",
    "Ordinance No. 854-B/99", "International student (bachelor)",
    "1st phase - special contingent (Madeira)", "2nd phase - general contingent",
    "3rd phase - general contingent", "Ordinance 533-A/99 b2 (different plan)",
    "Ordinance 533-A/99 b3 (other institution)", "Over 23 years old", "Transfer",
    "Change of course", "Technological specialization diploma holders",
    "Change of institution/course", "Short cycle diploma holders",
    "Change of institution/course (international)",
]
_PREV_QUAL = [
    "Secondary education", "Higher ed. - bachelor's degree", "Higher ed. - degree",
    "Higher ed. - master's", "Higher ed. - doctorate", "Frequency of higher education",
    "12th year - not completed", "11th year - not completed", "Other - 11th year",
    "10th year", "10th year - not completed", "Basic education 3rd cycle",
    "Basic education 2nd cycle", "Technological specialization course",
    "Higher ed. - degree (1st cycle)", "Professional higher technical course",
    "Higher ed. - master (2nd cycle)",
]
_MARITAL = ["Single", "Married", "Widower", "Divorced", "Facto union",
            "Legally separated"]
_NATIONALITY = [  # official codes 1, 2, 6, 11, ... 109
    "Portuguese", "German", "Spanish", "Italian", "Dutch", "English",
    "Lithuanian", "Angolan", "Cape Verdean", "Guinean", "Mozambican",
    "Santomean", "Turkish", "Brazilian", "Romanian", "Moldovan", "Mexican",
    "Ukrainian", "Russian", "Cuban", "Colombian",
]

# Education level of a parent - official codes in ascending order.
_EDU = {
    1: "Secondary school completed (12th year)",
    2: "Bachelor's degree",
    3: "College degree",
    4: "Master's degree",
    5: "Doctorate",
    6: "Some college (attended)",
    9: "12th year, not completed",
    10: "11th year, not completed",
    11: "7th year (old system)",
    12: "Other - 11th year",
    13: "2nd year complementary high school",
    14: "10th year",
    18: "General commerce course",
    19: "Up to 9th-11th year (basic education 3rd cycle)",
    20: "Complementary high school course",
    22: "Technical-vocational course",
    25: "Complementary high school, not completed",
    26: "7th year",
    27: "General high school, 2nd cycle",
    29: "9th year, not completed",
    30: "8th year",
    31: "Administration and commerce course",
    33: "Accounting and administration course",
    34: "Unknown",
    35: "Cannot read or write",
    36: "Can read, less than 4th year",
    37: "Up to 4th-5th year (basic education 1st cycle)",
    38: "Up to 6th-8th year (basic education 2nd cycle)",
    39: "Technological specialization course",
    40: "College degree (1st cycle)",
    41: "Specialized higher studies",
    42: "Professional higher technical course",
    43: "Master's degree (2nd cycle)",
    44: "Doctorate (3rd cycle)",
}
_MOTHER_EDU_CODES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27,
                     29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
_FATHER_EDU_CODES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22,
                     25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40,
                     41, 42, 43, 44]

# Occupation of a parent - official codes in ascending order (shortened).
_JOB = {
    0: "Student",
    1: "Legislator, director or executive manager",
    2: "Professional (scientific / intellectual work)",
    3: "Technician or associate professional",
    4: "Administrative / clerical staff",
    5: "Services, security or sales worker",
    6: "Farmer / skilled agriculture or fishery worker",
    7: "Skilled industry, construction or craft worker",
    8: "Machine operator or assembler",
    9: "Unskilled worker",
    10: "Armed forces",
    90: "Other situation",
    99: "Not specified",
    101: "Armed forces officer",
    102: "Armed forces sergeant",
    103: "Other armed forces personnel",
    112: "Director, administrative / commercial services",
    114: "Director, hotel / catering / trade services",
    121: "Engineer, scientist or mathematician",
    122: "Health professional",
    123: "Teacher",
    124: "Finance, accounting or administration specialist",
    125: "ICT specialist",
    131: "Science / engineering technician",
    132: "Health technician",
    134: "Legal, social, sports or cultural technician",
    135: "ICT technician",
    141: "Office worker / secretary / data processing",
    143: "Accounting, finance or registry operator",
    144: "Other administrative support",
    151: "Personal service worker",
    152: "Seller / salesperson",
    153: "Personal care worker",
    154: "Protection and security services",
    161: "Market-oriented farmer or animal producer",
    163: "Subsistence farmer, fisher or hunter",
    171: "Construction worker (except electrician)",
    172: "Metalworker",
    173: "Printing, precision instrument or craft worker",
    174: "Electrician / electronics worker",
    175: "Food, wood or clothing industry worker",
    181: "Fixed plant / machine operator",
    182: "Assembly worker",
    183: "Driver / mobile equipment operator",
    191: "Cleaning worker",
    192: "Unskilled farm, fishery or forestry worker",
    193: "Unskilled construction, manufacturing or transport worker",
    194: "Meal preparation assistant",
    195: "Street vendor / street services",
}
_MOTHER_JOB_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125,
                     131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173,
                     175, 191, 192, 193, 194]
_FATHER_JOB_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103,
                     112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141,
                     143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174,
                     175, 181, 182, 183, 192, 193, 194, 195]


def _ranked(codes, names):
    """Dataset code k = k-th official code (ascending)."""
    return {i: names[c] for i, c in enumerate(codes, 1)}


CODE_LABELS = {
    "Course": dict(enumerate(_COURSES, 1)),
    "Application mode": dict(enumerate(_APP_MODES, 1)),
    "Previous qualification": dict(enumerate(_PREV_QUAL, 1)),
    "Marital status": dict(enumerate(_MARITAL, 1)),
    "Nacionality": dict(enumerate(_NATIONALITY, 1)),
    "Mother's qualification": _ranked(_MOTHER_EDU_CODES, _EDU),
    "Father's qualification": _ranked(_FATHER_EDU_CODES, _EDU),
    "Mother's occupation": _ranked(_MOTHER_JOB_CODES, _JOB),
    "Father's occupation": _ranked(_FATHER_JOB_CODES, _JOB),
}
# Checked against the data: e.g. mothers' education codes 1/22/13/23/3/19 are
# the 6 most common (1069/1009/953/562/438/130 students) = Secondary /
# basic 1st cycle / basic 3rd cycle / basic 2nd cycle / degree / Unknown,
# matching the official distribution; nationality 14 = Brazilian (largest
# foreign group).

# Display order in the app: education from lowest to highest level,
# jobs in official group order; "Unknown"/"Not specified" go last.
_EDU_ORDER = [35, 36, 37, 11, 26, 30, 38, 29, 19, 14, 10, 12, 9, 1, 13, 20,
              25, 27, 18, 31, 33, 22, 39, 42, 6, 2, 3, 40, 41, 4, 43, 5, 44, 34]
_JOB_ORDER = [c for c in sorted(_JOB) if c not in (90, 99)] + [90, 99]


def _order(codes, official_order):
    rank = {c: i for i, c in enumerate(codes, 1)}   # official code -> data code
    return [rank[c] for c in official_order if c in rank]


OPTION_ORDER = {
    "Mother's qualification": _order(_MOTHER_EDU_CODES, _EDU_ORDER),
    "Father's qualification": _order(_FATHER_EDU_CODES, _EDU_ORDER),
    "Mother's occupation": _order(_MOTHER_JOB_CODES, _JOB_ORDER),
    "Father's occupation": _order(_FATHER_JOB_CODES, _JOB_ORDER),
}


def ordered_options(col, opts):
    order = OPTION_ORDER.get(col)
    if not order:
        return list(opts)
    pos = {c: i for i, c in enumerate(order)}
    return sorted(opts, key=lambda c: pos.get(c, len(pos)))


# Plain-language field names and help text for the app
FIELD_INFO = {
    "Course": ("Program / course", "The degree program the student enrolled in."),
    "Application mode": ("How the student applied", "Admission route, e.g. regular 1st phase, transfer, over-23 entry."),
    "Application order": ("Program choice rank", "Where this program was on the student's list of choices (1 = first choice, 0 = first choice in some records)."),
    "Daytime/evening attendance": ("Class schedule", "Daytime or evening classes."),
    "Previous qualification": ("Highest education before enrolling", "The student's own education level before this program."),
    "Tuition fees up to date": ("Tuition fees paid on time?", "Yes = the student has no overdue tuition."),
    "Debtor": ("Has unpaid debts to the school?", "Yes = the student owes the school money."),
    "Scholarship holder": ("Has a scholarship?", "Yes = the student receives a scholarship."),
    "Age at enrollment": ("Age when enrolled", "Age in years on enrollment."),
    "Gender": ("Sex", "As recorded in the dataset."),
    "Marital status": ("Civil status", ""),
    "Nacionality": ("Nationality", ""),
    "International": ("International student?", "Yes = foreign student."),
    "Displaced": ("Lives away from home?", "Yes = moved away from home to study."),
    "Educational special needs": ("Has special educational needs?", "Yes = requires learning or accessibility support."),
    "Mother's qualification": ("Mother's highest education", "Highest level of schooling the student's mother completed. Choose 'Unknown' if not known."),
    "Father's qualification": ("Father's highest education", "Highest level of schooling the student's father completed. Choose 'Unknown' if not known."),
    "Mother's occupation": ("Mother's job", "Type of work the mother does. Choose 'Not specified' or 'Other situation' if unsure."),
    "Father's occupation": ("Father's job", "Type of work the father does. Choose 'Not specified' or 'Other situation' if unsure."),
    "Unemployment rate": ("Unemployment rate (%)", "National unemployment rate in the year of enrollment."),
    "Inflation rate": ("Inflation rate (%)", "National inflation rate in the year of enrollment."),
    "GDP": ("GDP growth (%)", "National economic growth in the year of enrollment."),
}
for _s in ("1st", "2nd"):
    FIELD_INFO.update({
        f"Curricular units {_s} sem (enrolled)": ("Subjects enrolled", f"Number of subjects (curricular units) taken in the {_s} semester."),
        f"Curricular units {_s} sem (approved)": ("Subjects passed", f"Number of subjects passed in the {_s} semester."),
        f"Curricular units {_s} sem (grade)": ("Average grade (0-20)", f"Average grade of the {_s} semester on a 0-20 scale (10 = passing)."),
        f"Curricular units {_s} sem (evaluations)": ("Exams / assessments taken", f"Number of evaluations the student sat in the {_s} semester."),
        f"Curricular units {_s} sem (credited)": ("Subjects credited", f"Subjects credited from previous studies in the {_s} semester."),
        f"Curricular units {_s} sem (without evaluations)": ("Subjects with no exam taken", f"Subjects enrolled in the {_s} semester where the student took no evaluation."),
    })


def field_label(col):
    return FIELD_INFO.get(col, (col, ""))[0]


def field_help(col):
    return FIELD_INFO.get(col, (col, ""))[1] or None


# --------------------------------------------------------------------------
# Assessment stages: which information exists at each point in time
# --------------------------------------------------------------------------
_ECON = ["Unemployment rate", "Inflation rate", "GDP"]
_FIN_STATUS = ["Debtor", "Tuition fees up to date"]
_SEM = lambda s: [  # noqa: E731
    f"Curricular units {s} sem (credited)", f"Curricular units {s} sem (enrolled)",
    f"Curricular units {s} sem (evaluations)", f"Curricular units {s} sem (approved)",
    f"Curricular units {s} sem (grade)", f"Curricular units {s} sem (without evaluations)",
]

STAGES = {
    "enrollment": {
        "label": "Stage 1 - At enrollment",
        "short": "At enrollment",
        "description": "Background, program, socio-economic and financial "
                       "data known at enrollment (Realinho et al., 2022, "
                       "Table 1) - no grades yet.",
        "numeric": ["Application order", "Age at enrollment"] + _ECON
                   + ["Financial_Risk"],
        "categorical": list(CATEGORICAL_FEATURES),
        "activity_col": None,
    },
    "semester1": {
        "label": "Stage 2 - After 1st semester",
        "short": "After 1st semester",
        "description": "Adds 1st-semester academic results.",
        "numeric": ["Application order", "Age at enrollment"] + _ECON + _SEM("1st")
                   + ["Approval_Rate_1st", "Failed_Units_1st", "No_Units_1st",
                      "Financial_Risk"],
        "categorical": list(CATEGORICAL_FEATURES),
        "activity_col": "Curricular units 1st sem (evaluations)",
    },
    "year1": {
        "label": "Stage 3 - After 1st year",
        "short": "After 1st year",
        "description": "Adds 2nd-semester results (most accurate, but latest).",
        "numeric": ["Application order", "Age at enrollment"] + _ECON
                   + _SEM("1st") + _SEM("2nd")
                   + ["Approval_Rate_1st", "Failed_Units_1st", "No_Units_1st",
                      "Approval_Rate_2nd", "Failed_Units_2nd", "No_Units_2nd",
                      "Overall_Approval_Rate", "Average_Grade", "Financial_Risk"],
        "categorical": list(CATEGORICAL_FEATURES),
        "activity_col": "Curricular units 2nd sem (evaluations)",
    },
}
STAGE_ORDER = ["enrollment", "semester1", "year1"]


def stage_inputs(stage_key):
    """Raw columns a user must supply for a stage."""
    st_ = STAGES[stage_key]
    used = set(st_["numeric"]) | set(st_["categorical"])
    needs_sem1 = any("1st" in c for c in used)
    needs_sem2 = any("2nd" in c for c in used)
    cols = []
    for c in RAW_INPUT_COLUMNS:
        if c in used or (needs_sem1 and "1st sem" in c) or \
                (needs_sem2 and "2nd sem" in c) or \
                (c in _FIN_STATUS and "Financial_Risk" in used):
            cols.append(c)
    return cols


def disengagement_alert(row, stage_key):
    """Rule check outside the model: enrolled in units but took no
    evaluations in the latest semester = has very likely stopped attending
    (in the data, 220 of 221 such students dropped out)."""
    sem = {"semester1": "1st", "year1": "2nd"}.get(stage_key)
    if sem is None:
        return False
    enrolled = row.get(f"Curricular units {sem} sem (enrolled)", np.nan)
    evals = row.get(f"Curricular units {sem} sem (evaluations)", np.nan)
    return bool(pd.notna(enrolled) and pd.notna(evals) and enrolled > 0 and evals == 0)


# --------------------------------------------------------------------------
# XGBoost that balances classes by itself
# --------------------------------------------------------------------------
from sklearn.base import ClassifierMixin  # noqa: E402


class BalancedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost with 'balanced' class weights computed inside fit().

    Plain XGBClassifier needs sample_weight passed to fit(), which does not
    work inside StackingClassifier/VotingClassifier. This wrapper makes it
    behave like RandomForest(class_weight="balanced").
    """

    def __init__(self, n_estimators=500, max_depth=3, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        params = dict(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state, n_jobs=self.n_jobs,
        )
        if n_cls > 2:
            params.update(objective="multi:softprob", num_class=n_cls,
                          eval_metric="mlogloss")
        else:
            params.update(objective="binary:logistic", eval_metric="logloss")
        self.model_ = XGBClassifier(**params)
        self.model_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_booster(self):
        return self.model_.get_booster()


# --------------------------------------------------------------------------
# Risk levels
# --------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"medium": 0.40, "high": 0.70}


def risk_level(prob, thresholds=DEFAULT_THRESHOLDS):
    if prob < thresholds["medium"]:
        return "Low Risk"
    if prob < thresholds["high"]:
        return "Medium Risk"
    return "High Risk"


# --------------------------------------------------------------------------
# Friendly labels for explanations
# --------------------------------------------------------------------------
FRIENDLY_NAMES = {
    "Overall_Approval_Rate": "Overall approval rate (passed / enrolled units)",
    "Approval_Rate_1st": "1st-semester approval rate",
    "Approval_Rate_2nd": "2nd-semester approval rate",
    "Failed_Units_1st": "Failed units, 1st semester",
    "Failed_Units_2nd": "Failed units, 2nd semester",
    "Average_Grade": "Average grade (both semesters)",
    "Financial_Risk": "Financial risk flag (debtor or unpaid tuition)",
    "Tuition fees up to date": "Tuition fees up to date",
    "Debtor": "Has outstanding debt",
    "Scholarship holder": "Scholarship holder",
    "Age at enrollment": "Age at enrollment",
    "No_Units_1st": "No 1st-semester academic record",
    "No_Units_2nd": "No 2nd-semester academic record",
}

# Which support office or action fits each factor (used only when the
# factor is pushing the student's dropout risk UP).
INTERVENTIONS = {
    "academic": (
        "Academic support",
        "Refer to tutoring / academic adviser; review failed subjects and "
        "agree on a recovery study plan.",
    ),
    "grades": (
        "Academic coaching",
        "Monitor grades closely; offer study-skills coaching or peer mentoring.",
    ),
    "financial": (
        "Financial assistance",
        "Refer to the financial aid / scholarship office; discuss a tuition "
        "payment plan or available grants.",
    ),
    "life": (
        "Working / adult-learner support",
        "Discuss schedule flexibility, workload and personal commitments "
        "with a guidance counselor.",
    ),
    "program": (
        "Program-fit counseling",
        "Meet with the department adviser to check whether the program "
        "matches the student's goals.",
    ),
    "wellbeing": (
        "Student affairs / guidance",
        "Check housing, adjustment and accessibility needs; refer to the "
        "guidance office.",
    ),
    "background": (
        "First-generation / family support",
        "Guidance counselor follow-up on family and background support.",
    ),
}

FEATURE_TO_INTERVENTION = {
    # academic progress
    "Overall_Approval_Rate": "academic", "Approval_Rate_1st": "academic",
    "Approval_Rate_2nd": "academic", "Failed_Units_1st": "academic",
    "Failed_Units_2nd": "academic",
    "Curricular units 1st sem (approved)": "academic",
    "Curricular units 2nd sem (approved)": "academic",
    "Curricular units 1st sem (enrolled)": "academic",
    "Curricular units 2nd sem (enrolled)": "academic",
    "Curricular units 1st sem (evaluations)": "academic",
    "Curricular units 2nd sem (evaluations)": "academic",
    "Curricular units 1st sem (without evaluations)": "academic",
    "Curricular units 2nd sem (without evaluations)": "academic",
    "Curricular units 1st sem (credited)": "academic",
    "Curricular units 2nd sem (credited)": "academic",
    "No_Units_1st": "academic", "No_Units_2nd": "academic",
    # grades
    "Average_Grade": "grades",
    "Curricular units 1st sem (grade)": "grades",
    "Curricular units 2nd sem (grade)": "grades",
    # financial
    "Financial_Risk": "financial", "Tuition fees up to date": "financial",
    "Debtor": "financial", "Scholarship holder": "financial",
    # life situation
    "Age at enrollment": "life", "Marital status": "life",
    "Daytime/evening attendance": "life",
    # program choice
    "Course": "program", "Application mode": "program",
    "Application order": "program", "Previous qualification": "program",
    # wellbeing
    "Displaced": "wellbeing", "International": "wellbeing",
    "Educational special needs": "wellbeing", "Nacionality": "wellbeing",
    # background
    "Mother's qualification": "background", "Father's qualification": "background",
    "Mother's occupation": "background", "Father's occupation": "background",
    # Gender, Unemployment rate, Inflation rate, GDP: not actionable -> no entry
}

DISENGAGEMENT_ACTION = (
    "Enrolled in units but took no evaluations this semester - the student "
    "may have already stopped attending. Contact the student this week."
)

GENERAL_ACTION = {
    "High Risk": "Schedule a guidance counselor meeting within 1 week.",
    "Medium Risk": "Adviser check-in within the month; monitor midterm results.",
    "Low Risk": "Regular monitoring; no immediate action needed.",
}


def friendly(name):
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]
    if name.startswith("Curricular units"):
        sem = "1st" if "1st" in name else "2nd"
        return f"{field_label(name)} ({sem} sem)"
    return field_label(name)


# --------------------------------------------------------------------------
# Prediction + explanation for one stage model (see train_final.py)
#   `model` = bundle["stages"][stage_key]
# --------------------------------------------------------------------------
def _complete(X_raw):
    X = pd.DataFrame(X_raw).copy()
    for c in RAW_INPUT_COLUMNS:
        if c not in X.columns:
            X[c] = np.nan
    return X[RAW_INPUT_COLUMNS]


def predict_dropout(model, X_raw):
    """Dropout probability (0-1) for each student."""
    X = _complete(X_raw)
    w = model["rf_weight"]
    p_rf = model["rf_pipeline"].predict_proba(X)[:, 1]
    p_xgb = model["xgb_pipeline"].predict_proba(X)[:, 1]
    return w * p_rf + (1 - w) * p_xgb


def dropout_contributions(model, X_raw):
    """Per-student contribution of every ORIGINAL feature to the dropout
    score, from XGBoost's built-in TreeSHAP (pred_contribs). Positive values
    push toward dropout, negative values away from it."""
    import xgboost as xgb

    X = _complete(X_raw)
    pipe = model["xgb_pipeline"]
    Xt = pipe[:-1].transform(X)
    contribs = pipe[-1].get_booster().predict(
        xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)
    if contribs.ndim == 3:            # multi-class layout, take dropout class
        contribs = contribs[:, -1, :]
    contribs = contribs[:, :-1]       # drop bias column
    out = {name: contribs[:, idx].sum(axis=1) for name, idx in model["feature_groups"]}
    return pd.DataFrame(out, index=X.index)


def explain_student(contrib_row, top_n=5):
    s = contrib_row.sort_values(ascending=False)
    up = s[s > 0].head(top_n)
    down = s[s < 0].sort_values().head(3)
    return up, down


def recommend(level, risk_factors, alert=False):
    recs = []
    if alert:
        recs.append(("Immediate contact", DISENGAGEMENT_ACTION))
    recs.append(("Follow-up", GENERAL_ACTION[level]))
    if level == "Low Risk" and not alert:
        return recs
    seen = set()
    for feat in risk_factors.index:
        key = FEATURE_TO_INTERVENTION.get(feat)
        if key and key not in seen:
            seen.add(key)
            recs.append(INTERVENTIONS[key])
    return recs


def score_frame(model, stage_key, X_raw, explain=True):
    """Score many students at once (batch page)."""
    th = model["thresholds"]
    p = predict_dropout(model, X_raw)
    res = pd.DataFrame(index=X_raw.index)
    res["Dropout_Probability"] = p.round(3)
    res["Risk_Level"] = [risk_level(v, th) for v in p]
    res["Disengagement_Alert"] = [
        disengagement_alert(r, stage_key) for _, r in X_raw.iterrows()]
    if explain:
        contrib = dropout_contributions(model, X_raw)
        res["Top_Risk_Factors"] = [
            "; ".join(friendly(f) for f in explain_student(contrib.iloc[i], 3)[0].index)
            for i in range(len(contrib))]
    return res
