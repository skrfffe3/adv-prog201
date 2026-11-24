# HOPE – Explainable Multi-Stage Early Warning System for Student Dropout

A Streamlit dashboard that predicts each student's dropout risk at three
points (at enrollment, after the 1st semester, after the 1st year),
explains the prediction, and recommends interventions. Developed and
validated on the public UCI *Predict Students' Dropout and Academic
Success* dataset (Realinho et al., 2022).

## Quick setup (new computer)

Requirements: [Git](https://git-scm.com/) and Python 3.15
([python.org](https://www.python.org/downloads/)) – use the same Python
version as the rest of the team.

**Windows**

```
git clone <repo-url>
cd ADRIPS
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_final.py
streamlit run app.py
```

**Mac / Linux**

```
git clone <repo-url>
cd ADRIPS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_final.py
streamlit run app.py
```

The app opens at http://localhost:8501.

- `python train_final.py` (about 4 minutes) creates
  `models/adrips_model.joblib`. Skip it if that file is already in your
  copy. Results are identical on every computer (fixed random seeds).
- Always install with `pip install -r requirements.txt`: the saved model
  only loads with the pinned scikit-learn and XGBoost versions.
- Each person creates their own `venv/`; it is not stored in the repository.

## Running it again later

```
cd ADRIPS
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
streamlit run app.py
```

Stop the app with **Ctrl + C** in the terminal.

## Project files

```
ADRIPS/
├── app.py                 Streamlit dashboard
├── adrips_core.py         shared logic: features, stages, labels, explanations
├── train_final.py         trains the 3 stage models → models/ and reports/
├── experiments.py         comparison of 12 modelling approaches
├── main.ipynb             original exploration notebook
├── requirements.txt       pinned package versions
├── dataset/
│   └── predict-student-dropout-and-academic-success.csv
├── models/                created by train_final.py
└── reports/               training and experiment reports
```

## Using the app

- Pick the **assessment stage** in the sidebar; the form only asks for the
  data available at that stage.
- **Assess a student** – risk level, dropout probability, risk across
  stages, the factors behind the prediction, a disengagement alert and
  recommended interventions.
- **Batch screening** – upload a CSV with the dataset's column names →
  students ranked by dropout probability, downloadable.
- **Model information** – test-set results for each stage.

## Re-running the experiments

```
python experiments.py              (about 10 minutes; optional: pip install lightgbm)
python train_final.py --quick      (1-minute smoke test, numbers not for reporting)
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `'streamlit' is not recognized` | The venv is not active – run `venv\Scripts\activate` first. |
| "Model file not found" in the app | Run `python train_final.py`. |
| "Model file is from the older version" | Run `python train_final.py` again. |
| Errors loading the model after installing packages | Reinstall with `pip install -r requirements.txt` (versions must match). |
| `pip install` fails on a package | Check that you use the same Python version as the team. |
| Activation blocked on Windows PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
