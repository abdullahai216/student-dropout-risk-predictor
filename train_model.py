"""
train_model.py
Loads the real UCI "Predict Students' Dropout and Academic Success"
dataset (4,424 students, Realinho et al. 2021, CC BY 4.0) via the
official ucimlrepo package, derives the five FEATURE_COLUMNS from its
raw columns, trains a Logistic Regression + StandardScaler pipeline,
saves both artifacts to models/, and logs the resulting metrics into
the MySQL 'model_metadata' table.

Dataset source:
    https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Why ucimlrepo instead of a direct URL: some networks (campus wifi,
corporate firewalls, Cloudflare bot-checks) return an HTML interstitial
page instead of the actual file when you hit archive.ics.uci.edu
directly — pandas then silently reads that HTML as the CSV, and every
column name comes out wrong. ucimlrepo fetches through UCI's own API,
which sidesteps that.

Before running this, make sure ucimlrepo is installed in this venv:
    pip install ucimlrepo

Run this once before starting the Streamlit app:
    python train_model.py
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

from config import MODELS_DIR, DATA_DIR, MODEL_PATH, SCALER_PATH, SYNTHETIC_DATA_PATH
from database import init_db, save_model_metadata

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load the real dataset via ucimlrepo
# ---------------------------------------------------------------------
dataset = fetch_ucirepo(id=697)
X_raw = dataset.data.features
y_raw = dataset.data.targets
raw = pd.concat([X_raw, y_raw], axis=1)

print(f"Downloaded {raw.shape[0]} rows, {raw.shape[1]} columns.")

# Fail loudly and clearly instead of a cryptic KeyError several lines
# down, if UCI ever renames a column this script depends on.
required_cols = [
    "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 1st sem (evaluations)", "Curricular units 2nd sem (evaluations)",
    "Target",
]
missing = [c for c in required_cols if c not in raw.columns]
if missing:
    raise KeyError(
        f"Expected column(s) not found: {missing}\n"
        f"Actual columns in the downloaded dataset:\n{list(raw.columns)}"
    )

# ---------------------------------------------------------------------
# 2. Derive FEATURE_COLUMNS from the dataset's real columns
# ---------------------------------------------------------------------
# average_marks: mean of 1st/2nd semester grades, which are on a 0-20
# scale in the Portuguese system -> convert to a 0-100 percentage.
avg_grade_20 = raw[["Curricular units 1st sem (grade)",
                     "Curricular units 2nd sem (grade)"]].mean(axis=1)
average_marks = (avg_grade_20 * 5).clip(0, 100)

# gpa: same average grade, rescaled to a 0-4 scale.
gpa = (avg_grade_20 / 20 * 4).clip(0.0, 4.0)

# failures_count: units a student enrolled in but did not pass,
# summed across both semesters.
failures_count = (
    (raw["Curricular units 1st sem (enrolled)"] - raw["Curricular units 1st sem (approved)"])
    + (raw["Curricular units 2nd sem (enrolled)"] - raw["Curricular units 2nd sem (approved)"])
).clip(lower=0)

# backlog_subjects: units enrolled but never evaluated at all (a
# reasonable proxy for "fell behind" / carried-over coursework).
backlog_subjects = (
    raw["Curricular units 1st sem (without evaluations)"]
    + raw["Curricular units 2nd sem (without evaluations)"]
).clip(lower=0)

# attendance_percentage: the dataset has no literal daily-attendance
# log (few public datasets do), so this is a proxy — the share of
# enrolled units a student actually sat evaluations for. Swap this
# for a real attendance column if your institution tracks one.
enrolled_total = (
    raw["Curricular units 1st sem (enrolled)"] + raw["Curricular units 2nd sem (enrolled)"]
).replace(0, 1)  # avoid divide-by-zero
evaluated_total = (
    raw["Curricular units 1st sem (evaluations)"] + raw["Curricular units 2nd sem (evaluations)"]
)
attendance_percentage = ((evaluated_total / enrolled_total) * 100).clip(0, 100)

# target: collapse the original 3-class label (Dropout / Enrolled /
# Graduate) into the binary target this pipeline predicts on.
dropout_risk = (raw["Target"] == "Dropout").astype(int)

df = pd.DataFrame({
    "attendance_percentage": attendance_percentage,
    "average_marks": average_marks,
    "gpa": gpa,
    "failures_count": failures_count,
    "backlog_subjects": backlog_subjects,
    "dropout_risk": dropout_risk,
})
df.to_csv(SYNTHETIC_DATA_PATH, index=False)  # cached local copy for reuse/inspection
print(f"Derived {len(df)} student records with 5 features + target.")
print(df["dropout_risk"].value_counts(normalize=True).rename("proportion"))

# ---------------------------------------------------------------------
# 3. Train Logistic Regression pipeline
# ---------------------------------------------------------------------
X = df[["attendance_percentage", "average_marks", "gpa", "failures_count", "backlog_subjects"]]
y = df["dropout_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

# ---------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved scaler to {SCALER_PATH}")

# ---------------------------------------------------------------------
# 6. Log metadata to MySQL
# ---------------------------------------------------------------------
init_db()
save_model_metadata("v1.0.0", acc, prec, rec, auc)
print("Logged model metadata to MySQL 'model_metadata' table.")