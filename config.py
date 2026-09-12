"""
config.py
Central place for application constants, file paths, MySQL connection
settings, and the academic thresholds used to classify a student's
standing. Edit DB_CONFIG to match your local MySQL Workbench / MySQL
Server credentials.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "dropout_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
# Cached local copy of the derived features pulled from the real UCI
# dataset in train_model.py (kept for inspection/reuse — not synthetic).
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "uci_students_derived.csv")

# ---------------------------------------------------------------------
# MySQL connection settings
# Update these four values to match the server you created in
# MySQL Workbench (Database > Manage Connections).
# ---------------------------------------------------------------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "hr123",
    "database": "student_dropout_predictor",
}

# ---------------------------------------------------------------------
# Feature order — MUST match the column order used in train_model.py
# and the order the scaler was fit on.
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "attendance_percentage",
    "average_marks",
    "gpa",
    "failures_count",
    "backlog_subjects",
]

# ---------------------------------------------------------------------
# Risk-score cutoffs used to classify a student into an academic
# standing category. risk_score is P(dropout) from the model, 0.0-1.0.
# ---------------------------------------------------------------------
GOOD_STANDING_MAX = 0.25       # below this -> Good Standing
FIRST_PROBATION_MAX = 0.50     # below this -> First Probation
SECOND_PROBATION_MAX = 0.75    # below this -> Second Probation
# anything >= SECOND_PROBATION_MAX -> At Risk of Dropout / Cease

STATUS_GOOD = "Good Standing"
STATUS_PROBATION_1 = "First Probation"
STATUS_PROBATION_2 = "Second Probation"
STATUS_DROPOUT_RISK = "At Risk of Dropout / Cease"

# ---------------------------------------------------------------------
# Recovery targets — used to generate "what to improve" recommendations.
# These are reasonable defaults; tune them to match your institution's
# actual passing/retention requirements.
# ---------------------------------------------------------------------
ATTENDANCE_TARGET = 75.0   # percent
MARKS_TARGET = 50.0        # percent
GPA_TARGET = 2.0           # on a 4.0 scale

# ---------------------------------------------------------------------
# UI color standards
# ---------------------------------------------------------------------
COLOR_DROPOUT_RISK = "#E53E3E"     # Crimson Red
COLOR_PROBATION_2 = "#DD6B20"      # Amber
COLOR_PROBATION_1 = "#D69E2E"      # Gold
COLOR_GOOD = "#38A169"             # Emerald Green

DEFAULT_THRESHOLD = 0.50
