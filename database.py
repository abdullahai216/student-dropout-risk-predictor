import mysql.connector
import pandas as pd
from config import DB_CONFIG


def get_connection():
    """Open and return a new MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """
    Create the 'students' and 'model_metadata' tables (and their
    indexes) if they do not already exist. Safe to call every time the
    app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_name VARCHAR(100) NOT NULL,
            attendance_percentage DOUBLE NOT NULL,
            average_marks DOUBLE NOT NULL,
            gpa DOUBLE NOT NULL,
            failures_count INT NOT NULL,
            backlog_subjects INT NOT NULL,
            risk_score DOUBLE NOT NULL,
            status VARCHAR(40) NOT NULL DEFAULT 'Good Standing',
            main_reason VARCHAR(50) DEFAULT NULL,
            recommendation TEXT DEFAULT NULL,
            advisor_override VARCHAR(40) DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_version VARCHAR(50) NOT NULL,
            accuracy DOUBLE NOT NULL,
            precision_score DOUBLE NOT NULL,
            recall_score DOUBLE NOT NULL,
            auc_roc DOUBLE NOT NULL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
    """)

    # MySQL has no "CREATE INDEX IF NOT EXISTS" before v8.0.29 in all
    # editions, so guard with a check against information_schema.
    cursor.execute("""
        SELECT COUNT(1) FROM information_schema.statistics
        WHERE table_schema = %s AND table_name = 'students'
              AND index_name = 'idx_students_status'
    """, (DB_CONFIG["database"],))
    if cursor.fetchone()[0] == 0:
        cursor.execute("CREATE INDEX idx_students_status ON students(status);")

    cursor.execute("""
        SELECT COUNT(1) FROM information_schema.statistics
        WHERE table_schema = %s AND table_name = 'students'
              AND index_name = 'idx_students_created'
    """, (DB_CONFIG["database"],))
    if cursor.fetchone()[0] == 0:
        cursor.execute("CREATE INDEX idx_students_created ON students(created_at);")

    conn.commit()
    cursor.close()
    conn.close()


def insert_student(data_dict):
    """
    Insert one evaluated student record. data_dict must contain:
    student_name, attendance_percentage, average_marks, gpa,
    failures_count, backlog_subjects, risk_score, status,
    main_reason, recommendation
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students
            (student_name, attendance_percentage, average_marks, gpa,
             failures_count, backlog_subjects, risk_score, status,
             main_reason, recommendation)
        VALUES (%(student_name)s, %(attendance_percentage)s, %(average_marks)s,
                %(gpa)s, %(failures_count)s, %(backlog_subjects)s,
                %(risk_score)s, %(status)s, %(main_reason)s, %(recommendation)s)
    """, data_dict)
    conn.commit()
    last_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return last_id


def get_all_students():
    """Return every student record as a pandas DataFrame, newest first."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM students ORDER BY id DESC", conn)
    conn.close()
    return df


def update_advisor_override(student_id, override_status):
    """
    override_status is a status string (e.g. 'First Probation') or None.

    FIX: this previously wrote ONLY to the advisor_override column,
    which nothing in app.py ever reads (the cohort metrics, chart, and
    records table all read the `status` column) — so overrides had no
    visible effect anywhere in the UI. Now it also writes the new
    value into `status` itself, which is what's actually displayed;
    `advisor_override` is kept alongside it purely as an audit flag
    ("this record's status was manually set by an advisor, not the
    model") in case you want to filter/report on overridden records
    later.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE students SET status = %s, advisor_override = %s WHERE id = %s",
        (override_status, override_status, student_id),
    )
    conn.commit()
    cursor.close()
    conn.close()


def save_model_metadata(model_version, accuracy, precision, recall, auc_roc):
    """Log training metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_metadata
            (model_version, accuracy, precision_score, recall_score, auc_roc)
        VALUES (%s, %s, %s, %s, %s)
    """, (model_version, accuracy, precision, recall, auc_roc))
    conn.commit()
    cursor.close()
    conn.close()