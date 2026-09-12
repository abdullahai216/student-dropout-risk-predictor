CREATE DATABASE IF NOT EXISTS student_dropout_predictor
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE student_dropout_predictor;

CREATE TABLE IF NOT EXISTS students (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    student_name            VARCHAR(100) NOT NULL,
    attendance_percentage   DOUBLE       NOT NULL,
    average_marks           DOUBLE       NOT NULL,
    gpa                     DOUBLE       NOT NULL,
    failures_count          INT          NOT NULL,
    backlog_subjects        INT          NOT NULL,
    risk_score               DOUBLE      NOT NULL,
    status                   VARCHAR(40) NOT NULL DEFAULT 'Good Standing',
    main_reason              VARCHAR(50) DEFAULT NULL,
    recommendation            TEXT       DEFAULT NULL,
    advisor_override          VARCHAR(40) DEFAULT NULL,
    created_at                TIMESTAMP  DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS model_metadata (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    model_version       VARCHAR(50)  NOT NULL,
    accuracy            DOUBLE       NOT NULL,
    precision_score     DOUBLE       NOT NULL,
    recall_score        DOUBLE       NOT NULL,
    auc_roc             DOUBLE       NOT NULL,
    trained_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE INDEX idx_students_status  ON students(status);
CREATE INDEX idx_students_created ON students(created_at);

-- Sanity check
SHOW TABLES;
