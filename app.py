import streamlit as st
import pandas as pd

from config import (
    DEFAULT_THRESHOLD, COLOR_DROPOUT_RISK, COLOR_PROBATION_2,
    COLOR_PROBATION_1, COLOR_GOOD,
    STATUS_GOOD, STATUS_PROBATION_1, STATUS_PROBATION_2, STATUS_DROPOUT_RISK,
)
from database import init_db, insert_student, get_all_students, update_advisor_override
from utils.ml_pipeline import evaluate_student
from utils.visualization import plot_feature_contributions, plot_status_distribution

st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")

# Make sure tables exist before anything else runs.
init_db()

st.title("🎓 Student Dropout Risk Predictor")

STATUS_COLORS = {
    STATUS_GOOD: COLOR_GOOD,
    STATUS_PROBATION_1: COLOR_PROBATION_1,
    STATUS_PROBATION_2: COLOR_PROBATION_2,
    STATUS_DROPOUT_RISK: COLOR_DROPOUT_RISK,
}

left, right = st.columns([1, 1])

# ----------------------------------------------------------------------
# SECTION 1: Student Input Form
# ----------------------------------------------------------------------
with left:
    st.subheader("Evaluate a Student")
    with st.form("student_form"):
        student_name = st.text_input("Student Name", value=" ")
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 0.0)
        marks = st.slider("Average Marks (%)", 0.0, 100.0, 0.0)
        gpa = st.number_input("GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=0.0, step=0.1)
        failures = st.number_input("Number of Failed Subjects", min_value=0, max_value=20, value=0, step=1)
        backlogs = st.number_input("Number of Backlog Subjects", min_value=0, max_value=20, value=0, step=1)
        submitted = st.form_submit_button("⚡ Evaluate Student")

    if submitted:
        features = {
            "attendance_percentage": attendance,
            "average_marks": marks,
            "gpa": gpa,
            "failures_count": failures,
            "backlog_subjects": backlogs,
        }
        result = evaluate_student(features)

        insert_student({
            "student_name": student_name,
            "attendance_percentage": attendance,
            "average_marks": marks,
            "gpa": gpa,
            "failures_count": failures,
            "backlog_subjects": backlogs,
            "risk_score": result["risk_score"],
            "status": result["status"],
            "main_reason": result["main_reason"],
            "recommendation": result["recommendation"],
        })

        badge_color = STATUS_COLORS[result["status"]]
        st.markdown(
            f"**Status:** <span style='color:{badge_color}; font-weight:700'>"
            f"{result['status']}</span> — P(Dropout) = {result['risk_score']:.2%}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Main reason for risk:** {result['main_reason']}")

        # ------------------------------------------------------------
        # Suggestions — one bullet per risk-increasing factor, ranked
        # by severity, not just the single top factor.
        # ------------------------------------------------------------
        st.markdown("**Suggestions to reach Good Standing:**")
        for tip in result["suggestions"]:
            st.markdown(f"- {tip}")

        st.session_state["last_features"] = features
        st.session_state["last_contributions"] = result["contributions"]

# ----------------------------------------------------------------------
# SECTION 2: Cohort Summary
# ----------------------------------------------------------------------
df_all = get_all_students()

with right:
    st.subheader("Cohort Summary")
    total = len(df_all)
    if total > 0:
        status_counts = df_all["status"].value_counts().to_dict()
        # keep a stable, meaningful tier order rather than count order
        ordered = {s: status_counts.get(s, 0) for s in
                   [STATUS_GOOD, STATUS_PROBATION_1, STATUS_PROBATION_2, STATUS_DROPOUT_RISK]}

        c1, c2 = st.columns(2)
        c1.metric("Total Students Evaluated", f"{total:,}")
        c1.metric("At Risk of Dropout", f"{ordered[STATUS_DROPOUT_RISK]:,}")
        c2.metric("On Probation (1st + 2nd)", f"{ordered[STATUS_PROBATION_1] + ordered[STATUS_PROBATION_2]:,}")
        c2.metric("Good Standing", f"{ordered[STATUS_GOOD]:,}")

        st.pyplot(plot_status_distribution(ordered))
    else:
        st.info("No students logged yet. Submit the form on the left to get started.")

# ----------------------------------------------------------------------
# SECTION 3: Student Records Table & Advisor Override
# ----------------------------------------------------------------------
st.subheader("Student Records & Advisor Override")

if total > 0:
    fcol1, fcol2 = st.columns(2)
    search_name = fcol1.text_input("Search Student Name", value="")
    filter_status = fcol2.selectbox(
        "Filter Status", ["ALL", STATUS_GOOD, STATUS_PROBATION_1, STATUS_PROBATION_2, STATUS_DROPOUT_RISK]
    )

    view_df = df_all.copy()
    if search_name:
        view_df = view_df[view_df["student_name"].str.contains(search_name, case=False, na=False)]
    if filter_status != "ALL":
        view_df = view_df[view_df["status"] == filter_status]

    st.dataframe(
        view_df[["id", "student_name", "attendance_percentage", "average_marks",
                 "gpa", "risk_score", "status", "main_reason", "created_at"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Advisor Override** — manually move a student to a different standing tier")
    oc1, oc2 = st.columns([2, 2])
    row_id = oc1.number_input("Student Record ID to override", min_value=1, step=1)
    new_status = oc2.selectbox(
        "Set Status To", [STATUS_GOOD, STATUS_PROBATION_1, STATUS_PROBATION_2, STATUS_DROPOUT_RISK],
        key="override_select"
    )
    if st.button("✅ Apply Override"):
        update_advisor_override(int(row_id), new_status)
        st.success(f"Record {row_id} status overridden to '{new_status}'.")
        st.rerun()

    # ------------------------------------------------------------------
    # SECTION 3b: View stored suggestions for any past record
    # ------------------------------------------------------------------
    st.markdown("**View stored suggestions** for a past record")
    lookup_id = st.number_input("Student Record ID to view suggestions for",
                                 min_value=1, step=1, key="lookup_id")
    if st.button("🔍 Show Suggestions"):
        match = df_all[df_all["id"] == int(lookup_id)]
        if match.empty:
            st.warning(f"No record with ID {lookup_id}.")
        else:
            stored = match.iloc[0]["recommendation"]
            st.markdown(f"**Suggestions for record {int(lookup_id)}:**")
            for tip in str(stored).split(". "):
                tip = tip.strip()
                if tip:
                    st.markdown(f"- {tip.rstrip('.')}.")

    # ------------------------------------------------------------------
    # SECTION 4: Explainability Chart
    # ------------------------------------------------------------------
    st.subheader("Model Explainability — Feature Contribution Chart")
    if "last_contributions" in st.session_state:
        st.pyplot(plot_feature_contributions(st.session_state["last_contributions"]))
    else:
        st.caption("Evaluate a student above to see their feature contribution chart.")