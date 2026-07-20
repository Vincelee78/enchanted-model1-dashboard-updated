"""Streamlit frontend for the ENCHANTED Model 1 dashboard.

This module owns everything the user sees or clicks:
- page styling and header
- authenticated user's top-right identity display
- dashboard tabs
- task queues
- patient list, filters, and table
- patient review form
- audit log and LLM explanation screen
"""

import pandas as pd
import streamlit as st

from app.auth import logout_user, render_auth_gate
from app.backend import (
    BASE_DISPLAY_COLUMNS,
    COLUMN_LABELS,
    SORT_OPTIONS,
    build_review_record,
    call_bedrock_llm,
    filter_worklist,
    load_patient_worklist,
    load_review_log,
    save_review_decision,
)


CATEGORY_OPTIONS = [
    "Green",
    "All Cases",
    "Amber",
    "Red",
    "Pending CM",
    "Pending Clinician",
]

# Options used in the final decision dropdown on the Patient Review tab.
FINAL_DECISION_OPTIONS = [
    "Pending Review",
    "Proceed with Community Hospital Referral",
    "Consider Hospital-at-Home",
    "Continue Acute Hospital Care",
    "Requires Further Clinical Review",
    "Requires Further Nursing Review",
    "Patient / Family Counselling Required",
    "Not Suitable for Transfer",
]


def configure_page():
    """Configure Streamlit page metadata and global CSS overrides."""
    st.set_page_config(page_title="ENCHANTED Model 1", layout="wide")
    # Most CSS below improves readability of Streamlit's generated widgets,
    # especially labels, buttons, dropdowns, radio options, and multiselect tags.
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f7fbff 0%, #eef6fb 45%, #f8fafc 100%);
            color: #1f2937;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        h1 {
            color: #0b3a66;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        h2, h3 {
            color: #1f4e79;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #dbeafe;
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #1f2937 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 800;
        }

        .stRadio label,
        .stTextInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stTextArea label,
        div[data-testid="stExpander"] summary,
        div[data-testid="stTabs"] button {
            color: #1f2937 !important;
            font-weight: 700 !important;
        }

        .stRadio label,
        .stRadio label p,
        .stRadio label span,
        .stCheckbox label,
        .stCheckbox label p,
        .stCheckbox label span,
        .stMarkdown,
        .stCaptionContainer {
            color: #334155 !important;
            font-weight: 700 !important;
        }

        input,
        textarea,
        input::placeholder,
        textarea::placeholder,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input {
            color: #111827 !important;
            font-weight: 600 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        div[data-testid="stDataFrame"] {
            background: white;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            padding: 8px;
        }

        textarea {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            color: #1f2937 !important;
        }

        input {
            background-color: #ffffff !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 10px;
            border-color: #94a3b8 !important;
            background-color: #ffffff !important;
        }

        div[data-baseweb="select"] svg {
            color: #334155 !important;
            fill: #334155 !important;
        }

        .stMultiSelect div[data-baseweb="tag"],
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #dbeafe !important;
            border: 1px solid #93c5fd !important;
            border-radius: 8px !important;
            color: #172554 !important;
            font-weight: 700 !important;
        }

        .stMultiSelect div[data-baseweb="tag"] span,
        .stMultiSelect span[data-baseweb="tag"] span,
        .stMultiSelect div[data-baseweb="tag"] svg,
        .stMultiSelect span[data-baseweb="tag"] svg {
            color: #172554 !important;
            fill: #172554 !important;
        }

        .stRadio div[role="radiogroup"] {
            gap: 0.5rem 1.2rem;
        }

        .stRadio div[role="radiogroup"] label {
            margin-right: 0.75rem;
        }

        div.stButton > button {
            background: #0b3a66 !important;
            color: #ffffff !important;
            border: 1px solid #0b3a66 !important;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 800;
            box-shadow: 0 4px 10px rgba(11, 58, 102, 0.22);
        }

        div.stButton > button p,
        div.stButton > button span,
        div.stButton > button div {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        div.stButton > button:hover {
            background: #145ea8 !important;
            color: #ffffff !important;
            border: 1px solid #145ea8 !important;
        }

        div.stButton > button:focus {
            color: #ffffff !important;
            border: 2px solid #38bdf8 !important;
            outline: none !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: #0b3a66 !important;
            color: #ffffff !important;
            border: 1px solid #0b3a66 !important;
            font-weight: 800 !important;
        }

        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        hr {
            border: none;
            height: 1px;
            background: #dbeafe;
            margin: 1.5rem 0;
        }

        [data-testid="stDataFrame"] div[data-baseweb="tag"],
        [data-testid="stDataFrame"] div[data-baseweb="tag"] span,
        [data-testid="stDataFrame"] span[data-baseweb="tag"] {
            font-weight: 700 !important;
            color: #000000 !important;
        }

        .top-bar {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 24px;
        }

        .user-chip {
            min-width: 220px;
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            text-align: right;
        }

        .user-chip-name {
            color: #1f2937;
            font-size: 14px;
            font-weight: 700;
        }

        .user-chip-role {
            color: #1d4ed8;
            font-size: 13px;
            font-weight: 700;
            margin-top: 2px;
        }

        .user-chip-id {
            color: #64748b;
            font-size: 12px;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(user_profile):
    """Render the dashboard title and logged-in user's designation chip."""
    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <h1>ENCHANTED Model 1: Right-Siting Decision Support Dashboard</h1>
                <div style="font-size: 22px; color: #1f4e79; font-weight: 700;">
                    Rule-Based Screening, AI Risk Stratification and Right-Siting Review Support
                </div>
            </div>
            <div class="user-chip">
                <div class="user-chip-name">{user_profile["name"]}</div>
                <div class="user-chip-role">{user_profile["designation"]}</div>
                <div class="user-chip-id">{user_profile["hospital_id"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "For demonstration using sample data. Final right-siting, referral and transfer decisions remain with the clinical care team."
    )
    _, logout_col = st.columns([0.86, 0.14])
    if logout_col.button("Logout"):
        logout_user()

    st.markdown(
        """
        <div style="
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #dbeafe;
            border-left: 6px solid #2563eb;
            padding: 18px 22px;
            border-radius: 16px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 20px;
        ">
            <div style="font-size: 18px; font-weight: 700; color: #0b3a66;">
                Acute-to-Community Hospital / Hospital-at-Home Right-Siting Support
            </div>
            <div style="font-size: 14px; color: #475569; margin-top: 6px;">
            This dashboard combines rule-based clinical screening, AI-supported risk stratification,
            service suitability, nursing assessment and acceptance considerations to assist case managers
            and clinicians in reviewing the appropriate care pathway.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap;">
            <span style="background:#fee2e2; color:#7f1d1d; padding:8px 14px; border-radius:999px; font-weight:600;">
                Red: Rule-Based Exclusion
            </span>
            <span style="background:#fef3c7; color:#78350f; padding:8px 14px; border-radius:999px; font-weight:600;">
                Amber: Clinical Review Required
            </span>
            <span style="background:#dcfce7; color:#14532d; padding:8px 14px; border-radius:999px; font-weight:600;">
                Green: Potential Candidate
            </span>
            <span style="background:#dbeafe; color:#1e3a8a; padding:8px 14px; border-radius:999px; font-weight:600;">
                AI: Risk Stratification
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def colour_rule_category(value):
    """Return cell styling for screening and recommendation status values."""
    if value == "Green - Potential Candidate":
        return "background-color: #d4edda; color: #155724;"
    if value == "Amber - Review Required":
        return "background-color: #fff3cd; color: #856404;"
    if value == "Red - No-Go":
        return "background-color: #f8d7da; color: #721c24;"
    if value == "Community Hospital referral review":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    if value == "Further clinical review before CH referral":
        return "background-color: #fff3cd; color: #856404; font-weight: bold;"
    if value == "Pending review":
        return "background-color: #e2e3e5; color: #383d41; font-weight: bold;"
    return ""


def role_scope(shortlisted, user_profile):
    """Return only the cases that should be visible to the logged-in user."""
    selected_role = user_profile["designation"]

    if selected_role == "Case Manager":
        assigned_case_manager = user_profile.get("assigned_case_manager")
        return shortlisted[shortlisted["assigned_case_manager"] == assigned_case_manager]

    if selected_role == "Clinician":
        return shortlisted[shortlisted["workflow_status"] == "Pending Clinician"]

    return shortlisted


def render_metrics(role_scoped):
    """Render summary metrics for the cases visible to the current user."""
    metric_cols = st.columns(5)
    metric_cols[0].metric("Role-Scoped Cases", len(role_scoped))
    metric_cols[1].metric(
        "Green",
        (role_scoped["rule_category"] == "Green - Potential Candidate").sum(),
    )
    metric_cols[2].metric(
        "Amber",
        (role_scoped["rule_category"] == "Amber - Review Required").sum(),
    )
    metric_cols[3].metric("Red", (role_scoped["rule_category"] == "Red - No-Go").sum())
    metric_cols[4].metric(
        "Pending Clinician",
        (role_scoped["workflow_status"] == "Pending Clinician").sum(),
    )

    st.subheader("AI-Suggested Review Pathway Summary")
    pathway_cols = st.columns(4)
    pathway_cols[0].metric(
        "CH Review",
        (role_scoped["right_siting_recommendation"] == "Community Hospital review").sum(),
    )
    pathway_cols[1].metric(
        "Hospital-at-Home Review",
        (role_scoped["right_siting_recommendation"] == "Hospital-at-Home review").sum(),
    )
    pathway_cols[2].metric(
        "Continue Acute Care",
        (
            role_scoped["right_siting_recommendation"]
            == "Continue Acute Hospital care"
        ).sum(),
    )
    pathway_cols[3].metric(
        "Pending CM",
        (role_scoped["workflow_status"] == "Pending CM").sum(),
    )


def split_tasks_for_user(role_scoped, user_profile):
    """Split visible cases into assigned, ongoing, and past-reviewed groups."""
    review_log = load_review_log()

    if review_log.empty:
        reviewed_patient_ids = set()
    else:
        if user_profile["designation"] == "Case Manager":
            if "assigned_case_manager" in review_log.columns:
                matching_reviews = review_log[
                    review_log["assigned_case_manager"]
                    == user_profile.get("assigned_case_manager")
                ]
            else:
                matching_reviews = review_log.iloc[0:0]
        else:
            if "dashboard_role" in review_log.columns:
                matching_reviews = review_log[
                    review_log["dashboard_role"] == user_profile["designation"]
                ]
            else:
                matching_reviews = review_log.iloc[0:0]

        if "patient_id" in matching_reviews.columns:
            reviewed_patient_ids = set(matching_reviews["patient_id"].dropna())
        else:
            reviewed_patient_ids = set()

    past = role_scoped[role_scoped["patient_id"].isin(reviewed_patient_ids)]
    active = role_scoped[~role_scoped["patient_id"].isin(reviewed_patient_ids)]

    if user_profile["designation"] == "Case Manager":
        assigned = active[active["workflow_status"] == "Pending CM"]
    elif user_profile["designation"] == "Clinician":
        assigned = active[active["workflow_status"] == "Pending Clinician"]
    else:
        assigned = active[
            active["workflow_status"].isin(["Pending CM", "Pending Clinician"])
        ]

    ongoing = active[~active["patient_id"].isin(assigned["patient_id"])]

    return assigned, ongoing, past


def render_task_table(data, empty_message):
    """Render a compact task table for one task queue."""
    task_columns = [
        "patient_id",
        "encounter_id",
        "ward",
        "age",
        "rule_category",
        "risk_band",
        "workflow_status",
        "right_siting_recommendation",
    ]
    visible_columns = [col for col in task_columns if col in data.columns]

    if data.empty:
        st.info(empty_message)
        return

    st.dataframe(
        data[visible_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "patient_id": st.column_config.TextColumn("Patient ID"),
            "encounter_id": st.column_config.TextColumn("Encounter ID"),
            "ward": st.column_config.TextColumn("Ward"),
            "age": st.column_config.NumberColumn("Age"),
            "rule_category": st.column_config.TextColumn("Clinical Screening"),
            "risk_band": st.column_config.TextColumn("Risk Band"),
            "workflow_status": st.column_config.TextColumn("Status"),
            "right_siting_recommendation": st.column_config.TextColumn(
                "Right-Siting Recommendation"
            ),
        },
    )


def render_task_sections(role_scoped, user_profile):
    """Render the Overview & Tasks tab content."""
    assigned, ongoing, past = split_tasks_for_user(role_scoped, user_profile)

    st.subheader("Your Tasks")
    task_metric_cols = st.columns(3)
    task_metric_cols[0].metric("Tasks Assigned to You", len(assigned))
    task_metric_cols[1].metric("Ongoing", len(ongoing))
    task_metric_cols[2].metric("Past", len(past))

    assigned_tab, ongoing_tab, past_tab = st.tabs(
        ["Tasks assigned to you", "Ongoing", "Past"]
    )

    with assigned_tab:
        render_task_table(assigned, "No assigned tasks for your current designation.")

    with ongoing_tab:
        render_task_table(ongoing, "No ongoing cases outside your assigned task queue.")

    with past_tab:
        render_task_table(past, "No past reviewed cases yet.")


def render_worklist_controls(shortlisted, role_scoped):
    """Render search, sort, column, category, and advanced-filter controls."""
    control_cols = st.columns([1.4, 1.2, 1, 1])
    selected_category = control_cols[0].radio(
        "Case category",
        CATEGORY_OPTIONS,
        horizontal=True,
    )
    search_query = control_cols[1].text_input(
        "Search",
        placeholder="Patient, encounter, ward, specialty",
    )
    sort_label = control_cols[2].selectbox(
        "Sort by",
        list(SORT_OPTIONS.keys()),
        index=1,
    )
    sort_direction = control_cols[3].selectbox(
        "Sort direction",
        ["Descending", "Ascending"],
    )

    available_columns = [col for col in COLUMN_LABELS.keys() if col in shortlisted.columns]
    selected_columns = st.multiselect(
        "Columns",
        available_columns,
        default=[col for col in BASE_DISPLAY_COLUMNS if col in available_columns],
        format_func=lambda col: COLUMN_LABELS[col],
    )

    with st.expander("Advanced filters"):
        filter_cols = st.columns(5)
        ward_filter = filter_cols[0].multiselect(
            "Ward / Floor",
            sorted(role_scoped["ward"].dropna().unique()),
        )
        status_filter = filter_cols[1].multiselect(
            "Status",
            sorted(role_scoped["workflow_status"].dropna().unique()),
        )
        risk_filter = filter_cols[2].multiselect(
            "Risk Band",
            sorted(role_scoped["risk_band"].dropna().unique()),
        )
        specialty_filter = filter_cols[3].multiselect(
            "Specialty",
            sorted(role_scoped["specialty"].dropna().unique()),
        )
        age_filter = filter_cols[4].slider(
            "Age Range",
            int(shortlisted["age"].min()),
            int(shortlisted["age"].max()),
            (int(shortlisted["age"].min()), int(shortlisted["age"].max())),
        )

    filtered_worklist = filter_worklist(
        role_scoped,
        selected_category,
        search_query,
        ward_filter,
        status_filter,
        risk_filter,
        age_filter,
        specialty_filter,
        sort_label,
        sort_direction == "Ascending",
    )

    return filtered_worklist, selected_columns


def render_worklist_table(filtered_worklist, role_scoped, selected_columns):
    """Render the interactive patient list table with selected columns."""
    if selected_columns:
        style_subset = [
            col
            for col in ["rule_category", "ai_recommendation"]
            if col in selected_columns
        ]
        styled_df = filtered_worklist[selected_columns].style

        if style_subset:
            styled_df = styled_df.map(colour_rule_category, subset=style_subset)

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            column_config={
                "patient_id": st.column_config.TextColumn("Patient ID", width="medium"),
                "encounter_id": st.column_config.TextColumn(
                    "Encounter ID",
                    width="medium",
                ),
                "ward": st.column_config.TextColumn("Ward / Floor", width="small"),
                "specialty": st.column_config.TextColumn("Specialty", width="medium"),
                "age": st.column_config.NumberColumn("Age", width="small"),
                "los_days": st.column_config.NumberColumn(
                    "Days in Hospital",
                    width="small",
                ),
                "days_to_edd": st.column_config.NumberColumn(
                    "Days to EDD",
                    width="small",
                ),
                "rule_category": st.column_config.TextColumn(
                    "Clinical Screening",
                    width="large",
                ),
                "red_flags": st.column_config.ListColumn("Red Flags", width="large"),
                "amber_flags": st.column_config.ListColumn(
                    "Amber Flags",
                    width="large",
                ),
                "risk_score": st.column_config.NumberColumn(
                    "Risk Score",
                    width="small",
                    format="%.2f",
                ),
                "risk_band": st.column_config.TextColumn("Risk Band", width="medium"),
                "workflow_status": st.column_config.TextColumn(
                    "Status",
                    width="medium",
                ),
                "service_need": st.column_config.TextColumn(
                    "Service Need",
                    width="large",
                ),
                "service_suitability": st.column_config.TextColumn(
                    "Service Suitability",
                    width="large",
                ),
                "nursing_status": st.column_config.TextColumn(
                    "Nursing Assessment",
                    width="large",
                ),
                "nursing_flags": st.column_config.ListColumn(
                    "Nursing Flags",
                    width="large",
                ),
                "patient_acceptance_likelihood": st.column_config.TextColumn(
                    "Acceptance Likelihood",
                    width="medium",
                ),
                "counselling_required": st.column_config.TextColumn(
                    "Counselling Required",
                    width="medium",
                ),
                "right_siting_recommendation": st.column_config.TextColumn(
                    "Right-Siting Recommendation",
                    width="large",
                ),
                "ai_recommendation": st.column_config.TextColumn(
                    "AI Recommendation",
                    width="large",
                ),
            },
        )
    else:
        st.warning("Select at least one column to display.")

    st.caption(f"Showing {len(filtered_worklist)} of {len(role_scoped)} role-scoped cases.")


def render_patient_list_tab(shortlisted, role_scoped):
    """Render the Patient List tab and return its filtered dataframe."""
    st.subheader("All Patient List")
    st.caption(
        "Use the case category switch, search, sorting, column selector, and advanced filters to review the full list visible to your designation."
    )
    render_metrics(role_scoped)
    st.divider()

    filtered_worklist, selected_columns = render_worklist_controls(
        shortlisted,
        role_scoped,
    )
    render_worklist_table(filtered_worklist, role_scoped, selected_columns)

    return filtered_worklist


def select_patient_for_review(shortlisted, role_scoped):
    """Render patient selection for the review screen and return the row."""
    st.subheader("Select Patient for Detail Review")
    st.caption(
        "Choose a patient from your visible case list, then complete the review workflow below."
    )

    patient_options = role_scoped["patient_id"].tolist()

    if not patient_options:
        st.info("No patients are visible for your current designation.")
        st.stop()

    selected_patient = st.selectbox(
        "Patient",
        patient_options,
        format_func=lambda patient_id: (
            f"{patient_id} - "
            f"{shortlisted.loc[shortlisted['patient_id'] == patient_id, 'encounter_id'].iloc[0]}"
        ),
    )

    return shortlisted[shortlisted["patient_id"] == selected_patient].iloc[0]


def render_patient_review_tab(shortlisted, role_scoped, user_profile):
    """Render patient-level details and the final review decision form."""
    patient_row = select_patient_for_review(shortlisted, role_scoped)

    st.write("### Screening Output")
    st.write(f"**Rule-based category:** {patient_row['rule_category']}")
    st.write(f"**Workflow status:** {patient_row['workflow_status']}")
    st.write(f"**Assigned case manager:** {patient_row['assigned_case_manager']}")
    st.write(f"**Red flags:** {patient_row['red_flags']}")
    st.write(f"**Amber flags:** {patient_row['amber_flags']}")

    if pd.notna(patient_row["risk_score"]):
        st.write(f"**Predictive risk score:** {patient_row['risk_score']:.2f}")
    else:
        st.write("**Predictive risk score:** Not applicable")

    st.write(f"**Predictive risk band:** {patient_row['risk_band']}")
    st.write(f"**AI-supported recommendation:** *{patient_row['ai_recommendation']}*")

    st.write("### Service Suitability")
    st.write(f"**Service need:** {patient_row['service_need']}")
    st.write(f"**Service suitability:** {patient_row['service_suitability']}")

    st.write("### Nursing / Operational Assessment")
    st.write(f"**Nursing status:** {patient_row['nursing_status']}")
    st.write(f"**Nursing flags:** {patient_row['nursing_flags']}")

    st.write("### Acceptance / Counselling Assessment")
    st.write(
        f"**Patient acceptance likelihood:** {patient_row['patient_acceptance_likelihood']}"
    )
    st.write(f"**Counselling required:** {patient_row['counselling_required']}")

    st.write("### Right-Siting Recommendation")
    st.write(f"**Recommended care setting:** {patient_row['right_siting_recommendation']}")
    st.write(f"**AI-supported recommendation:** *{patient_row['ai_recommendation']}*")

    final_decision = st.selectbox(
        "Final right-siting decision",
        FINAL_DECISION_OPTIONS,
    )
    review_comments = st.text_area("Review comments / override reason")

    st.info(
        "The AI model provides decision support only. "
        "Final referral decisions remain with the case manager / clinical team."
    )

    if st.button("Submit Review Decision"):
        review_record = build_review_record(
            patient_row,
            user_profile["designation"],
            final_decision,
            review_comments,
        )
        save_review_decision(review_record)
        st.success("Review decision submitted and saved to audit log.")


def render_audit_and_llm_tab(shortlisted, role_scoped):
    """Render submitted review history and optional Bedrock explanation support."""
    st.subheader("Audit Log")
    review_log = load_review_log()

    if review_log.empty:
        st.info("No review decisions have been submitted yet.")
    else:
        st.dataframe(review_log, width="stretch")

    st.divider()
    st.subheader("LLM Explanation Support")

    patient_options = role_scoped["patient_id"].tolist()

    if not patient_options:
        st.info("No patients are visible for your current designation.")
        return

    selected_patient = st.selectbox(
        "Patient for LLM prompt",
        patient_options,
        key="llm_patient_selector",
    )
    patient_row = shortlisted[shortlisted["patient_id"] == selected_patient].iloc[0]

    st.text_area(
        "Prompt that would be sent to the LLM explanation layer",
        patient_row["llm_prompt"],
        height=400,
    )

    if st.button("Generate LLM Explanation"):
        with st.spinner("Generating explanation..."):
            llm_output = call_bedrock_llm(patient_row["llm_prompt"])

        st.markdown(llm_output)


def render_dashboard():
    """Main frontend orchestrator called by classification.py."""
    configure_page()
    user_profile = render_auth_gate()
    shortlisted = load_patient_worklist()

    render_header(user_profile)
    role_scoped = role_scope(shortlisted, user_profile)

    overview_tab, patient_list_tab, review_tab, audit_tab = st.tabs(
        [
            "Overview & Tasks",
            "Patient List",
            "Patient Review",
            "Audit & LLM",
        ]
    )

    with overview_tab:
        render_task_sections(role_scoped, user_profile)

    with patient_list_tab:
        render_patient_list_tab(shortlisted, role_scoped)

    with review_tab:
        render_patient_review_tab(shortlisted, role_scoped, user_profile)

    with audit_tab:
        render_audit_and_llm_tab(shortlisted, role_scoped)
