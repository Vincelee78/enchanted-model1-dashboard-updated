"""Case-manager-specific ENCHANTED dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backend import (
    BUCKET_CLINICAL_REVIEW,
    BUCKET_MONITORING,
    BUCKET_NOT_SUITABLE,
    BUCKET_POTENTIAL,
    WORKFLOW_PENDING_CM,
)
from app.components.patient_summary import (
    render_decision_form,
    render_filterable_worklist,
    render_llm_explanation,
    render_patient_summary,
    render_patient_table,
    select_patient,
)
from app.components.review_history import render_review_history


CASE_MANAGER_ACTIONS = [
    "Confirm Potential CH Candidate",
    "Continue Monitoring",
    "Escalate for Clinical Review",
    "Request Additional Information",
    "Remove from Active Review",
    "Override / Reclassify Screening Bucket",
]


def _render_metrics(worklist: pd.DataFrame) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Assigned Patients", len(worklist))
    metrics[1].metric(
        "Potential Candidates",
        int((worklist["rule_category"] == BUCKET_POTENTIAL).sum()),
    )
    metrics[2].metric(
        "Pending Monitoring",
        int((worklist["rule_category"] == BUCKET_MONITORING).sum()),
    )
    metrics[3].metric(
        "Clinical Review",
        int((worklist["rule_category"] == BUCKET_CLINICAL_REVIEW).sum()),
    )
    metrics[4].metric(
        "Not Suitable Now",
        int((worklist["rule_category"] == BUCKET_NOT_SUITABLE).sum()),
    )


def render_case_manager_dashboard(
    worklist: pd.DataFrame,
    user_profile: dict,
) -> None:
    """Render queues and actions appropriate for a CHoF case manager."""
    _render_metrics(worklist)

    queue_tab, monitoring_tab, review_tab, history_tab = st.tabs(
        [
            "My Work Queue",
            "Pending Monitoring",
            "Patient Screening & Review",
            "Review History",
        ]
    )

    with queue_tab:
        st.subheader("Actionable Case Manager Queue")
        st.caption(
            "Potential CH candidates and patients routed for case-manager confirmation."
        )
        actionable = worklist[
            (worklist["workflow_status"] == WORKFLOW_PENDING_CM)
            | (worklist["rule_category"] == BUCKET_POTENTIAL)
        ].copy()
        render_filterable_worklist(
            actionable,
            key_prefix="cm_queue",
            category_options=["All Cases", BUCKET_POTENTIAL],
            default_columns=[
                "patient_id",
                "encounter_id",
                "ward",
                "specialty",
                "rule_category",
                "risk_band",
                "service_suitability",
                "right_siting_recommendation",
                "ai_recommendation",
            ],
        )

    with monitoring_tab:
        st.subheader("Patients Requiring Reassessment")
        st.caption(
            "These patients have dynamic criteria or pending information and should be reassessed after data refresh."
        )
        monitoring = worklist[worklist["rule_category"] == BUCKET_MONITORING].copy()
        render_patient_table(
            monitoring,
            columns=[
                "patient_id",
                "encounter_id",
                "ward",
                "specialty",
                "amber_flags",
                "risk_band",
                "workflow_status",
                "ai_recommendation",
            ],
            empty_message="No assigned patients currently require monitoring.",
        )

    with review_tab:
        patient = select_patient(
            worklist,
            key_prefix="cm_review",
            title="Patient Screening and Review",
            caption="Review the screening rationale, relevant clinical information and advisory next action.",
        )
        if patient is not None:
            render_patient_summary(patient, show_raw_prompt=False)
            render_decision_form(
                patient,
                user_profile,
                action_options=CASE_MANAGER_ACTIONS,
                action_label="Case manager action",
                comments_label="Case manager comments / override rationale",
                key_prefix="cm_review",
                source_queue="Case Manager Review",
                allow_bucket_override=True,
            )
            with st.expander("Optional LLM explanation"):
                render_llm_explanation(patient, key_prefix="cm_review")

    with history_tab:
        render_review_history(user_profile, title="My Case Manager Review History")