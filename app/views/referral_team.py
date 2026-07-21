"""JCH referral-team-specific ENCHANTED dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backend import (
    BUCKET_MONITORING,
    BUCKET_POTENTIAL,
    WORKFLOW_PENDING_CM,
)
from app.components.patient_summary import (
    render_decision_form,
    render_filterable_worklist,
    render_patient_summary,
    render_patient_table,
    select_patient,
)
from app.components.review_history import render_review_history


REFERRAL_ACTIONS = [
    "Accept for Referral Processing",
    "Hold for Operational Checks",
    "Return to Case Manager for Clarification",
    "Request Clinical Clarification",
    "Return to Pending Monitoring",
    "Not Eligible at Referral Review",
    "Override / Reclassify Screening Bucket",
]


def _referral_candidates(worklist: pd.DataFrame) -> pd.DataFrame:
    return worklist[
        (worklist["workflow_status"] == WORKFLOW_PENDING_CM)
        | (worklist["rule_category"] == BUCKET_POTENTIAL)
        | (worklist["right_siting_recommendation"] == "Community Hospital review")
    ].copy()


def _render_metrics(worklist: pd.DataFrame) -> None:
    candidates = _referral_candidates(worklist)
    metrics = st.columns(5)
    metrics[0].metric("Referral Candidates", len(candidates))
    metrics[1].metric(
        "Potential CH Candidates",
        int((worklist["rule_category"] == BUCKET_POTENTIAL).sum()),
    )
    metrics[2].metric(
        "Pending Monitoring",
        int((worklist["rule_category"] == BUCKET_MONITORING).sum()),
    )
    metrics[3].metric(
        "Counselling Likely",
        int((worklist["patient_acceptance_likelihood"] == "Low").sum()),
    )
    metrics[4].metric(
        "Service Suitable",
        int((worklist["service_suitability"] == "JCH Service Suitable").sum()),
    )


def render_referral_dashboard(worklist: pd.DataFrame, user_profile: dict) -> None:
    """Render downstream referral and operational-readiness workflows."""
    _render_metrics(worklist)
    candidates = _referral_candidates(worklist)

    queue_tab, operations_tab, review_tab, history_tab = st.tabs(
        [
            "Referral Queue",
            "Operational Readiness",
            "Referral Review",
            "Cross-Role History",
        ]
    )

    with queue_tab:
        st.subheader("Pre-Screened Referral Queue")
        st.caption(
            "This queue is a prototype representation of patients surfaced after AI-assisted and case-manager screening."
        )
        render_filterable_worklist(
            candidates,
            key_prefix="referral_queue",
            category_options=["All Cases", "Potential CH Candidate"],
            default_columns=[
                "patient_id",
                "encounter_id",
                "ward",
                "specialty",
                "rule_category",
                "risk_band",
                "service_suitability",
                "nursing_status",
                "patient_acceptance_likelihood",
                "right_siting_recommendation",
            ],
        )

    with operations_tab:
        st.subheader("Operational Readiness Review")
        st.caption(
            "Review service fit, nursing complexity, acceptance/counselling needs and cases waiting for reassessment."
        )
        operational = worklist[
            worklist["rule_category"].isin([BUCKET_POTENTIAL, BUCKET_MONITORING])
        ].copy()
        render_patient_table(
            operational,
            columns=[
                "patient_id",
                "encounter_id",
                "ward",
                "rule_category",
                "service_need",
                "service_suitability",
                "nursing_status",
                "nursing_flags",
                "patient_acceptance_likelihood",
                "counselling_required",
                "workflow_status",
            ],
            empty_message="No cases currently require referral or operational review.",
        )

    with review_tab:
        patient = select_patient(
            candidates,
            key_prefix="referral_review",
            title="Referral and Transfer Review",
            caption="Confirm downstream eligibility and operational readiness without replacing the clinical decision.",
        )
        if patient is not None:
            render_patient_summary(patient, show_raw_prompt=False)
            render_decision_form(
                patient,
                user_profile,
                action_options=REFERRAL_ACTIONS,
                action_label="Referral team action",
                comments_label="Operational / referral review comments",
                key_prefix="referral_review",
                source_queue="Referral Team Review",
                allow_bucket_override=True,
            )

    with history_tab:
        render_review_history(
            user_profile,
            title="Cross-Role Review and Handoff History",
        )