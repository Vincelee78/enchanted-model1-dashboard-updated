"""Clinician-specific ENCHANTED dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backend import BUCKET_CLINICAL_REVIEW
from app.components.patient_summary import (
    render_decision_form,
    render_filterable_worklist,
    render_llm_explanation,
    render_patient_summary,
    select_patient,
)
from app.components.review_history import render_review_history


CLINICIAN_ACTIONS = [
    "Agree Suitable for CH Consideration",
    "Return to Pending Monitoring",
    "Continue Acute Hospital Care",
    "Request Further Assessment",
    "Confirm Not Suitable at Current Review",
    "Override / Reclassify Screening Bucket",
]


def _render_metrics(worklist: pd.DataFrame) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Clinical Review Queue", len(worklist))
    metrics[1].metric(
        "Rule-Based Review Flags",
        int((worklist["rule_category"] == BUCKET_CLINICAL_REVIEW).sum()),
    )
    metrics[2].metric(
        "High Prototype Risk",
        int((worklist["risk_band"] == "High Risk").sum()),
    )
    metrics[3].metric(
        "Nursing Review Required",
        int((worklist["nursing_status"] == "Nursing Review Required").sum()),
    )


def render_clinician_dashboard(worklist: pd.DataFrame, user_profile: dict) -> None:
    """Render a focused view for cases requiring clinical judgement."""
    _render_metrics(worklist)

    queue_tab, summary_tab, decision_tab, history_tab = st.tabs(
        [
            "Clinical Review Queue",
            "Patient Clinical Summary",
            "Clinical Decision",
            "Decision History",
        ]
    )

    with queue_tab:
        st.subheader("Cases Requiring Clinical Review")
        st.caption(
            "Cases appear here because of judgement-based screening flags, high prototype risk or unresolved clinical/operational concerns."
        )
        render_filterable_worklist(
            worklist,
            key_prefix="clinician_queue",
            category_options=["All Cases", "Needs Clinical Review", "Pending Monitoring", "Potential CH Candidate"],
            default_columns=[
                "patient_id",
                "encounter_id",
                "ward",
                "specialty",
                "rule_category",
                "review_flags",
                "amber_flags",
                "risk_band",
                "nursing_status",
                "ai_recommendation",
            ],
        )

    with summary_tab:
        patient = select_patient(
            worklist,
            key_prefix="clinician_summary",
            title="Clinical Summary",
            caption="Review the factors that led to escalation before recording a decision.",
        )
        if patient is not None:
            render_patient_summary(patient, show_raw_prompt=False)
            render_review_history(
                user_profile,
                patient_id=str(patient.get("patient_id")),
                title="Existing Clinician Decisions for This Patient",
            )
            with st.expander("Optional LLM explanation"):
                render_llm_explanation(patient, key_prefix="clinician_summary")

    with decision_tab:
        patient = select_patient(
            worklist,
            key_prefix="clinician_decision",
            title="Record Clinical Decision",
            caption="The clinical rationale should be documented when overriding or reclassifying the screening output.",
        )
        if patient is not None:
            render_patient_summary(patient, show_raw_prompt=False)
            render_decision_form(
                patient,
                user_profile,
                action_options=CLINICIAN_ACTIONS,
                action_label="Clinical decision",
                comments_label="Clinical rationale / required follow-up",
                key_prefix="clinician_decision",
                source_queue="Clinician Review",
                allow_bucket_override=True,
            )

    with history_tab:
        render_review_history(user_profile, title="Clinician Decision History")