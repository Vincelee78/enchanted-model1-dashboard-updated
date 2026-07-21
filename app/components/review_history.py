"""Shared audit-log and review-history presentation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.backend import load_review_log


def _scope_history(log: pd.DataFrame, user_profile: dict, patient_id: str | None) -> pd.DataFrame:
    if log.empty:
        return log

    role = user_profile.get("designation")
    result = log.copy()

    if role == "Case Manager" and "assigned_case_manager" in result.columns:
        assigned = user_profile.get("assigned_case_manager")
        result = result[result["assigned_case_manager"].astype(str) == str(assigned)]
    elif role == "Clinician" and "dashboard_role" in result.columns:
        # Clinicians see their own submitted decisions by default.
        result = result[result["dashboard_role"] == "Clinician"]
    elif role == "JCH Referral Team" and "dashboard_role" in result.columns:
        # Referral team can see the full cross-role history because downstream
        # handoff decisions depend on upstream CM and clinician reviews.
        result = result

    if patient_id and "patient_id" in result.columns:
        result = result[result["patient_id"].astype(str) == str(patient_id)]

    if "timestamp" in result.columns:
        parsed = pd.to_datetime(result["timestamp"], errors="coerce")
        result = result.assign(_parsed_timestamp=parsed).sort_values(
            "_parsed_timestamp", ascending=False, na_position="last"
        )
        result = result.drop(columns=["_parsed_timestamp"])

    return result


def render_review_history(
    user_profile: dict,
    *,
    patient_id: str | None = None,
    title: str = "Review History",
) -> None:
    """Render role-scoped audit history."""
    st.subheader(title)
    log = load_review_log()
    scoped = _scope_history(log, user_profile, patient_id)

    if scoped.empty:
        st.info("No review decisions have been recorded for this view yet.")
        return

    preferred = [
        "timestamp",
        "dashboard_role",
        "patient_id",
        "encounter_id",
        "source_queue",
        "original_screening_bucket",
        "override_bucket",
        "workflow_status",
        "risk_band",
        "final_decision",
        "review_comments",
    ]
    visible = [column for column in preferred if column in scoped.columns]
    st.dataframe(scoped[visible], width="stretch", hide_index=True)