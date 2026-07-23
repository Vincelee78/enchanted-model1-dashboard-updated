"""Reusable patient tables, summaries, filters and review forms."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

from app.backend import (
    BASE_DISPLAY_COLUMNS,
    BUCKET_CLINICAL_REVIEW,
    BUCKET_MONITORING,
    BUCKET_NOT_SUITABLE,
    BUCKET_POTENTIAL,
    COLUMN_LABELS,
    SORT_OPTIONS,
    build_review_record,
    call_bedrock_llm,
    clear_review_log_cache,
    filter_worklist,
    save_review_decision,
)


ALL_BUCKETS = [
    "All Cases",
    BUCKET_POTENTIAL,
    BUCKET_MONITORING,
    BUCKET_CLINICAL_REVIEW,
    BUCKET_NOT_SUITABLE,
]


def _bucket_style(value: object) -> str:
    styles = {
        BUCKET_POTENTIAL: "background-color:#d4edda;color:#155724;font-weight:700;",
        BUCKET_MONITORING: "background-color:#fff3cd;color:#856404;font-weight:700;",
        BUCKET_CLINICAL_REVIEW: "background-color:#e0e7ff;color:#3730a3;font-weight:700;",
        BUCKET_NOT_SUITABLE: "background-color:#f8d7da;color:#721c24;font-weight:700;",
    }
    return styles.get(str(value), "")


def _safe_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    return [value]


def _display_list(value: object) -> str:
    items = _safe_list(value)
    return "None" if not items else "\n".join(f"- {item}" for item in items)


def render_patient_table(
    data: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    empty_message: str = "No patients match the current view.",
) -> None:
    """Render a consistently configured patient table."""
    if data.empty:
        st.info(empty_message)
        return

    requested = list(columns or BASE_DISPLAY_COLUMNS)
    visible = [column for column in requested if column in data.columns]
    table = data[visible].copy()

    styled = table.style
    if "rule_category" in visible:
        styled = styled.map(_bucket_style, subset=["rule_category"])

    configs = {
        "patient_id": st.column_config.TextColumn("Patient ID", width="medium"),
        "encounter_id": st.column_config.TextColumn("Encounter ID", width="medium"),
        "ward": st.column_config.TextColumn("Ward / Floor", width="small"),
        "specialty": st.column_config.TextColumn("Specialty", width="medium"),
        "age": st.column_config.NumberColumn("Age", width="small"),
        "los_days": st.column_config.NumberColumn("Days in Hospital", width="small"),
        "days_to_edd": st.column_config.NumberColumn("Days to EDD", width="small"),
        "rule_category": st.column_config.TextColumn("Screening Bucket", width="large"),
        "red_flags": st.column_config.ListColumn("Hard Exclusion Flags", width="large"),
        "amber_flags": st.column_config.ListColumn("Monitoring Flags", width="large"),
        "review_flags": st.column_config.ListColumn("Clinical Review Flags", width="large"),
        "risk_score": st.column_config.NumberColumn(
            "Prototype Risk Score", width="small", format="%.2f"
        ),
        "risk_band": st.column_config.TextColumn("Prototype Risk Band", width="medium"),
        "workflow_status": st.column_config.TextColumn("Workflow Status", width="medium"),
        "service_need": st.column_config.TextColumn("Service Need", width="large"),
        "service_suitability": st.column_config.TextColumn("Service Suitability", width="large"),
        "nursing_status": st.column_config.TextColumn("Nursing Assessment", width="large"),
        "nursing_flags": st.column_config.ListColumn("Nursing Flags", width="large"),
        "patient_acceptance_likelihood": st.column_config.TextColumn(
            "Acceptance Likelihood", width="medium"
        ),
        "right_siting_recommendation": st.column_config.TextColumn(
            "Suggested Review Pathway", width="large"
        ),
        "ai_recommendation": st.column_config.TextColumn(
            "Advisory Next Action", width="large"
        ),
    }
    st.dataframe(styled, width="stretch", hide_index=True, column_config=configs)


def render_filterable_worklist(
    data: pd.DataFrame,
    *,
    key_prefix: str,
    category_options: list[str] | None = None,
    default_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Render common worklist controls and return the filtered rows."""
    if data.empty:
        st.info("No patients are available for this role or queue.")
        return data

    categories = category_options or ALL_BUCKETS
    controls = st.columns([1.5, 1.3, 1.0, 1.0])
    category = controls[0].selectbox(
        "Screening category",
        categories,
        key=f"{key_prefix}_category",
    )
    search_query = controls[1].text_input(
        "Search",
        placeholder="Patient, encounter, ward, specialty",
        key=f"{key_prefix}_search",
    )
    sort_label = controls[2].selectbox(
        "Sort by",
        list(SORT_OPTIONS),
        key=f"{key_prefix}_sort",
    )
    sort_direction = controls[3].selectbox(
        "Direction",
        ["Descending", "Ascending"],
        key=f"{key_prefix}_direction",
    )

    available_columns = [column for column in COLUMN_LABELS if column in data.columns]
    defaults = default_columns or BASE_DISPLAY_COLUMNS
    selected_columns = st.multiselect(
        "Columns",
        available_columns,
        default=[column for column in defaults if column in available_columns],
        format_func=lambda column: COLUMN_LABELS.get(column, column),
        key=f"{key_prefix}_columns",
    )

    with st.expander("Advanced filters"):
        filter_cols = st.columns(5)
        ward_values = sorted(data.get("ward", pd.Series(dtype=str)).dropna().astype(str).unique())
        status_values = sorted(
            data.get("workflow_status", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        risk_values = sorted(
            data.get("risk_band", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        specialty_values = sorted(
            data.get("specialty", pd.Series(dtype=str)).dropna().astype(str).unique()
        )

        wards = filter_cols[0].multiselect(
            "Ward / Floor", ward_values, key=f"{key_prefix}_wards"
        )
        statuses = filter_cols[1].multiselect(
            "Workflow status", status_values, key=f"{key_prefix}_statuses"
        )
        risk_bands = filter_cols[2].multiselect(
            "Risk band", risk_values, key=f"{key_prefix}_risks"
        )
        specialties = filter_cols[3].multiselect(
            "Specialty", specialty_values, key=f"{key_prefix}_specialties"
        )

        numeric_age = pd.to_numeric(data.get("age"), errors="coerce").dropna()
        if numeric_age.empty:
            age_range = None
            filter_cols[4].caption("Age filter unavailable")
        else:
            min_age = int(numeric_age.min())
            max_age = int(numeric_age.max())
            age_range = filter_cols[4].slider(
                "Age range",
                min_age,
                max_age,
                (min_age, max_age),
                key=f"{key_prefix}_age",
            )

    filtered = filter_worklist(
        data,
        category,
        search_query,
        wards,
        statuses,
        risk_bands,
        age_range,
        specialties,
        sort_label,
        sort_direction == "Ascending",
    )
    render_patient_table(filtered, columns=selected_columns)
    st.caption(f"Showing {len(filtered)} of {len(data)} cases in this role-scoped view.")
    return filtered


def select_patient(
    worklist: pd.DataFrame,
    *,
    key_prefix: str,
    title: str = "Select patient",
    caption: str | None = None,
) -> pd.Series | None:
    """Render a patient selector and return the selected row."""
    st.subheader(title)
    if caption:
        st.caption(caption)
    if worklist.empty:
        st.info("No patients are available in this queue.")
        return None

    options = worklist["patient_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "Patient",
        options,
        key=f"{key_prefix}_patient",
        format_func=lambda patient_id: _patient_option_label(worklist, patient_id),
    )
    matching = worklist[worklist["patient_id"].astype(str) == str(selected_id)]
    return None if matching.empty else matching.iloc[0]


def _patient_option_label(worklist: pd.DataFrame, patient_id: str) -> str:
    matching = worklist[worklist["patient_id"].astype(str) == str(patient_id)]
    if matching.empty:
        return str(patient_id)
    row = matching.iloc[0]
    return f"{patient_id} — {row.get('encounter_id', '')} — {row.get('rule_category', '')}"


def render_patient_summary(patient_row: pd.Series, *, show_raw_prompt: bool = False) -> None:
    """Render a role-neutral clinical and operational patient summary."""
    heading_cols = st.columns(4)
    heading_cols[0].metric("Patient", patient_row.get("patient_id", "—"))
    heading_cols[1].metric("Encounter", patient_row.get("encounter_id", "—"))
    heading_cols[2].metric("Ward", patient_row.get("ward", "—"))
    heading_cols[3].metric("Screening Bucket", patient_row.get("rule_category", "—"))

    left, right = st.columns([1.15, 1.0])
    with left:
        st.markdown("### Screening rationale")
        st.markdown(f"**Hard exclusions**\n\n{_display_list(patient_row.get('red_flags'))}")
        st.markdown(f"**Monitoring flags**\n\n{_display_list(patient_row.get('amber_flags'))}")
        st.markdown(
            f"**Clinical review flags**\n\n{_display_list(patient_row.get('review_flags'))}"
        )
        st.markdown(f"**Workflow status:** {patient_row.get('workflow_status', '—')}")
        st.markdown(
            f"**Assigned case manager:** {patient_row.get('assigned_case_manager', '—')}"
        )

    with right:
        st.markdown("### Decision-support output")
        score = patient_row.get("risk_score")
        score_text = "Not available" if pd.isna(score) else f"{float(score):.2f}"
        st.markdown(f"**Prototype risk score:** {score_text}")
        st.markdown(f"**Prototype risk band:** {patient_row.get('risk_band', '—')}")
        st.markdown(
            f"**Suggested pathway:** {patient_row.get('right_siting_recommendation', '—')}"
        )
        st.info(str(patient_row.get("ai_recommendation", "No advisory action available.")))

    with st.expander("Clinical values available in the current sample dataset"):
        values = {
            "Age": patient_row.get("age"),
            "Specialty": patient_row.get("specialty"),
            "Days in hospital": patient_row.get("los_days"),
            "Days to EDD": patient_row.get("days_to_edd"),
            "Systolic BP": patient_row.get("systolic_bp"),
            "Diastolic BP": patient_row.get("diastolic_bp"),
            "Heart rate": patient_row.get("heart_rate"),
            "Temperature": patient_row.get("temperature"),
            "SpO2": patient_row.get("spo2"),
            "Oxygen device": patient_row.get("oxygen_device"),
            "Oxygen flow rate": patient_row.get("oxygen_flow_rate"),
            "Haemoglobin": patient_row.get("hb"),
            "Platelet": patient_row.get("platelet"),
            "ANC": patient_row.get("anc"),
            "Sodium": patient_row.get("sodium"),
            "Potassium": patient_row.get("potassium"),
        }
        st.dataframe(
            pd.DataFrame([{"Field": key, "Value": value} for key, value in values.items()]),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Service, nursing and acceptance assessment"):
        st.markdown(f"**Service need:** {patient_row.get('service_need', '—')}")
        st.markdown(
            f"**Service suitability:** {patient_row.get('service_suitability', '—')}"
        )
        st.markdown(f"**Nursing status:** {patient_row.get('nursing_status', '—')}")
        st.markdown(f"**Nursing flags:** {_display_list(patient_row.get('nursing_flags'))}")
        st.markdown(
            f"**Acceptance likelihood:** {patient_row.get('patient_acceptance_likelihood', '—')}"
        )
        st.markdown(
            f"**Counselling required:** {patient_row.get('counselling_required', '—')}"
        )

    if show_raw_prompt:
        with st.expander("Generated LLM prompt"):
            st.text_area(
                "Prompt",
                str(patient_row.get("llm_prompt", "")),
                height=340,
                disabled=True,
                key=f"prompt_{patient_row.get('patient_id')}_{patient_row.get('encounter_id')}",
            )


def render_decision_form(
    patient_row: pd.Series,
    user_profile: dict,
    *,
    action_options: list[str],
    action_label: str,
    comments_label: str,
    key_prefix: str,
    source_queue: str,
    allow_bucket_override: bool = True,
) -> None:
    """Render and persist a role-specific human review form."""
    st.markdown("### Record review action")
    action = st.selectbox(
        action_label,
        action_options,
        key=f"{key_prefix}_action",
    )

    override_bucket = None
    if allow_bucket_override and "override" in action.lower():
        override_bucket = st.selectbox(
            "Override / reclassified bucket",
            [
                BUCKET_POTENTIAL,
                BUCKET_MONITORING,
                BUCKET_CLINICAL_REVIEW,
                BUCKET_NOT_SUITABLE,
            ],
            key=f"{key_prefix}_override_bucket",
        )

    comments = st.text_area(comments_label, key=f"{key_prefix}_comments")
    st.info(
        "The dashboard output is advisory. The user remains accountable for reviewing "
        "the available clinical information and recording the rationale for overrides."
    )

    if st.button("Submit Review Decision", key=f"{key_prefix}_submit"):
        if ("override" in action.lower() or "reclassify" in action.lower()) and not comments.strip():
            st.error("Please provide an override or reclassification rationale.")
            return

        record = build_review_record(
            patient_row,
            user_profile.get("designation", "Unknown"),
            action,
            comments,
            source_queue=source_queue,
            override_bucket=override_bucket,
        )
        save_review_decision(record)
        clear_review_log_cache()
        st.success("Review decision saved to the audit log.")


def render_llm_explanation(patient_row: pd.Series, *, key_prefix: str) -> None:
    """Render the optional Bedrock explanation tool for one patient."""
    st.markdown("### LLM explanation support")
    st.caption(
        "The generated explanation must be checked against the source clinical information."
    )
    if st.button("Generate Explanation", key=f"{key_prefix}_llm_button"):
        with st.spinner("Generating explanation..."):
            output = call_bedrock_llm(str(patient_row.get("llm_prompt", "")))
        st.markdown(output)