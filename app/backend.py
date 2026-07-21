"""Backend services for the ENCHANTED Model 1 Streamlit dashboard.

The backend owns:
- sample-data loading
- deterministic rule-based screening
- prototype Random Forest scoring
- workflow routing and advisory recommendations
- review/audit persistence
- patient-list filtering and sorting
- optional AWS Bedrock explanation calls

The current rules remain an interim MVP. They use only fields available in the
sample CSV and do not yet represent the complete 104-field catalogue or final
Epic/EAI mappings.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterable

import boto3
import joblib
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError
from pandas.errors import EmptyDataError
from streamlit.errors import StreamlitSecretNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("ENCHANTED_DATA_PATH", PROJECT_ROOT / "shortlisted.csv"))
MODEL_PATH = Path(
    os.getenv(
        "ENCHANTED_MODEL_PATH",
        PROJECT_ROOT / "models" / "enchanted_model1_random_forest.joblib",
    )
)
REVIEW_LOG_PATH = Path(
    os.getenv("ENCHANTED_REVIEW_LOG_PATH", PROJECT_ROOT / "case_manager_review_log.csv")
)
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


BUCKET_POTENTIAL = "Potential CH Candidate"
BUCKET_MONITORING = "Pending Monitoring"
BUCKET_CLINICAL_REVIEW = "Needs Clinical Review"
BUCKET_NOT_SUITABLE = "Not Suitable at Current Review"

WORKFLOW_PENDING_CM = "Pending CM"
WORKFLOW_MONITORING = "Monitoring"
WORKFLOW_PENDING_CLINICIAN = "Pending Clinician"
WORKFLOW_NO_ACTIVE_REVIEW = "No Active Review"


# The saved Random Forest expects this exact feature order. Do not append the
# 104 criteria fields until a replacement model has been retrained.
FEATURES = [
    "age",
    "los_days",
    "days_to_edd",
    "copd_flag",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature",
    "spo2",
    "oxygen_flow_rate",
    "news2",
    "hb",
    "platelet",
    "anc",
    "sodium",
    "potassium",
    "pending_surgery_flag",
    "active_procedure_flag",
    "active_precaution_flag",
    "active_iv_med_flag",
]

ROLE_DESCRIPTIONS = {
    "Case Manager": "Assigned cases, monitoring, confirmation and escalation",
    "Clinician": "Escalated cases requiring clinical judgement or override review",
    "JCH Referral Team": "Validated candidates, transfer eligibility and operational readiness",
}

BASE_DISPLAY_COLUMNS = [
    "patient_id",
    "encounter_id",
    "ward",
    "specialty",
    "age",
    "los_days",
    "days_to_edd",
    "rule_category",
    "red_flags",
    "amber_flags",
    "review_flags",
    "risk_score",
    "risk_band",
    "workflow_status",
    "service_suitability",
    "nursing_status",
    "patient_acceptance_likelihood",
    "right_siting_recommendation",
    "ai_recommendation",
]

COLUMN_LABELS = {
    "patient_id": "Patient ID",
    "encounter_id": "Encounter ID",
    "ward": "Ward / Floor",
    "specialty": "Specialty",
    "age": "Age",
    "sex": "Sex",
    "los_days": "Days in Hospital",
    "days_to_edd": "Days to EDD",
    "admission_datetime": "Admission Date",
    "rule_category": "Screening Bucket",
    "red_flags": "Hard Exclusion Flags",
    "amber_flags": "Monitoring Flags",
    "review_flags": "Clinical Review Flags",
    "risk_score": "Prototype Risk Score",
    "risk_band": "Prototype Risk Band",
    "workflow_status": "Workflow Status",
    "service_need": "Service Need",
    "service_suitability": "Service Suitability",
    "nursing_status": "Nursing Assessment",
    "nursing_flags": "Nursing Flags",
    "patient_acceptance_likelihood": "Acceptance Likelihood",
    "counselling_required": "Counselling Required",
    "right_siting_recommendation": "Suggested Review Pathway",
    "ai_recommendation": "Advisory Next Action",
    "assigned_case_manager": "Assigned Case Manager",
}

SORT_OPTIONS = {
    "Risk Score": "risk_score",
    "Days in Hospital": "los_days",
    "Status": "workflow_status",
    "Age": "age",
    "Admission Date": "admission_datetime",
}


def _number(row: pd.Series, field: str) -> float:
    """Safely convert one row value to a numeric value or NaN."""
    return pd.to_numeric(row.get(field), errors="coerce")


def _flag_is_true(row: pd.Series, field: str) -> bool:
    """Interpret common prototype boolean/coded values safely."""
    value = row.get(field, 0)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "active", "positive"}
    return bool(value)


def rule_based_screening(row: pd.Series) -> tuple[str, list[str], list[str], list[str]]:
    """Return a four-bucket screening result using fields in the sample CSV.

    Priority is:
    hard exclusion > clinical review > dynamic monitoring > potential candidate.
    """
    hard_exclusion_flags: list[str] = []
    monitoring_flags: list[str] = []
    clinical_review_flags: list[str] = []

    age = _number(row, "age")
    if pd.notna(age) and age < 18:
        hard_exclusion_flags.append("Age < 18 years")

    if _flag_is_true(row, "pregnancy_flag"):
        hard_exclusion_flags.append("Pregnancy")

    oxygen_device = str(row.get("oxygen_device", "")).strip().lower()
    oxygen_flow = _number(row, "oxygen_flow_rate")
    spo2 = _number(row, "spo2")
    copd = _flag_is_true(row, "copd_flag")

    if any(device in oxygen_device for device in ("venti", "non-rebreather", "nrm")):
        hard_exclusion_flags.append("Venti mask / non-rebreather mask")

    on_room_air = (
        (pd.notna(oxygen_flow) and oxygen_flow == 0)
        or oxygen_device in {"", "room air", "ra", "none"}
    )
    if on_room_air and pd.notna(spo2):
        if copd and spo2 < 88:
            hard_exclusion_flags.append("SpO2 < 88% on room air with special target")
        elif not copd and spo2 < 92:
            hard_exclusion_flags.append("SpO2 < 92% on room air")

    if pd.notna(oxygen_flow) and oxygen_flow > 2:
        monitoring_flags.append("Supplemental oxygen > 2 L/min via nasal prongs")

    temperature = _number(row, "temperature")
    if pd.notna(temperature):
        if temperature < 35:
            hard_exclusion_flags.append("Temperature < 35°C")
        elif temperature >= 38:
            monitoring_flags.append("Temperature ≥ 38°C")

    heart_rate = _number(row, "heart_rate")
    if pd.notna(heart_rate):
        if heart_rate < 40 or heart_rate >= 120:
            hard_exclusion_flags.append("Heart rate outside 40–119 bpm")
        elif 40 <= heart_rate <= 59 or 100 <= heart_rate <= 119:
            monitoring_flags.append("Heart rate in amber range")

    systolic_bp = _number(row, "systolic_bp")
    if pd.notna(systolic_bp):
        if systolic_bp < 80 or systolic_bp > 180:
            hard_exclusion_flags.append("Systolic BP outside 80–180 mmHg")
        elif 80 <= systolic_bp <= 89 or 161 <= systolic_bp <= 180:
            monitoring_flags.append("Systolic BP in amber range")

    diastolic_bp = _number(row, "diastolic_bp")
    if pd.notna(diastolic_bp):
        if diastolic_bp < 40 or diastolic_bp > 110:
            hard_exclusion_flags.append("Diastolic BP outside 40–110 mmHg")
        elif 40 <= diastolic_bp <= 49 or 91 <= diastolic_bp <= 110:
            monitoring_flags.append("Diastolic BP in amber range")

    # Interim case-manager interpretation of sample-data laboratory fields.
    hb = _number(row, "hb")
    if pd.notna(hb) and hb <= 9:
        monitoring_flags.append("Haemoglobin ≤ 9")

    platelet = _number(row, "platelet")
    if pd.notna(platelet) and (platelet < 100 or platelet >= 1000):
        monitoring_flags.append("Platelet count outside case-manager review range")

    anc = _number(row, "anc")
    if pd.notna(anc) and anc < 1.0:
        monitoring_flags.append("ANC < 1.0")

    sodium = _number(row, "sodium")
    if pd.notna(sodium) and (sodium < 130 or sodium >= 150):
        monitoring_flags.append("Serum sodium outside 130–149")

    potassium = _number(row, "potassium")
    if pd.notna(potassium) and (potassium < 3.0 or potassium >= 5.3):
        monitoring_flags.append("Serum potassium outside 3.0–5.2")

    if _flag_is_true(row, "pending_surgery_flag"):
        monitoring_flags.append("Planned / pending surgery requires reassessment")

    # These generic sample fields do not identify the exact medication,
    # procedure or precaution. Route them for human review until EAI mapping is
    # sufficiently granular.
    if _flag_is_true(row, "active_iv_med_flag"):
        clinical_review_flags.append("Active IV medication requires medication-specific review")

    if _flag_is_true(row, "active_procedure_flag"):
        clinical_review_flags.append("Active procedure requires clinical review")

    if _flag_is_true(row, "active_precaution_flag"):
        clinical_review_flags.append("Active precaution requires clearance / review")

    if hard_exclusion_flags:
        bucket = BUCKET_NOT_SUITABLE
    elif clinical_review_flags:
        bucket = BUCKET_CLINICAL_REVIEW
    elif monitoring_flags:
        bucket = BUCKET_MONITORING
    else:
        bucket = BUCKET_POTENTIAL

    return bucket, hard_exclusion_flags, monitoring_flags, clinical_review_flags


def nursing_operational_assessment(row: pd.Series) -> tuple[str, list[str]]:
    """Assess nursing and operational complexity without making a final decision."""
    flags: list[str] = []

    if str(row.get("isolation_requirement", "")).strip().lower() == "airborne":
        flags.append("Airborne isolation / clearance review")
    if str(row.get("infectious_status", "")).strip().lower() == "active infection":
        flags.append("Active infectious concern")
    if str(row.get("wound_care_need", "")).strip().lower() == "complex":
        flags.append("Complex wound care")
    if _flag_is_true(row, "behavioural_concern_flag"):
        flags.append("Behavioural concern")
    if str(row.get("nursing_complexity", "")).strip().lower() == "high":
        flags.append("High nursing complexity")
    if _flag_is_true(row, "social_support_concern"):
        flags.append("Social support concern")

    return ("Nursing Review Required" if flags else "Nursing Suitable"), flags


def service_suitability_assessment(row: pd.Series) -> str:
    """Return the prototype service-scope assessment."""
    suitable_services = {
        "Rehabilitation",
        "Long-term IV antibiotics",
        "Wound care",
        "Nursing care",
        "Lower-acuity monitoring",
    }
    service_need = row.get("service_need")
    if service_need in suitable_services:
        return "JCH Service Suitable"
    if service_need == "Specialist acute monitoring":
        return "Service Not Suitable for JCH"
    return "Service Suitability Unclear"


def risk_band(probability: float) -> str:
    """Convert one prototype probability into a display band."""
    if probability >= 0.30:
        return "High Risk"
    if probability >= 0.15:
        return "Medium Risk"
    return "Low Risk"


def right_siting_recommendation(row: pd.Series) -> str:
    """Combine screening, service, nursing, risk and acceptance factors."""
    bucket = row["rule_category"]
    if bucket == BUCKET_NOT_SUITABLE:
        return "Continue Acute Hospital care"
    if bucket == BUCKET_MONITORING:
        return "Monitor and reassess"
    if bucket == BUCKET_CLINICAL_REVIEW:
        return "Further clinical / nursing review required"

    if row.get("service_suitability") == "Service Not Suitable for JCH":
        return "Further clinical / service review required"
    if row.get("nursing_status") == "Nursing Review Required":
        return "Further clinical / nursing review required"

    if (
        row.get("service_need") in {"Long-term IV antibiotics", "Lower-acuity monitoring"}
        and str(row.get("nursing_complexity", "")).strip().lower() == "low"
        and row.get("patient_acceptance_likelihood") in {"High", "Medium"}
    ):
        return "Hospital-at-Home review"

    if (
        row.get("service_suitability") == "JCH Service Suitable"
        and row.get("nursing_status") == "Nursing Suitable"
        and row.get("risk_band") in {"Low Risk", "Medium Risk"}
    ):
        return "Community Hospital review"

    return "Further clinical / nursing review required"


def ai_review_recommendation(row: pd.Series) -> str:
    """Return an advisory next action for the current bucket."""
    bucket = row["rule_category"]
    if bucket == BUCKET_NOT_SUITABLE:
        return "Not suitable at current review; reassess only if relevant information changes"
    if bucket == BUCKET_MONITORING:
        return "Continue monitoring and reassess after refreshed data or pending results"
    if bucket == BUCKET_CLINICAL_REVIEW:
        return "Clinical judgement is required before CH shortlisting"

    if row.get("right_siting_recommendation") == "Hospital-at-Home review":
        return "Consider Hospital-at-Home review"
    if row.get("right_siting_recommendation") == "Community Hospital review":
        if row.get("patient_acceptance_likelihood") == "Low":
            return "Potential CH candidate; counselling likely required"
        return "Potential CH candidate for case manager confirmation"
    return "Further review is required before a right-siting recommendation"


def assign_case_manager(row: pd.Series) -> str:
    """Assign a prototype case manager from the sample ward field."""
    ward_to_manager = {
        "Ward A": "Case Manager A",
        "Ward B": "Case Manager B",
        "Ward C": "Case Manager C",
    }
    return ward_to_manager.get(row.get("ward"), "Case Manager Pool")


def workflow_status(row: pd.Series) -> str:
    """Route each patient to the future-state queue."""
    bucket = row["rule_category"]
    if bucket == BUCKET_NOT_SUITABLE:
        return WORKFLOW_NO_ACTIVE_REVIEW
    if bucket == BUCKET_MONITORING:
        return WORKFLOW_MONITORING
    if bucket == BUCKET_CLINICAL_REVIEW:
        return WORKFLOW_PENDING_CLINICIAN
    if row.get("risk_band") == "High Risk":
        return WORKFLOW_PENDING_CLINICIAN
    if row.get("right_siting_recommendation") in {
        "Community Hospital review",
        "Hospital-at-Home review",
    }:
        return WORKFLOW_PENDING_CM
    return WORKFLOW_PENDING_CLINICIAN


@st.cache_resource
def load_model() -> Any | None:
    """Load the prototype model, returning None when it is unavailable."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None
    except Exception as error:  # pragma: no cover - depends on model artefact
        st.warning(f"Prototype model could not be loaded: {error}")
        return None


def _ensure_columns(data: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Add missing prototype fields as null columns to prevent UI crashes."""
    result = data.copy()
    for column in required:
        if column not in result.columns:
            result[column] = pd.NA
    return result


@st.cache_data
def load_patient_worklist() -> pd.DataFrame:
    """Load sample patients and calculate all dashboard-derived fields."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sample patient file was not found at {DATA_PATH}. "
            "Set ENCHANTED_DATA_PATH or place shortlisted.csv at the project root."
        )

    data = pd.read_csv(DATA_PATH)
    essential_defaults = {
        "pregnancy_flag": 0,
        "copd_flag": 0,
        "pending_surgery_flag": 0,
        "active_procedure_flag": 0,
        "active_precaution_flag": 0,
        "active_iv_med_flag": 0,
        "behavioural_concern_flag": 0,
        "social_support_concern": 0,
        "oxygen_device": "",
        "isolation_requirement": "None",
        "infectious_status": "None",
        "wound_care_need": "None",
        "nursing_complexity": "Low",
        "service_need": "Unknown",
        "patient_acceptance_likelihood": "Unknown",
        "counselling_required": "Unknown",
        "ward": "Unknown",
        "specialty": "Unknown",
    }
    for column, default in essential_defaults.items():
        if column not in data.columns:
            data[column] = default

    data = _ensure_columns(
        data,
        set(FEATURES)
        | {
            "patient_id",
            "encounter_id",
            "admission_datetime",
            "sex",
        },
    )

    data[["rule_category", "red_flags", "amber_flags", "review_flags"]] = data.apply(
        lambda row: pd.Series(rule_based_screening(row)), axis=1
    )
    data[["nursing_status", "nursing_flags"]] = data.apply(
        lambda row: pd.Series(nursing_operational_assessment(row)), axis=1
    )
    data["service_suitability"] = data.apply(service_suitability_assessment, axis=1)

    data["risk_score"] = pd.Series(float("nan"), index=data.index, dtype="float64")
    data["risk_band"] = "Not applicable"

    model = load_model()
    eligible_mask = data["rule_category"] != BUCKET_NOT_SUITABLE
    model_fields_available = all(field in data.columns for field in FEATURES)

    if model is not None and eligible_mask.any() and model_fields_available:
        try:
            feature_frame = data.loc[eligible_mask, FEATURES].apply(
                pd.to_numeric, errors="coerce"
            )
            if not feature_frame.isna().any().any():
                probabilities = model.predict_proba(feature_frame)[:, 1]
                data.loc[eligible_mask, "risk_score"] = probabilities
                data.loc[eligible_mask, "risk_band"] = [risk_band(x) for x in probabilities]
            else:
                st.info(
                    "Prototype risk scores are unavailable for rows with missing model features."
                )
        except Exception as error:  # pragma: no cover - model-specific
            st.warning(f"Prototype risk scoring was skipped: {error}")

    data["right_siting_recommendation"] = data.apply(
        right_siting_recommendation, axis=1
    )
    data["ai_recommendation"] = data.apply(ai_review_recommendation, axis=1)
    data["assigned_case_manager"] = data.apply(assign_case_manager, axis=1)
    data["workflow_status"] = data.apply(workflow_status, axis=1)
    data["admission_datetime"] = pd.to_datetime(
        data["admission_datetime"], dayfirst=True, errors="coerce"
    )
    data["llm_prompt"] = data.apply(build_llm_prompt, axis=1)
    return data


def build_llm_prompt(row: pd.Series) -> str:
    """Build a patient-specific explanation prompt."""
    return f"""
Summary of the patient's ENCHANTED Model 1 screening output:

Patient ID: {row.get('patient_id')}
Encounter ID: {row.get('encounter_id')}

Screening output:
- Actionable bucket: {row.get('rule_category')}
- Hard exclusion flags: {row.get('red_flags')}
- Monitoring flags: {row.get('amber_flags')}
- Clinical review flags: {row.get('review_flags')}
- Prototype predictive risk score: {row.get('risk_score')}
- Prototype predictive risk band: {row.get('risk_band')}

Selected clinical values available in the sample dataset:
- Age: {row.get('age')}
- COPD flag: {row.get('copd_flag')}
- Systolic BP: {row.get('systolic_bp')}
- Diastolic BP: {row.get('diastolic_bp')}
- Heart rate: {row.get('heart_rate')}
- Temperature: {row.get('temperature')}
- SpO2: {row.get('spo2')}
- Oxygen device: {row.get('oxygen_device')}
- Oxygen flow rate: {row.get('oxygen_flow_rate')}
- Haemoglobin: {row.get('hb')}
- Platelet: {row.get('platelet')}
- ANC: {row.get('anc')}
- Sodium: {row.get('sodium')}
- Potassium: {row.get('potassium')}

Operational assessment:
- Service need: {row.get('service_need')}
- Service suitability: {row.get('service_suitability')}
- Nursing status: {row.get('nursing_status')}
- Nursing flags: {row.get('nursing_flags')}
- Patient acceptance likelihood: {row.get('patient_acceptance_likelihood')}
- Counselling required: {row.get('counselling_required')}
- Suggested pathway: {row.get('right_siting_recommendation')}
- Advisory next action: {row.get('ai_recommendation')}

Produce four sections:
1. Explanation of screening bucket.
2. Key points requiring verification.
3. Advisory next action, using the wording above.
4. Reminder that the output is advisory and final referral/transfer decisions
   remain with the case manager, clinician and care team.

Do not invent clinical information. State that the prototype uses sample data,
partial criteria coverage and provisional field mappings.
""".strip()


def _get_secret(key: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


@st.cache_resource
def get_bedrock_client() -> Any | None:
    """Create a Bedrock client only when the required secrets exist."""
    access_key = _get_secret("AWS_ACCESS_KEY_ID")
    secret_key = _get_secret("AWS_SECRET_ACCESS_KEY")
    region = _get_secret("AWS_DEFAULT_REGION")
    if not all([access_key, secret_key, region]):
        return None

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return session.client("bedrock-runtime", region_name=region)


def call_bedrock_llm(prompt: str) -> str:
    """Call the configured Bedrock model or return an explanatory fallback."""
    client = get_bedrock_client()
    if client is None:
        return (
            "Bedrock explanation is not configured. Add AWS credentials and region "
            "to Streamlit secrets to enable this feature."
        )

    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 700, "temperature": 0.2},
        )
        return response["output"]["message"]["content"][0]["text"]
    except ClientError as error:
        return f"Bedrock ClientError: {error.response['Error']['Message']}"
    except Exception as error:  # pragma: no cover - network/service dependent
        return f"Unexpected error calling Bedrock: {error}"


def get_role_scoped_worklist(data: pd.DataFrame, user_profile: dict[str, Any]) -> pd.DataFrame:
    """Return the worklist appropriate for the authenticated role."""
    role = user_profile.get("designation")
    if role == "Case Manager":
        assigned = user_profile.get("assigned_case_manager")
        if assigned:
            return data[data["assigned_case_manager"] == assigned].copy()
        return data.iloc[0:0].copy()

    if role == "Clinician":
        return data[
            (data["workflow_status"] == WORKFLOW_PENDING_CLINICIAN)
            | (data["rule_category"] == BUCKET_CLINICAL_REVIEW)
        ].copy()

    if role == "JCH Referral Team":
        return data.copy()

    return data.iloc[0:0].copy()


def apply_category_filter(data: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter by a screening bucket or workflow queue."""
    if category in {
        BUCKET_POTENTIAL,
        BUCKET_MONITORING,
        BUCKET_CLINICAL_REVIEW,
        BUCKET_NOT_SUITABLE,
    }:
        return data[data["rule_category"] == category]
    if category in {
        WORKFLOW_PENDING_CM,
        WORKFLOW_MONITORING,
        WORKFLOW_PENDING_CLINICIAN,
        WORKFLOW_NO_ACTIVE_REVIEW,
    }:
        return data[data["workflow_status"] == category]
    return data


def apply_search(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search across common worklist fields."""
    if not query.strip() or data.empty:
        return data
    searchable = [
        column
        for column in (
            "patient_id",
            "encounter_id",
            "ward",
            "specialty",
            "service_need",
            "rule_category",
            "risk_band",
            "workflow_status",
        )
        if column in data.columns
    ]
    blob = data[searchable].fillna("").astype(str).agg(" ".join, axis=1)
    return data[blob.str.lower().str.contains(query.strip().lower(), regex=False)]


def apply_advanced_filters(
    data: pd.DataFrame,
    wards: list[str],
    statuses: list[str],
    risk_bands: list[str],
    age_range: tuple[int, int] | None,
    specialties: list[str],
) -> pd.DataFrame:
    """Apply the advanced worklist filters."""
    result = data.copy()
    if wards and "ward" in result.columns:
        result = result[result["ward"].isin(wards)]
    if statuses and "workflow_status" in result.columns:
        result = result[result["workflow_status"].isin(statuses)]
    if risk_bands and "risk_band" in result.columns:
        result = result[result["risk_band"].isin(risk_bands)]
    if specialties and "specialty" in result.columns:
        result = result[result["specialty"].isin(specialties)]
    if age_range is not None and "age" in result.columns:
        ages = pd.to_numeric(result["age"], errors="coerce")
        result = result[ages.between(age_range[0], age_range[1], inclusive="both")]
    return result


def sort_worklist(data: pd.DataFrame, sort_label: str, ascending: bool) -> pd.DataFrame:
    """Sort the worklist safely."""
    column = SORT_OPTIONS.get(sort_label)
    if not column or column not in data.columns:
        return data
    return data.sort_values(
        by=column,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )


def filter_worklist(
    data: pd.DataFrame,
    category: str,
    search_query: str,
    wards: list[str],
    statuses: list[str],
    risk_bands: list[str],
    age_range: tuple[int, int] | None,
    specialties: list[str],
    sort_label: str,
    sort_ascending: bool,
) -> pd.DataFrame:
    """Run the complete filtering pipeline."""
    result = apply_category_filter(data, category)
    result = apply_search(result, search_query)
    result = apply_advanced_filters(
        result, wards, statuses, risk_bands, age_range, specialties
    )
    return sort_worklist(result, sort_label, sort_ascending)


def build_review_record(
    patient_row: pd.Series,
    selected_role: str,
    final_decision: str,
    review_comments: str,
    *,
    source_queue: str | None = None,
    override_bucket: str | None = None,
) -> dict[str, Any]:
    """Create one auditable review record."""
    return {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dashboard_role": selected_role,
        "assigned_case_manager": patient_row.get("assigned_case_manager"),
        "patient_id": patient_row.get("patient_id"),
        "encounter_id": patient_row.get("encounter_id"),
        "source_queue": source_queue,
        "original_screening_bucket": patient_row.get("rule_category"),
        "override_bucket": override_bucket,
        "hard_exclusion_flags": patient_row.get("red_flags", []),
        "monitoring_flags": patient_row.get("amber_flags", []),
        "clinical_review_flags": patient_row.get("review_flags", []),
        "workflow_status": patient_row.get("workflow_status"),
        "risk_score": patient_row.get("risk_score"),
        "risk_band": patient_row.get("risk_band"),
        "right_siting_recommendation": patient_row.get("right_siting_recommendation"),
        "ai_recommendation": patient_row.get("ai_recommendation"),
        "final_decision": final_decision,
        "review_comments": review_comments,
    }


def _serialise_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return ""
    return value


def save_review_decision(review_record: dict[str, Any]) -> None:
    """Append one record to the shared CSV audit log."""
    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialised = {key: _serialise_value(value) for key, value in review_record.items()}
    new_row = pd.DataFrame([serialised])
    try:
        existing = pd.read_csv(REVIEW_LOG_PATH)
        updated = pd.concat([existing, new_row], ignore_index=True, sort=False)
    except (FileNotFoundError, EmptyDataError):
        updated = new_row
    updated.to_csv(REVIEW_LOG_PATH, index=False)


@st.cache_data(ttl=5)
def load_review_log() -> pd.DataFrame:
    """Load review history, returning an empty table when none exists."""
    try:
        return pd.read_csv(REVIEW_LOG_PATH)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def clear_review_log_cache() -> None:
    """Refresh the cached audit history after a write."""
    load_review_log.clear()