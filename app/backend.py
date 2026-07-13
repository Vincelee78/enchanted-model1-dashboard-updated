"""Backend logic for the ENCHANTED Model 1 dashboard.

This module keeps non-UI logic away from the Streamlit frontend:
- load sample patient data and the trained Random Forest model
- apply rule-based clinical screening
- calculate risk scores and risk bands
- derive right-siting and workflow recommendations
- build prompts and call AWS Bedrock for LLM explanations
- filter/sort worklists and save review audit records
"""

import datetime

import boto3
import joblib
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError
from pandas.errors import EmptyDataError
from streamlit.errors import StreamlitSecretNotFoundError


DATA_PATH = "shortlisted.csv"
MODEL_PATH = "models/enchanted_model1_random_forest.joblib"
REVIEW_LOG_PATH = "case_manager_review_log.csv"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Columns passed into the trained Random Forest risk model.
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
    "Case Manager": "Assigned cases, approval workflow, escalation, and hold decisions",
    "JCH Referral Team": "All cases, queue monitoring, metrics, and reassignment support",
    "Clinician": "Cases awaiting clinical approval, override review, and oversight",
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
    "rule_category": "Clinical Screening",
    "red_flags": "Red Flags",
    "amber_flags": "Amber Flags",
    "risk_score": "Risk Score",
    "risk_band": "Risk Band",
    "workflow_status": "Status",
    "service_need": "Service Need",
    "service_suitability": "Service Suitability",
    "nursing_status": "Nursing Assessment",
    "nursing_flags": "Nursing Flags",
    "patient_acceptance_likelihood": "Acceptance Likelihood",
    "counselling_required": "Counselling Required",
    "right_siting_recommendation": "Right-Siting Recommendation",
    "ai_recommendation": "AI Recommendation",
}

SORT_OPTIONS = {
    "Risk Score": "risk_score",
    "Days in Hospital": "los_days",
    "Status": "workflow_status",
    "Age": "age",
    "Admission Date": "admission_datetime",
}


def rule_based_screening(row):
    """Classify a patient as Red, Amber, or Green using deterministic rules."""
    red_flags = []
    amber_flags = []

    if row["pregnancy_flag"] == 1:
        red_flags.append("Pregnancy")

    if row["news2"] >= 5:
        red_flags.append("NEWS2 >= 5")
    elif row["news2"] > 0:
        amber_flags.append("NEWS2 > 0")

    if row["oxygen_flow_rate"] > 2:
        red_flags.append("Oxygen flow > 2 L/min")
    elif row["oxygen_flow_rate"] > 0:
        amber_flags.append("Low-flow O2")

    if row["copd_flag"] == 1:
        if row["spo2"] < 88:
            red_flags.append("COPD SpO2 < 88")
        elif row["spo2"] <= 92:
            amber_flags.append("COPD SpO2 88-92")
    else:
        if row["spo2"] < 91:
            red_flags.append("SpO2 < 91")
        elif row["spo2"] < 96:
            amber_flags.append("SpO2 < 96")

    if row["temperature"] >= 38:
        red_flags.append("Temperature >= 38")
    elif row["temperature"] >= 37.5:
        amber_flags.append("Temperature 37.5-37.9")

    if row["heart_rate"] >= 120:
        red_flags.append("Heart rate >= 120")
    elif row["heart_rate"] >= 100:
        amber_flags.append("Heart rate 100-119")

    if row["pending_surgery_flag"] == 1:
        red_flags.append("Pending surgery")

    if row["active_iv_med_flag"] == 1:
        red_flags.append("IV medication")

    if row["active_procedure_flag"] == 1:
        amber_flags.append("Procedure order")

    if row["active_precaution_flag"] == 1:
        amber_flags.append("Precaution order")

    if red_flags:
        category = "Red - No-Go"
    elif amber_flags:
        category = "Amber - Review Required"
    else:
        category = "Green - Potential Candidate"

    return category, red_flags, amber_flags


def nursing_operational_assessment(row):
    """Assess operational nursing suitability and return nursing flags."""
    nursing_flags = []

    if row["isolation_requirement"] == "Airborne":
        nursing_flags.append("Airborne isolation requirement")

    if row["infectious_status"] == "Active infection":
        nursing_flags.append("Active infectious concern")

    if row["wound_care_need"] == "Complex":
        nursing_flags.append("Complex wound care")

    if row["behavioural_concern_flag"] == 1:
        nursing_flags.append("Behavioural concern")

    if row["nursing_complexity"] == "High":
        nursing_flags.append("High nursing complexity")

    if row["social_support_concern"] == 1:
        nursing_flags.append("Social support concern")

    if "Airborne isolation requirement" in nursing_flags:
        nursing_status = "Nursing Not Suitable"
    elif nursing_flags:
        nursing_status = "Nursing Review Required"
    else:
        nursing_status = "Nursing Suitable"

    return nursing_status, nursing_flags


def service_suitability_assessment(row):
    """Check whether the requested service fits the JCH service scope."""
    suitable_services = [
        "Rehabilitation",
        "Long-term IV antibiotics",
        "Wound care",
        "Nursing care",
        "Lower-acuity monitoring",
    ]

    if row["service_need"] in suitable_services:
        return "JCH Service Suitable"

    if row["service_need"] == "Specialist acute monitoring":
        return "Service Not Suitable for JCH"

    return "Service Suitability Unclear"


def risk_band(probability):
    """Convert a model probability into Low, Medium, or High risk."""
    if probability >= 0.30:
        return "High Risk"
    if probability >= 0.15:
        return "Medium Risk"
    return "Low Risk"


def right_siting_recommendation(row):
    """Combine clinical, service, nursing, risk, and acceptance inputs."""
    if row["rule_category"] == "Red - No-Go":
        return "Continue Acute Hospital care"

    if row["service_suitability"] == "Service Not Suitable for JCH":
        return "Continue Acute Hospital care"

    if row["nursing_status"] == "Nursing Not Suitable":
        return "Continue Acute Hospital care"

    if (
        row["service_need"] in ["Long-term IV antibiotics", "Lower-acuity monitoring"]
        and row["nursing_complexity"] == "Low"
        and row["patient_acceptance_likelihood"] in ["High", "Medium"]
        and row["rule_category"] != "Red - No-Go"
    ):
        return "Hospital-at-Home review"

    if (
        row["service_suitability"] == "JCH Service Suitable"
        and row["nursing_status"] == "Nursing Suitable"
        and row["risk_band"] in ["Low Risk", "Medium Risk"]
    ):
        return "Community Hospital review"

    return "Further clinical / nursing review required"


def ai_review_recommendation(row):
    """Produce the short recommendation shown to reviewers in the dashboard."""
    if row["right_siting_recommendation"] == "Continue Acute Hospital care":
        return "Continue Acute Hospital care; not suitable for immediate CH transfer"

    if row["right_siting_recommendation"] == "Hospital-at-Home review":
        return "Consider Hospital-at-Home review"

    if row["right_siting_recommendation"] == "Community Hospital review":
        if row["patient_acceptance_likelihood"] == "Low":
            return "Community Hospital review; counselling likely required"
        return "Community Hospital referral review"

    if row["right_siting_recommendation"] == "Further clinical / nursing review required":
        return "Further clinical / nursing review required before right-siting decision"

    return "Pending review"


def assign_case_manager(row):
    """Assign a prototype case manager based on ward."""
    ward_to_manager = {
        "Ward A": "Case Manager A",
        "Ward B": "Case Manager B",
        "Ward C": "Case Manager C",
    }
    return ward_to_manager.get(row["ward"], "Case Manager Pool")


def workflow_status(row):
    """Route each case to the case-manager or clinician task queue."""
    if row["rule_category"] == "Red - No-Go":
        return "Pending Clinician"

    if row["risk_band"] == "High Risk":
        return "Pending Clinician"

    if row["right_siting_recommendation"] in [
        "Community Hospital review",
        "Hospital-at-Home review",
    ]:
        return "Pending CM"

    return "Pending Clinician"


@st.cache_resource
def load_model():
    """Load the saved Random Forest model once per Streamlit process."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_patient_worklist():
    """Load patient data and enrich it with all dashboard-derived fields.

    The returned dataframe is what the frontend uses for task tabs, patient
    lists, review forms, and LLM prompts.
    """
    data = pd.read_csv(DATA_PATH)
    model = load_model()

    # Prototype-only outcome label. Replace this with real outcome data when
    # training or validating the model on production data.
    data["rebound_72h"] = (
        (data["news2"] >= 5)
        | (data["oxygen_flow_rate"] > 2)
        | (data["spo2"] < 91)
        | (data["hb"] < 9)
        | (data["sodium"] < 130)
        | (data["potassium"] > 5.5)
    ).astype(int)

    data[["rule_category", "red_flags", "amber_flags"]] = data.apply(
        lambda row: pd.Series(rule_based_screening(row)),
        axis=1,
    )

    data[["nursing_status", "nursing_flags"]] = data.apply(
        lambda row: pd.Series(nursing_operational_assessment(row)),
        axis=1,
    )

    data["service_suitability"] = data.apply(service_suitability_assessment, axis=1)
    data["risk_score"] = None
    data["risk_band"] = "Not applicable"

    # Red cases are already excluded by rules, so only Amber/Green cases are
    # sent to the ML risk model.
    eligible_mask = data["rule_category"].isin(
        ["Amber - Review Required", "Green - Potential Candidate"]
    )

    if eligible_mask.any():
        data.loc[eligible_mask, "risk_score"] = model.predict_proba(
            data.loc[eligible_mask, FEATURES]
        )[:, 1]

        data.loc[eligible_mask, "risk_band"] = data.loc[
            eligible_mask, "risk_score"
        ].apply(risk_band)

    data["right_siting_recommendation"] = data.apply(
        right_siting_recommendation,
        axis=1,
    )
    data["ai_recommendation"] = data.apply(ai_review_recommendation, axis=1)
    data["assigned_case_manager"] = data.apply(assign_case_manager, axis=1)
    data["workflow_status"] = data.apply(workflow_status, axis=1)
    data["admission_datetime"] = pd.to_datetime(
        data["admission_datetime"],
        dayfirst=True,
        errors="coerce",
    )
    data["llm_prompt"] = data.apply(build_llm_prompt, axis=1)

    return data


def build_llm_prompt(row):
    """Build the patient-specific prompt sent to the LLM explanation layer."""
    return f"""
Summary of patient's clinical screening and AI risk assessment:

Patient ID: {row["patient_id"]}
Encounter ID: {row["encounter_id"]}

Patient screening output:
- Rule-based category: {row["rule_category"]}
- Red flags: {row["red_flags"]}
- Amber flags: {row["amber_flags"]}
- Predictive risk score: {row["risk_score"]}
- Predictive risk band: {row["risk_band"]}

Key clinical values:
- Age: {row["age"]}
- COPD flag: {row["copd_flag"]}
- Systolic BP: {row["systolic_bp"]}
- Diastolic BP: {row["diastolic_bp"]}
- Heart rate: {row["heart_rate"]}
- Temperature: {row["temperature"]}
- SpO2: {row["spo2"]}
- Oxygen device: {row["oxygen_device"]}
- Oxygen flow rate: {row["oxygen_flow_rate"]}
- NEWS2: {row["news2"]}
- Hb: {row["hb"]}
- Platelet: {row["platelet"]}
- ANC: {row["anc"]}
- Sodium: {row["sodium"]}
- Potassium: {row["potassium"]}
- Pending surgery flag: {row["pending_surgery_flag"]}
- Active procedure flag: {row["active_procedure_flag"]}
- Active precaution flag: {row["active_precaution_flag"]}
- Active IV medication flag: {row["active_iv_med_flag"]}

Service / nursing / acceptance assessment:
- Service need: {row["service_need"]}
- Service suitability: {row["service_suitability"]}
- Nursing status: {row["nursing_status"]}
- Nursing flags: {row["nursing_flags"]}
- Patient acceptance likelihood: {row["patient_acceptance_likelihood"]}
- Counselling required: {row["counselling_required"]}
- Right-siting recommendation: {row["right_siting_recommendation"]}

Please produce the response in this exact format:

**1. Explanation of Screening Output**
Provide a short explanation of why the patient received this screening output. Explain both:
- the rule-based screening category; and
- the predictive risk score / risk band from the machine learning model.

**2. Key Review Points**
List key points that the case manager or clinician should review, based only on the clinical values provided above.

**3. Final AI-Supported Right-Siting Recommendation**
Explain the final AI-supported recommendation based on:
- clinical screening;
- predictive risk band;
- service suitability;
- nursing / operational assessment;
- acceptance / counselling needs;
- right-siting recommendation.

Use the final AI-supported recommendation exactly as stated:
{row["ai_recommendation"]}

Do not make a final transfer decision. The final decision remains with the clinical team.

Apply the following interpretation rules strictly:
- If the rule-based category is "Red - No-Go", explain that the patient is not suitable for CH referral at this stage due to rule-based exclusion criteria.
- If the rule-based category is "Amber - Review Required" and the predictive risk band is "High Risk", explain that the patient is not suitable for immediate CH referral and requires priority clinical review. Do not describe this as a rule-based red flag exclusion.
- If the rule-based category is "Amber - Review Required" and the predictive risk band is "Medium Risk", explain that further clinical review is needed before CH referral.
- If the predictive risk band is "Low Risk", explain that the patient may proceed for CH referral review, subject to case manager / clinician assessment.
- Do not invent red flag exclusions if the Red flags list is empty.
- Do not make a final transfer or referral decision on behalf of the clinical team.

**4. Clinical Decision Reminder**
State that the final AI-supported recommendation is advisory only. The final referral / transfer decision remains with the case manager, clinician, and care team.
"""


@st.cache_resource
def get_bedrock_client():
    """Create an AWS Bedrock runtime client when AWS secrets are configured."""
    access_key = get_secret("AWS_ACCESS_KEY_ID")
    secret_key = get_secret("AWS_SECRET_ACCESS_KEY")
    region = get_secret("AWS_DEFAULT_REGION")

    if not all([access_key, secret_key, region]):
        return None

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    return session.client(
        service_name="bedrock-runtime",
        region_name=region,
    )


def get_secret(key, default=None):
    """Read optional Streamlit secrets safely in local development."""
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


def call_bedrock_llm(prompt):
    """Send the generated prompt to AWS Bedrock and return the explanation text."""
    client = get_bedrock_client()

    if client is None:
        return (
            "LLM explanation is not configured yet. Add AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION to Streamlit secrets "
            "to enable Bedrock explanations. The generated prompt above can still "
            "be reviewed manually."
        )

    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 500, "temperature": 0.2},
        )
        return response["output"]["message"]["content"][0]["text"]
    except ClientError as error:
        return f"Bedrock ClientError: {error.response['Error']['Message']}"
    except Exception as error:
        return f"Unexpected error calling Bedrock: {str(error)}"


def apply_role_scope(data, role, case_manager):
    """Restrict a dataframe to the cases visible for a selected role."""
    if role == "Case Manager":
        return data[data["assigned_case_manager"] == case_manager]

    if role == "Clinician":
        return data[data["workflow_status"] == "Pending Clinician"]

    return data


def apply_category_filter(data, category):
    """Filter cases by dashboard category buttons such as Green or Pending CM."""
    if category == "Green":
        return data[data["rule_category"] == "Green - Potential Candidate"]
    if category == "Amber":
        return data[data["rule_category"] == "Amber - Review Required"]
    if category == "Red":
        return data[data["rule_category"] == "Red - No-Go"]
    if category == "Pending CM":
        return data[data["workflow_status"] == "Pending CM"]
    if category == "Pending Clinician":
        return data[data["workflow_status"] == "Pending Clinician"]
    return data


def apply_search(data, query):
    """Search across patient, encounter, ward, specialty, and status fields."""
    if not query.strip():
        return data

    searchable_cols = [
        "patient_id",
        "encounter_id",
        "ward",
        "specialty",
        "service_need",
        "rule_category",
        "risk_band",
        "workflow_status",
    ]
    query = query.strip().lower()
    search_blob = data[searchable_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return data[search_blob.str.lower().str.contains(query, regex=False)]


def apply_advanced_filters(data, wards, statuses, risk_bands, age_range, specialties):
    """Apply the advanced filter panel selections to the worklist."""
    filtered = data.copy()

    if wards:
        filtered = filtered[filtered["ward"].isin(wards)]

    if statuses:
        filtered = filtered[filtered["workflow_status"].isin(statuses)]

    if risk_bands:
        filtered = filtered[filtered["risk_band"].isin(risk_bands)]

    if specialties:
        filtered = filtered[filtered["specialty"].isin(specialties)]

    return filtered[
        filtered["age"].between(age_range[0], age_range[1], inclusive="both")
    ]


def sort_worklist(data, sort_label, ascending):
    """Sort the worklist using the user-facing sort label from the UI."""
    return data.sort_values(
        by=SORT_OPTIONS[sort_label],
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )


def filter_worklist(
    data,
    category,
    search_query,
    wards,
    statuses,
    risk_bands,
    age_range,
    specialties,
    sort_label,
    sort_ascending,
):
    """Apply category, search, advanced filters, and sorting in one pipeline."""
    filtered = apply_category_filter(data, category)
    filtered = apply_search(filtered, search_query)
    filtered = apply_advanced_filters(
        filtered,
        wards,
        statuses,
        risk_bands,
        age_range,
        specialties,
    )
    return sort_worklist(filtered, sort_label, sort_ascending)


def build_review_record(patient_row, selected_role, final_decision, review_comments):
    """Create one audit-log row for a submitted review decision."""
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dashboard_role": selected_role,
        "assigned_case_manager": patient_row["assigned_case_manager"],
        "patient_id": patient_row["patient_id"],
        "encounter_id": patient_row["encounter_id"],
        "rule_category": patient_row["rule_category"],
        "workflow_status": patient_row["workflow_status"],
        "risk_score": patient_row["risk_score"],
        "risk_band": patient_row["risk_band"],
        "right_siting_recommendation": patient_row["right_siting_recommendation"],
        "ai_recommendation": patient_row["ai_recommendation"],
        "final_decision": final_decision,
        "review_comments": review_comments,
    }


def save_review_decision(review_record):
    """Append a review decision to the CSV audit log."""
    try:
        existing_log = pd.read_csv(REVIEW_LOG_PATH)
        updated_log = pd.concat(
            [existing_log, pd.DataFrame([review_record])],
            ignore_index=True,
        )
    except (FileNotFoundError, EmptyDataError):
        updated_log = pd.DataFrame([review_record])

    updated_log.to_csv(REVIEW_LOG_PATH, index=False)


def load_review_log():
    """Load submitted review decisions, returning an empty dataframe if absent."""
    try:
        return pd.read_csv(REVIEW_LOG_PATH)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()
