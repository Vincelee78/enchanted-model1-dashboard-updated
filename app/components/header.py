"""Shared page styling and role-specific dashboard headers."""

from __future__ import annotations

import html

import streamlit as st

from app.auth import logout_user


ROLE_UI_CONFIG = {
    "Case Manager": {
        "title": "ENCHANTED Model 1: Case Manager Screening Dashboard",
        "subtitle": "AI-Assisted Screening, Monitoring and CH Candidate Review",
        "description": (
            "Review the AI-screened inpatient pool, monitor changing eligibility, "
            "confirm potential CH candidates and escalate cases requiring clinical judgement."
        ),
    },
    "Clinician": {
        "title": "ENCHANTED Model 1: Clinical Review Dashboard",
        "subtitle": "Clinical Review, Override and Right-Siting Decision Support",
        "description": (
            "Review cases escalated for clinical judgement, examine the screening reasons "
            "and record an advisory clinical decision."
        ),
    },
    "JCH Referral Team": {
        "title": "ENCHANTED Model 1: Referral Review Dashboard",
        "subtitle": "Transfer Eligibility, Operational Readiness and Referral Tracking",
        "description": (
            "Review validated or potentially suitable CH cases, assess operational readiness "
            "and manage downstream referral actions."
        ),
    },
}


def configure_page() -> None:
    """Set Streamlit metadata and shared CSS once per app render."""
    st.set_page_config(page_title="ENCHANTED Model 1", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f7fbff 0%, #eef6fb 45%, #f8fafc 100%);
            color: #1f2937;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }
        h1 { color:#0b3a66; font-weight:800; letter-spacing:-0.5px; }
        h2, h3 { color:#1f4e79; font-weight:700; }
        div[data-testid="stMetric"] {
            background:rgba(255,255,255,.9);
            border:1px solid #dbeafe;
            padding:16px;
            border-radius:15px;
            box-shadow:0 4px 12px rgba(15,23,42,.06);
        }
        div[data-testid="stDataFrame"] {
            background:white;
            border-radius:14px;
            box-shadow:0 4px 14px rgba(15,23,42,.06);
            padding:8px;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background:#0b3a66 !important;
            color:#fff !important;
            border:1px solid #0b3a66 !important;
            border-radius:10px;
            font-weight:800 !important;
        }
        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background:#145ea8 !important;
            border-color:#145ea8 !important;
        }
        div[data-baseweb="select"] > div,
        input, textarea {
            background:#fff !important;
            color:#111827 !important;
            border-radius:10px !important;
        }
        .top-bar {
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:24px;
        }
        .user-chip {
            min-width:225px;
            background:#fff;
            border:1px solid #dbeafe;
            border-radius:12px;
            padding:12px 14px;
            box-shadow:0 4px 12px rgba(15,23,42,.06);
            text-align:right;
        }
        .user-chip-name { color:#1f2937; font-size:14px; font-weight:700; }
        .user-chip-role { color:#1d4ed8; font-size:13px; font-weight:700; margin-top:2px; }
        .user-chip-id { color:#64748b; font-size:12px; margin-top:2px; }
        .role-card {
            background:rgba(255,255,255,.92);
            border:1px solid #dbeafe;
            border-left:6px solid #2563eb;
            padding:16px 20px;
            border-radius:15px;
            box-shadow:0 4px 14px rgba(15,23,42,.06);
            margin:12px 0 16px 0;
        }
        .screening-bucket-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #dbeafe;
            border-radius: 20px;
            padding: 27px 24px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }
        .screening-bucket-label {
            color: #374151;
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 16px;
        }
        .screening-bucket-value {
            color: #30313d;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.15;
            white-space: normal;
        }
        .bucket-row { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
        .bucket-pill { padding:7px 12px; border-radius:999px; font-weight:700; font-size:13px; }
        .pill-red { background:#fee2e2; color:#7f1d1d; }
        .pill-amber { background:#fef3c7; color:#78350f; }
        .pill-green { background:#dcfce7; color:#14532d; }
        .pill-indigo { background:#e0e7ff; color:#3730a3; }
        .pill-blue { background:#dbeafe; color:#1e3a8a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(user_profile: dict) -> None:
    """Render the role-aware title, identity chip and dashboard guardrail."""
    role = user_profile.get("designation", "Case Manager")
    config = ROLE_UI_CONFIG.get(role, ROLE_UI_CONFIG["Case Manager"])

    name = html.escape(str(user_profile.get("name", "Authenticated user")))
    hospital_id = html.escape(str(user_profile.get("hospital_id", "")))
    safe_role = html.escape(str(role))

    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <h1>{html.escape(config['title'])}</h1>
                <div style="font-size:21px;color:#1f4e79;font-weight:700;">
                    {html.escape(config['subtitle'])}
                </div>
            </div>
            <div class="user-chip">
                <div class="user-chip-name">{name}</div>
                <div class="user-chip-role">{safe_role}</div>
                <div class="user-chip-id">{hospital_id}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Prototype using sample data and partial criteria coverage. Final referral and "
        "transfer decisions remain with the case manager, clinician and care team."
    )

    _, logout_col = st.columns([0.88, 0.12])
    if logout_col.button("Logout", key="global_logout"):
        logout_user()

    st.markdown(
        f"""
        <div class="role-card">
            <div style="font-size:17px;font-weight:800;color:#0b3a66;">{safe_role} workspace</div>
            <div style="font-size:14px;color:#475569;margin-top:5px;">
                {html.escape(config['description'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_bucket_legend()


def render_bucket_legend() -> None:
    st.markdown(
        """
        <div class="bucket-row">
            <span class="bucket-pill pill-green">Potential CH Candidate</span>
            <span class="bucket-pill pill-amber">Pending Monitoring</span>
            <span class="bucket-pill pill-indigo">Needs Clinical Review</span>
            <span class="bucket-pill pill-red">Not Suitable at Current Review</span>
            <span class="bucket-pill pill-blue">Prototype ML Risk Support</span>
        </div>
        """,
        unsafe_allow_html=True,
    )