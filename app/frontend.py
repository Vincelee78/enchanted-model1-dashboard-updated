"""Top-level Streamlit orchestrator for role-specific ENCHANTED views."""

from __future__ import annotations

import streamlit as st

from app.auth import render_auth_gate
from app.backend import get_role_scoped_worklist, load_patient_worklist
from app.components.header import configure_page, render_header
from app.views.case_manager import render_case_manager_dashboard
from app.views.clinician import render_clinician_dashboard
from app.views.referral_team import render_referral_dashboard


SUPPORTED_ROLES = {
    "Case Manager",
    "Clinician",
    "JCH Referral Team",
}


def render_dashboard() -> None:
    """Authenticate the user and route to the correct role-specific workspace."""
    configure_page()
    user_profile = render_auth_gate()

    try:
        worklist = load_patient_worklist()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.exception(error)
        st.stop()

    role = user_profile.get("designation")
    if role not in SUPPORTED_ROLES:
        st.error(f"Unsupported dashboard role: {role!r}")
        st.stop()

    render_header(user_profile)
    role_scoped = get_role_scoped_worklist(worklist, user_profile)

    if role == "Case Manager":
        render_case_manager_dashboard(role_scoped, user_profile)
    elif role == "Clinician":
        render_clinician_dashboard(role_scoped, user_profile)
    elif role == "JCH Referral Team":
        render_referral_dashboard(role_scoped, user_profile)


if __name__ == "__main__":
    render_dashboard()