"""Authentication and user-profile helpers for the dashboard.

This file currently supports two modes:
1. Prototype mode: validate a small set of demo hospital IDs.
2. OAuth-ready mode: use Streamlit's login/user APIs when OAuth secrets are set.

The rest of the dashboard receives a single user_profile dictionary from here
and uses that profile to decide which cases the user can see.
"""

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


# Demo registry used until the app is connected to a real hospital identity
# provider. In production, this should come from OAuth claims, LDAP/AD, or a DB.
STAFF_REGISTRY = {
    "HOSP-CM-001": {
        "name": "Sarah Roberts",
        "designation": "Case Manager",
        "assigned_case_manager": "Case Manager A",
    },
    "HOSP-CM-002": {
        "name": "Daniel Lee",
        "designation": "Case Manager",
        "assigned_case_manager": "Case Manager B",
    },
    "HOSP-REF-001": {
        "name": "JCH Referral Team",
        "designation": "JCH Referral Team",
        "assigned_case_manager": None,
    },
    "HOSP-CLN-001": {
        "name": "Dr Anita Rao",
        "designation": "Clinician",
        "assigned_case_manager": None,
    },
}


def _oauth_enabled():
    """Return whether OAuth mode is enabled through Streamlit secrets."""
    return bool(_get_secret("OAUTH_ENABLED", False))


def _get_secret(key, default=None):
    """Read a Streamlit secret without crashing when no secrets file exists."""
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


def _profile_from_oauth_user():
    """Convert Streamlit's OAuth user object into the app's user_profile shape."""
    hospital_id = st.user.get("hospital_id") or st.user.get("preferred_username")
    email = st.user.get("email")
    lookup_key = hospital_id or email

    if lookup_key in STAFF_REGISTRY:
        return {"hospital_id": lookup_key, **STAFF_REGISTRY[lookup_key]}

    if not bool(_get_secret("OAUTH_ALLOW_UNKNOWN_USERS", False)):
        st.error("Your hospital ID is authenticated, but it is not authorised for this dashboard.")
        if st.button("Logout"):
            st.logout()
        st.stop()

    designation = st.user.get("designation") or st.user.get("role") or "Clinician"
    return {
        "hospital_id": lookup_key or "Authenticated user",
        "name": st.user.get("name") or email or "Authenticated user",
        "designation": designation,
        "assigned_case_manager": st.user.get("assigned_case_manager"),
    }


def render_auth_gate():
    """Render login UI and return the authenticated user's profile.

    This function intentionally stops the Streamlit run with st.stop() until a
    valid user is available. Downstream dashboard code can therefore assume it
    has a user_profile.
    """
    if _oauth_enabled():
        if not st.user.is_logged_in:
            st.title("ENCHANTED Model 1")
            st.caption("Sign in with your hospital account to continue.")
            if st.button("Sign in with hospital OAuth"):
                st.login()
            st.stop()

        return _profile_from_oauth_user()

    st.title("ENCHANTED Model 1")
    st.caption("Prototype login. Use hospital ID validation until OAuth secrets are configured.")

    with st.form("hospital_id_login"):
        hospital_id = st.text_input(
            "Hospital ID",
            placeholder="Example: HOSP-CM-001, HOSP-REF-001, HOSP-CLN-001",
        )
        submitted = st.form_submit_button("Continue")

    if submitted:
        normalised_id = hospital_id.strip().upper()
        if normalised_id in STAFF_REGISTRY:
            st.session_state["user_profile"] = {
                "hospital_id": normalised_id,
                **STAFF_REGISTRY[normalised_id],
            }
            st.rerun()
        else:
            st.error("Hospital ID not recognised for this prototype.")

    if "user_profile" not in st.session_state:
        st.info(
            "Demo IDs: HOSP-CM-001, HOSP-CM-002, HOSP-REF-001, HOSP-CLN-001."
        )
        st.stop()

    return st.session_state["user_profile"]


def logout_user():
    """Clear local login state and log out from OAuth when OAuth is active."""
    st.session_state.pop("user_profile", None)
    if _oauth_enabled() and st.user.is_logged_in:
        st.logout()
    st.rerun()
