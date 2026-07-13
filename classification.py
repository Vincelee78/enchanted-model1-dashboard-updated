"""Streamlit entry point for the ENCHANTED Model 1 dashboard.

The implementation lives in the app package so the project is easier to read:
- app.auth handles login and user identity.
- app.backend handles data, rules, model scoring, filtering, logging, and LLM calls.
- app.frontend handles the Streamlit screens and widgets.
"""

from app.frontend import render_dashboard


if __name__ == "__main__":
    render_dashboard()
