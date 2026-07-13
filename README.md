# ENCHANTED Model 1 Dashboard

Streamlit dashboard for ENCHANTED Model 1 right-siting decision support. The app combines rule-based screening, AI risk stratification, service suitability, nursing/operational review, role-based task queues, and optional AWS Bedrock explanation generation.

This is a prototype using sample data. Final referral, transfer, and right-siting decisions remain with the clinical care team.

## What Was Improved

- Refactored the original single-file Streamlit app into a clearer frontend/backend structure.
- Added prototype hospital ID login with designation-based access.
- Prepared the authentication flow for future OAuth/OIDC integration.
- Added a top-right logged-in user display showing name, hospital ID, and designation.
- Removed the manual left-side role selector.
- Added role-based task queues:
  - Tasks assigned to you
  - Ongoing
  - Past
- Converted the dashboard into tabs:
  - Overview & Tasks
  - Patient List
  - Patient Review
  - Audit & LLM
- Improved patient list controls:
  - Category switching
  - Search
  - Sorting
  - Column selection
  - Advanced filters
- Improved UI contrast and formatting for buttons, labels, dropdowns, radio options, and multiselect chips.
- Added safer handling for missing AWS Bedrock credentials.
- Added code comments/docstrings to make each file and function easier to understand.
- Added a `.gitignore` so real secrets, `.env` files, and Python cache files are not committed.

## Project Structure

```text
classification.py
  Streamlit entry point. Calls app.frontend.render_dashboard().

app/auth.py
  Login and user profile handling.
  Supports prototype hospital ID login and is prepared for OAuth/OIDC.

app/backend.py
  Data loading, clinical rules, model scoring, recommendations, filters,
  audit logging, LLM prompt generation, and AWS Bedrock calls.

app/frontend.py
  Streamlit UI layout, tabs, metrics, tables, forms, and styling.

.streamlit/secrets.example.toml
  Example secrets template. Do not put real secrets in this file.
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run classification.py
```

Open:

```text
http://localhost:8501
```

## Demo Login IDs

Use one of these prototype hospital IDs:

```text
HOSP-CM-001   Case Manager
HOSP-CM-002   Case Manager
HOSP-REF-001  JCH Referral Team
HOSP-CLN-001  Clinician
```

These demo users are defined in `app/auth.py`.

## Risk Score Logic

The dashboard loads a trained Random Forest model from:

```text
models/enchanted_model1_random_forest.joblib
```

Risk scores are generated only for Amber and Green patients:

```python
model.predict_proba(data.loc[eligible_mask, FEATURES])[:, 1]
```

The probability is then converted into a risk band:

```text
>= 0.30  High Risk
>= 0.15  Medium Risk
<  0.15  Low Risk
```

Red / No-Go patients are not scored by the ML model because they are already excluded by deterministic clinical rules.

## AWS Bedrock Setup

The app can generate patient-level explanation/report text using AWS Bedrock.

Current model ID:

```text
anthropic.claude-3-haiku-20240307-v1:0
```

The Bedrock call is implemented in:

```text
app/backend.py
```

Function:

```python
call_bedrock_llm(prompt)
```

### Required Streamlit Secrets

Do not hardcode or commit real AWS credentials.

For local development, create:

```text
.streamlit/secrets.toml
```

Add these secret names with your real values:

```text
AWS_ACCESS_KEY_ID      your AWS access key ID
AWS_SECRET_ACCESS_KEY  your AWS secret access key
AWS_DEFAULT_REGION     your AWS region, for example us-east-1
```

For Streamlit Community Cloud, add the same values in the app's **Secrets** settings.

The real `.streamlit/secrets.toml` file is ignored by git through `.gitignore`.

### Required AWS Setup

In AWS:

1. Open Amazon Bedrock.
2. Choose the same region as `AWS_DEFAULT_REGION`.
3. Enable model access for Claude 3 Haiku or the model you plan to use.
4. Ensure the IAM user/role has permission to invoke Bedrock models.

Example IAM actions:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "*"
}
```

If Bedrock is not configured, the app will not crash. It will show a message that LLM explanation is not configured yet.

## OAuth / Hospital Login Setup

Prototype login is enabled by default.

To prepare OAuth/OIDC, copy:

```text
.streamlit/secrets.example.toml
```

to:

```text
.streamlit/secrets.toml
```

Then configure real provider values and set:

```toml
OAUTH_ENABLED = true
```

Do not commit `.streamlit/secrets.toml`.

## Files Not To Commit

Do not commit:

```text
.streamlit/secrets.toml
.env
.env.*
__pycache__/
```

These are already covered by `.gitignore`.

## Deployment Notes

For Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Create a new Streamlit app from `classification.py`.
3. Add AWS and OAuth secrets in Streamlit Cloud if needed.
4. Deploy.

For production use, replace CSV files with a proper database and connect authentication to the hospital identity provider.
