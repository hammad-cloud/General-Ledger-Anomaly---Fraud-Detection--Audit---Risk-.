# LedgerGuard CRM

AI-powered general ledger anomaly and exception detection CRM built with Python and Streamlit.

## Overview

This project implements a defensible, explainable anomaly detection workflow for journal entries, using an Isolation Forest model to detect suspicious transactions and SHAP to explain why they were flagged.

## Repository structure

```text
gl-anomaly-detector/
├── config/
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── model.py
│   └── explainability.py
├── app.py
├── requirements.txt
├── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- Generates synthetic ledger entries with realistic user, account, and approval patterns.
- Injects controlled outliers for high-value, off-hours, and unsupported approval scenarios.
- Trains an Isolation Forest model to flag suspicious entries.
- Normalizes model decision scores into a 0-1 risk score.
- Presents a professional risk operations workspace with KPI cards, model health, search, and an exception queue.
- Provides a case review workspace for individual flagged records using SHAP waterfall plots.
- Falls back to a Plotly contributor chart with an actionable dependency message if the SHAP renderer is unavailable.
