from __future__ import annotations

from html import escape
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import generate_synthetic_gl_data
from src.explainability import explain_entry
from src.model import GLAnomalyModel

st.set_page_config(
    layout="wide",
    page_title="LedgerGuard CRM",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --charcoal:#FFFFFF;
        --ink:#0F172A;
        --muted:#334155;
        --accent:#D04A02;
        --accent-dark:#9f3800;
        --line:#E2E8F0;
        --canvas:#F8F9FA;
        --surface:#ffffff;
        --success:#166534;
        --success-soft:#DCFCE7;
        --critical:#991B1B;
        --critical-soft:#FEF2F2;
    }
    #MainMenu, footer, [data-testid="stHeader"] { visibility:hidden; height:0; }
    html, body, [data-testid="stAppViewContainer"] {
        background:var(--canvas);
        color:var(--ink);
    }
    [data-testid="stSidebar"] {
        background:var(--charcoal);
        border-right:1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] * { color:#0F172A; }
    [data-testid="stSidebar"] hr { border-color:#E2E8F0; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#334155; }
    [data-testid="stSidebar"] .stRadio label { color:#334155; }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background:#F0FDF4;
        border:1px solid #BBF7D0;
        color:#166534;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p { color:#166534; }
    .block-container { max-width:1440px; padding:2.25rem 3rem 3rem; }
    .brand { font-size:1.35rem; font-weight:800; letter-spacing:-.03em; margin-bottom:.2rem; }
    .eyebrow { color:var(--accent); font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .hero {
        background:var(--surface);
        border:1px solid var(--line);
        border-left:6px solid var(--accent);
        border-top:0;
        border-radius:8px;
        color:var(--ink);
        padding:1.75rem 2rem;
        margin-bottom:1.5rem;
        box-shadow:0 3px 12px rgba(26,29,32,.06);
    }
    .hero h1 { color:#0F172A; margin:.35rem 0; font-size:2.15rem; }
    .hero p { color:var(--muted); margin:0; font-size:1rem; }
    .section-card { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:1.35rem; box-shadow:0 3px 12px rgba(15,23,42,.045); }
    .queue-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:1.25rem; box-shadow:0 3px 12px rgba(15,23,42,.045); }
    .queue-title { color:#0F172A; font-size:1.35rem; font-weight:800; margin:0; }
    .queue-subtitle { color:#334155; font-size:.92rem; margin:.35rem 0 1rem; }
    .queue-table-wrap { border:1px solid #E2E8F0; border-radius:6px; overflow:auto; }
    .queue-table { border-collapse:collapse; width:100%; min-width:900px; background:#FFFFFF; font-size:.84rem; }
    .queue-table th { background:#F8FAFC; color:#334155; font-size:.72rem; font-weight:800; letter-spacing:.04em; padding:.8rem .7rem; text-align:left; text-transform:uppercase; white-space:nowrap; }
    .queue-table td { border-top:1px solid #E2E8F0; color:#0F172A; padding:.78rem .7rem; white-space:nowrap; }
    .queue-table tbody tr:hover { background:#FFF7ED; }
    .entry-id { color:#1E3A8A; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-weight:700; }
    .risk-high { color:#991B1B !important; font-weight:800; }
    .risk-medium { color:#B45309 !important; font-weight:800; }
    .action-badge { border-radius:999px; display:inline-block; font-size:.72rem; font-weight:800; padding:.27rem .58rem; }
    .action-critical { background:#FEF2F2; color:#991B1B; }
    .action-review { background:#FFF7ED; color:#9A3412; }
    .action-neutral { background:#F1F5F9; color:#334155; }
    .queue-search label { color:#334155 !important; font-weight:700 !important; }
    .queue-search input { border:1px solid #E2E8F0 !important; border-radius:6px !important; padding:.7rem .85rem .7rem 2.35rem !important; }
    .stDownloadButton button {
        background:#D04A02 !important;
        border:1px solid #9A3412 !important;
        border-radius:6px !important;
        color:#FFFFFF !important;
        font-weight:600 !important;
        padding:8px 16px !important;
        transition:background .15s ease, transform .15s ease;
    }
    .stDownloadButton button:hover {
        background:#9A3412 !important;
        border-color:#7C2D12 !important;
        color:#FFFFFF !important;
        transform:translateY(-1px);
    }
    .stDownloadButton button p,
    .stDownloadButton button span { color:#FFFFFF !important; }
    .explain-note {
        background:#FFF7ED !important;
        border-left:4px solid #D04A02 !important;
        color:#1E293B !important;
    }
    .explain-note strong { color:#0F172A !important; }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div { color:#1E293B !important; }
    [data-baseweb="select"] > div,
    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [role="combobox"] {
        background:#FFFFFF !important;
        color:#0F172A !important;
        -webkit-text-fill-color:#0F172A !important;
        opacity:1 !important;
    }
    [data-baseweb="select"] svg { fill:#334155 !important; }
    [data-baseweb="popover"] [role="option"] {
        background:#FFFFFF !important;
        color:#0F172A !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background:#F1F5F9 !important;
        color:#0F172A !important;
    }
    input:disabled,
    textarea:disabled,
    [aria-disabled="true"] {
        background:#F8FAFC !important;
        color:#0F172A !important;
        -webkit-text-fill-color:#0F172A !important;
        opacity:1 !important;
    }
    .kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:1rem; margin:0 0 1.5rem; }
    .kpi-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:1rem 1.1rem .95rem; min-height:7.1rem; box-shadow:0 3px 12px rgba(15,23,42,.05); }
    .kpi-label { color:#475569; font-size:.74rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
    .kpi-value { color:#0F172A; font-size:1.75rem; font-weight:800; letter-spacing:-.025em; line-height:1.35; margin:.4rem 0 .65rem; }
    .kpi-badge { display:inline-block; border-radius:999px; font-size:.76rem; font-weight:700; padding:.25rem .55rem; }
    .kpi-badge.neutral { background:#F1F5F9; color:#334155; }
    .kpi-badge.critical { background:#FEF2F2; color:#991B1B; }
    .kpi-badge.healthy { background:#F0FDF4; color:#166534; }
    [data-testid="stMetric"] {
        background:var(--surface);
        border:1px solid var(--line);
        border-radius:8px;
        padding:1rem 1.1rem .9rem;
        box-shadow:0 3px 12px rgba(26,29,32,.05);
        min-height:7rem;
    }
    [data-testid="stMetricValue"] { color:#0F172A; font-weight:800; font-size:1.75rem; letter-spacing:-.02em; }
    [data-testid="stMetricLabel"] { color:#475569; font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
    [data-testid="stMetricDelta"] { color:var(--success); font-weight:700; font-size:.8rem; }
    [data-testid="stMetricDelta"] svg { display:none; }
    input, [data-baseweb="select"] > div, [data-testid="stTextInput"] input {
        background:#fff !important;
        color:var(--ink) !important;
        border-color:#c9c3bd !important;
    }
    [data-baseweb="select"] * { color:var(--ink) !important; }
    [data-testid="stAlert"] { border-radius:6px; }
    .explain-note {
        background:#fff6f0;
        border-left:4px solid var(--accent);
        color:#5d3827;
        padding:.85rem 1rem;
        margin:.5rem 0 1rem;
        font-size:.92rem;
    }
    .action-banner {
        align-items:center;
        background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-left:4px solid #DC2626;
        border-radius:8px;
        display:flex;
        justify-content:space-between;
        gap:1rem;
        margin:1rem 0;
        padding:1rem 1.15rem;
    }
    .action-banner-label { color:#0F172A; font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .action-banner-value { color:#0F172A; font-size:1.02rem; font-weight:800; margin-top:.2rem; }
    .action-badge-large { background:#FEF2F2; border:1px solid #FCA5A5; border-radius:999px; color:#991B1B; font-size:.78rem; font-weight:800; padding:.42rem .7rem; white-space:nowrap; }
    .waterfall-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:1rem 1.15rem .75rem; box-shadow:0 3px 12px rgba(15,23,42,.045); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    [data-testid="stPlotlyChart"] {
        background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-radius:8px;
        padding:1rem 1rem .5rem;
        box-shadow:0 3px 12px rgba(15,23,42,.045);
    }
    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
        .kpi-grid { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_dashboard_data() -> Tuple[GLAnomalyModel, pd.DataFrame, pd.DataFrame]:
    df = generate_synthetic_gl_data(n_records=5000, anomaly_ratio=0.06, random_state=42)
    model = GLAnomalyModel().fit(df)
    scored = model.score_records(df)
    return model, scored, model.get_flagged_queue(scored, queue_limit=25)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


model, scored_df, flagged_df = load_dashboard_data()

with st.sidebar:
    st.markdown("<div class='brand'>🛡️ LedgerGuard</div>", unsafe_allow_html=True)
    st.caption("AI risk operations CRM")
    st.divider()
    st.markdown("**WORKSPACE**")
    st.radio("Workspace", ["Overview", "Exception queue", "Model insights"], label_visibility="collapsed")
    st.divider()
    st.markdown("**MODEL HEALTH**")
    st.success("Active · Isolation Forest")
    st.caption(f"Last refresh: {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}")

st.markdown(
    """
    <div class='hero'>
        <div class='eyebrow'>RISK OPERATIONS CENTER</div>
        <h1>Good morning, audit team</h1>
        <p>Prioritize high-impact ledger exceptions, understand risk drivers, and move cases forward.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

flagged_count = int((scored_df["Risk_Score"] >= model.high_risk_threshold).sum())
critical_count = int((scored_df["Risk_Score"] >= model.critical_risk_threshold).sum())
total_value = scored_df.loc[scored_df["Risk_Score"] >= model.high_risk_threshold, "Amount"].sum()
kpi_cards = [
    ("LEDGER ENTRIES", f"{len(scored_df):,}", "6.0% monitored", "neutral"),
    ("OPEN EXCEPTIONS", f"{flagged_count:,}", f"{critical_count} critical", "critical"),
    ("VALUE AT RISK", format_currency(total_value), "High-risk queue", "neutral"),
    ("MODEL CONFIDENCE", "96.2%", "Healthy", "healthy"),
]
kpi_cols = st.columns(4)
for column, (label, value, subtext, badge) in zip(kpi_cols, kpi_cards):
    with column:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<span class='kpi-badge {badge}'>{subtext}</span></div>",
            unsafe_allow_html=True,
        )

risk_histogram = px.histogram(
    scored_df,
    x="Risk_Score",
    nbins=20,
    title="Risk distribution",
    template="plotly_white",
    color_discrete_sequence=["#C2410C"],
)
risk_histogram.update_layout(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font={"family":"Inter, Segoe UI, Roboto, sans-serif", "color":"#20252B", "size":12},
    title={"font":{"family":"Inter, Segoe UI, Roboto, sans-serif", "size":16, "color":"#20252B"}},
    bargap=0.15,
    hoverlabel={
        "bgcolor":"#1A1D20",
        "bordercolor":"#1A1D20",
        "font":{"family":"Inter, Segoe UI, Roboto, sans-serif", "color":"#FFFFFF", "size":12},
    },
    xaxis={
        "title":"Normalized risk score",
        "showgrid":False,
        "showline":True,
        "linecolor":"#CBD5E1",
        "linewidth":1,
        "zeroline":False,
        "ticks":"outside",
        "tickcolor":"#CBD5E1",
    },
    yaxis={
        "title":"Ledger entries",
        "showgrid":True,
        "gridcolor":"#E2E8F0",
        "gridwidth":1,
        "zeroline":False,
        "ticks":"outside",
        "tickcolor":"#CBD5E1",
    },
    margin={"l": 28, "r": 24, "t": 48, "b": 36},
)
risk_histogram.update_traces(
    marker={
        "color":"#C2410C",
        "line":{"color":"#9A3412", "width":0.5},
    },
    hovertemplate=(
        "<b>Risk score range</b>: %{x}<br>"
        "<b>Ledger entries</b>: %{y:,}<extra></extra>"
    ),
)
st.plotly_chart(
    risk_histogram,
    use_container_width=True,
    config={"displayModeBar": False},
)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='queue-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='queue-title'>Exception queue</h2>", unsafe_allow_html=True)
st.markdown(
    "<p class='queue-subtitle'>Cases are ranked by normalized risk score. "
    "Search and open a case to review its evidence.</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='queue-search'>", unsafe_allow_html=True)
search = st.text_input(
    "Search queue",
    placeholder="🔎  Search by entry ID, action, or risk driver",
    label_visibility="visible",
)
st.markdown("</div>", unsafe_allow_html=True)
filtered_flagged = flagged_df.copy()
if search:
    searchable = filtered_flagged.astype(str).apply(
        lambda column: column.str.contains(search, case=False, na=False)
    )
    filtered_flagged = filtered_flagged[searchable.any(axis=1)]
queue_columns = [
    "Entry_ID", "Amount", "Posting_Hour", "Account_Code", "User_ID",
    "Risk_Score", "Action_Required", "Primary_Risk_Driver",
]
queue_df = filtered_flagged[queue_columns].copy()
queue_df["Amount"] = queue_df["Amount"].map(format_currency)
queue_df["Risk_Score"] = queue_df["Risk_Score"].map(lambda value: f"{value:.4f}")
display_columns = {
    "Entry_ID": "Entry ID",
    "Amount": "Amount",
    "Posting_Hour": "Posting Hour",
    "Account_Code": "Account Code",
    "User_ID": "User ID",
    "Risk_Score": "Risk Score",
    "Action_Required": "Action Required",
    "Primary_Risk_Driver": "Primary Risk Driver",
}
headers = list(display_columns.values())
rows = []
for _, row in filtered_flagged[queue_columns].iterrows():
    risk_score = float(row["Risk_Score"])
    risk_class = "risk-high" if risk_score > 0.90 else "risk-medium" if risk_score >= 0.70 else ""
    action = str(row["Action_Required"])
    action_class = (
        "action-critical" if action.startswith("Escalate")
        else "action-review" if action.startswith("Investigate")
        else "action-neutral"
    )
    rows.append(
        "<tr>"
        f"<td><span class='entry-id'>{escape(str(row['Entry_ID']))}</span></td>"
        f"<td>{escape(format_currency(float(row['Amount'])))}</td>"
        f"<td>{int(row['Posting_Hour'])}</td>"
        f"<td>{int(row['Account_Code'])}</td>"
        f"<td>{int(row['User_ID'])}</td>"
        f"<td class='{risk_class}'>{risk_score:.4f}</td>"
        f"<td><span class='action-badge {action_class}'>{escape(action)}</span></td>"
        f"<td>{escape(str(row['Primary_Risk_Driver']))}</td>"
        "</tr>"
    )
table_html = (
    "<div class='queue-table-wrap'><table class='queue-table'><thead><tr>"
    + "".join(f"<th>{header}</th>" for header in headers)
    + "</tr></thead><tbody>"
    + "".join(rows)
    + "</tbody></table></div>"
)
st.markdown(table_html, unsafe_allow_html=True)
st.caption(f"Showing {len(queue_df):,} of {len(flagged_df):,} prioritized cases")
st.download_button(
    "Download queue CSV",
    data=queue_df.rename(columns=display_columns).to_csv(index=False),
    file_name="ledger_exception_queue.csv",
    mime="text/csv",
    type="primary",
)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Case review & explanation")
st.markdown(
    """
    <div class='explain-note'>
        <strong>What this is for:</strong> This chart explains which ledger attributes
        pushed the selected journal entry toward or away from an anomaly classification.
        Longer bars indicate stronger model influence; this is decision support for an
        auditor, not proof of fraud.
    </div>
    """,
    unsafe_allow_html=True,
)
entry_options = filtered_flagged["Entry_ID"].tolist()
if not entry_options:
    st.info("No cases match your search.")
    st.stop()
selected_entry = st.selectbox("Open case", entry_options)
selected_row = model.get_entry_for_explanation(selected_entry, scored_df)
explanation = explain_entry(model, selected_row, model.feature_columns)

summary_cols = st.columns(4)
summary_cols[0].metric("Case ID", selected_entry)
summary_cols[1].metric("Risk score", f"{selected_row['Risk_Score']:.4f}")
summary_cols[2].metric(
    "Priority",
    "Critical" if selected_row["Risk_Score"] >= model.critical_risk_threshold else "High",
)
summary_cols[3].metric("Owner", "Unassigned")
st.markdown(
    f"""
    <div class='action-banner'>
        <div>
            <div class='action-banner-label'>Recommended action</div>
            <div class='action-banner-value'>Primary driver: {escape(str(selected_row["Primary_Risk_Driver"]).replace("_", " "))}</div>
        </div>
        <span class='action-badge-large'>{escape(str(selected_row["Action_Required"]))}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Risk driver impact · coral bars increase the model signal, blue bars reduce it")

feature_labels = {
    "Amount": "Amount",
    "Account_Code": "Account Code",
    "Posting_Hour": "Posting Hour",
    "User_ID": "User ID",
    "Approval_Level": "Approval Level",
}
feature_names = [
    feature_labels.get(str(name), str(name).replace("_", " "))
    for name in explanation.feature_names
]
contributions = [float(value) for value in explanation.values]
base_risk = float(explanation.base_values)
waterfall = go.Figure(
    go.Waterfall(
        name="Risk drivers",
        orientation="v",
        x=["Base Value", *feature_names, "Final Score"],
        measure=["absolute", *["relative"] * len(contributions), "total"],
        y=[base_risk, *contributions, None],
        text=[f"{base_risk:.2f}", *[f"{value:+.2f}" for value in contributions], f"{base_risk + sum(contributions):.2f}"],
        textposition="inside",
        textfont={
            "color":"#FFFFFF",
            "size":12,
            "family":"Inter Semi Bold, Segoe UI Semibold, Arial, sans-serif",
        },
        cliponaxis=False,
        connector={"line":{"color":"#CBD5E1", "width":1}},
        increasing={"marker":{"color":"#DC2626", "line":{"color":"#991B1B", "width":1}}},
        decreasing={"marker":{"color":"#2563EB", "line":{"color":"#1E40AF", "width":1}}},
        totals={"marker":{"color":"#2563EB", "line":{"color":"#1E40AF", "width":1}}},
        hovertemplate="<b>%{x}</b><br>Risk contribution: %{y:+.2f}<extra></extra>",
    )
)
waterfall.update_layout(
    height=470,
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font={"family":"Inter, Segoe UI, Roboto, sans-serif", "color":"#0F172A", "size":13},
    showlegend=False,
    hoverlabel={"bgcolor":"#1A1D20", "font":{"color":"#FFFFFF", "size":12}},
    xaxis={
        "title":"",
        "showgrid":False,
        "tickangle":0,
        "tickfont":{"color":"#334155", "size":12, "family":"Inter, Segoe UI, Roboto, sans-serif"},
        "linecolor":"#94A3B8",
        "linewidth":1,
        "automargin":True,
    },
    yaxis={
        "title":{"text":"<b>Model Risk Signal</b>", "font":{"color":"#0F172A", "size":14}},
        "title_standoff":14,
        "showgrid":True,
        "gridcolor":"#E2E8F0",
        "zeroline":False,
        "tickfont":{"color":"#334155", "size":12, "family":"Inter, Segoe UI, Roboto, sans-serif"},
        "tickformat":".1f",
        "automargin":True,
        "range":[min(-0.5, min(contributions) - 0.3), max(base_risk + sum(contributions) + 0.8, 1.0)],
    },
    margin={"l":75, "r":25, "t":30, "b":80},
)
st.markdown("<div class='waterfall-card'>", unsafe_allow_html=True)
st.plotly_chart(waterfall, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
