"""
HR Analytics — Attrition Prediction & Resume Screening
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hr_analytics import (
    SAMPLE_JD,
    SAMPLE_RESUME_GOOD,
    SAMPLE_RESUME_POOR,
    compute_hr_kpis,
    load_hr_data,
    load_sbert_model,
    predict_attrition_risk,
    score_resume,
    train_attrition_model,
)

# -- Alpino chart palette --
_ALPINO_GOLD = "#FFC107"
_ALPINO_ORANGE = "#FF9800"
_ALPINO_DEEP = "#FF5722"
_ALPINO_GREEN = "#4CAF50"
_ALPINO_SEQ = [_ALPINO_GREEN, _ALPINO_GOLD, _ALPINO_DEEP]

st.set_page_config(page_title="HR Analytics | Alpino AI", page_icon="👥", layout="wide")

st.title("HR Analytics")
st.info(
    "**XGBoost attrition prediction** with department-level risk scoring  |  "
    "**SBERT resume screening** for smarter hiring beyond the 95% portal dependency"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
try:
    uploaded_csv = st.sidebar.file_uploader("Upload custom HR data (CSV)", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success(f"Loaded {len(df)} records.")
    else:
        df = load_hr_data()
except Exception as e:
    st.error(f"Failed to load HR data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
kpis = compute_hr_kpis(df)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Headcount", kpis["total_employees"])
k2.metric("Early Attrition", f"{kpis['early_attrition_rate']}%",
          help="Tenure < 6 months who left")
k3.metric("Avg Tenure", f"{kpis['avg_tenure_months']} mo")
k4.metric("Overall Attrition", f"{kpis['overall_attrition_rate']}%")
k5.metric("Portal Hiring", f"{kpis['portal_hiring_pct']}%")
k6.metric("Promotion Rate", f"{kpis['promotion_rate']}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Attrition by Department
# ---------------------------------------------------------------------------
st.subheader("Attrition Rate by Department")

dept_df = kpis["dept_attrition"].reset_index()
dept_df.columns = ["Department", "Attrition Rate (%)", "Headcount"]

fig_dept = px.bar(
    dept_df,
    x="Attrition Rate (%)",
    y="Department",
    orientation="h",
    text="Attrition Rate (%)",
    color="Attrition Rate (%)",
    color_continuous_scale=_ALPINO_SEQ,
    range_color=[0, 70],
)
fig_dept.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_dept.update_layout(
    height=380,
    yaxis={"categoryorder": "total ascending"},
    showlegend=False,
    coloraxis_showscale=False,
    margin={"l": 10, "r": 50, "t": 10, "b": 10},
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_dept.update_xaxes(showgrid=False)
fig_dept.update_yaxes(showgrid=False)
st.plotly_chart(fig_dept, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# XGBoost Feature Importance
# ---------------------------------------------------------------------------
st.subheader("Why Are They Leaving?")

try:
    with st.spinner("Fitting gradient-boosted ensemble with class-weighted stratification..."):
        model_result = train_attrition_model()

    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Accuracy", f"{model_result['accuracy'] * 100:.1f}%")
    pm2.metric("ROC AUC", f"{model_result['roc_auc']:.3f}")
    pm3.metric("Features", len(model_result["feature_names"]))

    feat_imp = model_result["feature_importance"]
    top_n = min(10, len(feat_imp))
    top_features = dict(list(feat_imp.items())[:top_n])

    feat_df = pd.DataFrame({
        "Feature": list(top_features.keys()),
        "Importance": list(top_features.values()),
    })

    fig_feat = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale=[_ALPINO_GOLD, _ALPINO_DEEP],
    )
    fig_feat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_feat.update_layout(
        height=380,
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        coloraxis_showscale=False,
        margin={"l": 10, "r": 50, "t": 10, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_feat.update_xaxes(showgrid=False)
    fig_feat.update_yaxes(showgrid=False)
    st.plotly_chart(fig_feat, width="stretch")

    # --- Risk Scores ---
    st.markdown("---")
    st.subheader("Employee Risk Scores")
    st.caption(
        "High (>0.7): Immediate 1-on-1  |  "
        "Medium (0.4-0.7): Training cohort  |  Low (<0.4): Standard"
    )

    try:
        scored_df = predict_attrition_risk(df, model_result)

        rd1, rd2, rd3 = st.columns(3)
        risk_counts = scored_df["risk_category"].value_counts()
        rd1.metric("High Risk", risk_counts.get("High", 0))
        rd2.metric("Medium Risk", risk_counts.get("Medium", 0))
        rd3.metric("Low Risk", risk_counts.get("Low", 0))

        dept_risk = (
            scored_df.groupby("Department")["risk_score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        dept_risk.columns = ["Department", "Avg Risk Score"]

        fig_risk = px.bar(
            dept_risk,
            x="Avg Risk Score",
            y="Department",
            orientation="h",
            text="Avg Risk Score",
            color="Avg Risk Score",
            color_continuous_scale=_ALPINO_SEQ,
            range_color=[0, 1],
        )
        fig_risk.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_risk.update_layout(
            height=340,
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            coloraxis_showscale=False,
            margin={"l": 10, "r": 50, "t": 10, "b": 10},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_risk.update_xaxes(showgrid=False)
        fig_risk.update_yaxes(showgrid=False)
        st.plotly_chart(fig_risk, width="stretch")

        with st.expander("Full Employee Risk Table", expanded=False):
            display_cols = [
                "Employee_ID", "Department", "Tenure_Months",
                "Performance_Rating", "Hiring_Source", "Training_Hours",
                "Promotion_Received", "risk_score", "risk_category",
            ]
            st.dataframe(
                scored_df[display_cols].sort_values("risk_score", ascending=False),
                width="stretch",
                hide_index=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score", min_value=0, max_value=1, format="%.2f",
                    ),
                },
            )
    except Exception as e:
        st.error(f"Risk scoring error: {e}")

except Exception as e:
    st.error(f"Model training encountered an issue: {e}")

# ---------------------------------------------------------------------------
# Resume Screening — Mode Selection
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("AI Resume Screener")
st.caption(
    "Sentence-BERT (all-MiniLM-L6-v2)  |  Shortlist >= 65%  |  "
    "Review 45-65%  |  Reject < 45%"
)

resume_mode = st.radio(
    "Screening Mode:",
    options=["Single Candidate", "Batch Processing"],
    horizontal=True,
    index=0,
)

# ===========================================================================
# SINGLE CANDIDATE SCREENING
# ===========================================================================
if resume_mode == "Single Candidate":
    prefill = st.radio(
        "Demo quick-fill:",
        options=["Empty", "Strong Match (Marketing)", "Poor Match (Software Dev)"],
        horizontal=True,
        index=0,
    )

    if prefill == "Strong Match (Marketing)":
        default_resume = SAMPLE_RESUME_GOOD
    elif prefill == "Poor Match (Software Dev)":
        default_resume = SAMPLE_RESUME_POOR
    else:
        default_resume = ""

    res_col, jd_col = st.columns(2)

    with res_col:
        st.markdown("**Candidate Resume**")
        resume_text = st.text_area(
            "Resume", value=default_resume, height=280, label_visibility="collapsed",
        )

    with jd_col:
        st.markdown("**Job Description**")
        jd_text = st.text_area(
            "JD", value=SAMPLE_JD, height=280, label_visibility="collapsed",
        )

    if st.button("Calculate Fit Score", type="primary", width="stretch"):
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Provide both a resume and a job description.")
        else:
            try:
                with st.spinner("Computing 384-dimensional semantic similarity vectors..."):
                    sbert_model = load_sbert_model()
                    result = score_resume(resume_text, jd_text, model=sbert_model)

                score = result["match_percentage"]
                recommendation = result["recommendation"]

                sc1, sc2 = st.columns([2, 1])

                with sc1:
                    st.progress(min(score / 100, 1.0), text=f"{score:.1f}% semantic match")

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={"suffix": "%", "font": {"color": "#FAFAFA"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                            "bar": {"color": _ALPINO_GOLD},
                            "bgcolor": "#262730",
                            "steps": [
                                {"range": [0, 45], "color": "#3E2723"},
                                {"range": [45, 65], "color": "#4E342E"},
                                {"range": [65, 100], "color": "#1B5E20"},
                            ],
                            "threshold": {
                                "line": {"color": _ALPINO_GREEN, "width": 3},
                                "thickness": 0.8,
                                "value": 65,
                            },
                        },
                    ))
                    fig_gauge.update_layout(
                        height=250,
                        margin={"l": 20, "r": 20, "t": 30, "b": 10},
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#FAFAFA"},
                    )
                    st.plotly_chart(fig_gauge, width="stretch")

                with sc2:
                    if recommendation == "SHORTLIST":
                        st.success(
                            f"**SHORTLIST**\n\n{score:.1f}% — strong semantic "
                            f"alignment with role requirements."
                        )
                    elif recommendation == "REVIEW":
                        st.warning(
                            f"**MANUAL REVIEW**\n\n{score:.1f}% — may have "
                            f"transferable skills worth evaluating."
                        )
                    else:
                        st.error(
                            f"**REJECT**\n\n{score:.1f}% — profile does not "
                            f"align with role requirements."
                        )

            except Exception as e:
                st.error(f"Resume scoring encountered an issue: {e}")

# ===========================================================================
# BATCH APPLICANT SCORING
# ===========================================================================
else:
    st.markdown("---")
    st.subheader("Batch Applicant Scoring")
    st.caption(
        "Upload a CSV with a `resume_text` column. Each row is scored against "
        "the Job Description using the cached SBERT model."
    )

    batch_jd_col, batch_upload_col = st.columns([1, 1], gap="large")

    with batch_jd_col:
        st.markdown("**Job Description (target role)**")
        batch_jd = st.text_area(
            "Batch JD",
            value=SAMPLE_JD,
            height=250,
            label_visibility="collapsed",
            key="batch_jd_input",
        )

    with batch_upload_col:
        st.markdown("**Applicant Roster CSV**")
        st.caption("Required column: `resume_text`. Optional: `name`, `email`.")
        batch_csv = st.file_uploader(
            "Upload Applicant Roster (CSV)",
            type=["csv"],
            key="batch_resume_csv",
        )

    if batch_csv is not None and batch_jd.strip():
        try:
            roster_df = pd.read_csv(batch_csv)

            # Detect the resume text column
            text_col = None
            for candidate_col in ["resume_text", "applicant_profiles", "resume", "profile", "text"]:
                if candidate_col in roster_df.columns:
                    text_col = candidate_col
                    break

            if text_col is None:
                st.error(
                    "CSV must contain a column named `resume_text`, `applicant_profiles`, "
                    "`resume`, `profile`, or `text`."
                )
            else:
                st.markdown(f"**{len(roster_df)} applicant(s) detected.** Scoring...")

                with st.spinner("Computing 384-dimensional semantic similarity vectors..."):
                    sbert_model = load_sbert_model()

                    scores = []
                    recommendations = []
                    total = len(roster_df)
                    progress = st.progress(0, text="Batch encoding applicant vectors...")

                    for i, (_, row) in enumerate(roster_df.iterrows()):
                        progress.progress(
                            (i + 1) / total,
                            text=f"Scoring applicant {i + 1}/{total}...",
                        )
                        resume = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        if not resume.strip():
                            scores.append(0.0)
                            recommendations.append("REJECT")
                            continue

                        try:
                            result = score_resume(resume, batch_jd, model=sbert_model)
                            scores.append(float(result["match_percentage"]))
                            recommendations.append(str(result["recommendation"]))
                        except Exception:
                            scores.append(0.0)
                            recommendations.append("ERROR")

                    progress.empty()

                roster_df["Match Score (%)"] = scores
                roster_df["Recommendation"] = recommendations

                # Sort by match score descending (leaderboard)
                leaderboard = roster_df.sort_values(
                    "Match Score (%)", ascending=False,
                ).reset_index(drop=True)

                # Summary metrics
                lb1, lb2, lb3, lb4 = st.columns(4)
                lb1.metric("Total Applicants", len(leaderboard))
                lb2.metric(
                    "Shortlisted",
                    len(leaderboard[leaderboard["Recommendation"] == "SHORTLIST"]),
                )
                lb3.metric(
                    "Review",
                    len(leaderboard[leaderboard["Recommendation"] == "REVIEW"]),
                )
                lb4.metric(
                    "Rejected",
                    len(leaderboard[leaderboard["Recommendation"] == "REJECT"]),
                )

                # Display columns — show name/email if available, plus score
                show_cols = []
                for c in ["name", "email", "Name", "Email"]:
                    if c in leaderboard.columns:
                        show_cols.append(c)
                show_cols += ["Match Score (%)", "Recommendation"]

                st.subheader("Applicant Leaderboard")
                st.dataframe(
                    leaderboard[show_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Match Score (%)": st.column_config.ProgressColumn(
                            "Match Score",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),
                    },
                )

        except Exception as e:
            st.error(f"Batch processing encountered an issue: {e}")

    elif batch_csv is None:
        st.info("Upload a CSV file with applicant resume text to begin batch scoring.")
