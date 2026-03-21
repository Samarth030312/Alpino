"""
HR Analytics Engine — Attrition Prediction & Resume Screening
=============================================================

  Layer 1 — Predictive Attrition Model (XGBoost):
    Trains on HR data to predict Attrition_Target.
    Returns risk scores (0-1) per employee and feature importances
    for SHAP-style interpretability.

  Layer 2 — NLP Resume Screening (Sentence-BERT):
    Encodes resume and job description text into dense vectors,
    computes cosine similarity as a fit score.
    Threshold: >= 0.65 → Shortlist, < 0.65 → Review/Reject.

Usage:
    from modules.hr_analytics import (
        load_hr_data,
        train_attrition_model,
        predict_attrition_risk,
        score_resume,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
HR_DATA_PATH = BASE_DIR / "data" / "synthetic" / "hr" / "hr_data.csv"


# =========================================================================
# DATA LOADING
# =========================================================================

def load_hr_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load HR dataset from CSV."""
    p = Path(path) if path else HR_DATA_PATH
    return pd.read_csv(p)


def compute_hr_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level HR KPIs from the dataset."""
    total = len(df)
    attrition_rate = df["Attrition_Target"].mean() * 100
    avg_tenure = df["Tenure_Months"].mean()

    # Early attrition: tenure < 6 AND attrited
    early_mask = (df["Tenure_Months"] < 6) & (df["Attrition_Target"] == 1)
    early_attrition_rate = early_mask.sum() / total * 100

    portal_pct = (df["Hiring_Source"] == "Portal").mean() * 100
    promotion_rate = df["Promotion_Received"].mean() * 100
    avg_training = df["Training_Hours"].mean()

    # Per-department attrition
    dept_attrition = (
        df.groupby("Department")["Attrition_Target"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "attrition_rate", "count": "headcount"})
        .sort_values("attrition_rate", ascending=False)
    )
    dept_attrition["attrition_rate"] = (dept_attrition["attrition_rate"] * 100).round(1)

    return {
        "total_employees": total,
        "overall_attrition_rate": round(attrition_rate, 1),
        "early_attrition_rate": round(early_attrition_rate, 2),
        "avg_tenure_months": round(avg_tenure, 1),
        "portal_hiring_pct": round(portal_pct, 1),
        "promotion_rate": round(promotion_rate, 1),
        "avg_training_hours": round(avg_training, 1),
        "dept_attrition": dept_attrition,
    }


# =========================================================================
# LAYER 1 — ATTRITION PREDICTION (XGBoost)
# =========================================================================

def _preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode categorical variables and return feature matrix.

    Uses pd.get_dummies for Department, Performance_Rating, Hiring_Source.
    Returns (X dataframe, list of feature names).
    """
    feature_cols = [
        "Tenure_Months",
        "Training_Hours",
        "Promotion_Received",
        "Department",
        "Performance_Rating",
        "Hiring_Source",
    ]
    df_feat = df[feature_cols].copy()
    df_encoded = pd.get_dummies(
        df_feat,
        columns=["Department", "Performance_Rating", "Hiring_Source"],
        drop_first=False,
    )
    return df_encoded, list(df_encoded.columns)


@st.cache_resource(show_spinner="Training XGBoost attrition model...")
def train_attrition_model(
    data_path: str | None = None,
) -> dict[str, Any]:
    """Train XGBoost classifier on HR data.

    Returns dict with: model, feature_names, feature_importance,
    accuracy, roc_auc, classification_report, X_test, y_test.
    """
    df = load_hr_data(data_path)
    X, feature_names = _preprocess_features(df)
    y = df["Attrition_Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Handle class imbalance with scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.0
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance (gain-based)
    importance = model.feature_importances_
    feat_imp = dict(zip(feature_names, importance))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    return {
        "model": model,
        "feature_names": feature_names,
        "feature_importance": feat_imp,
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "classification_report": report,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict_attrition_risk(
    df: pd.DataFrame,
    model_result: dict[str, Any],
) -> pd.DataFrame:
    """Score all employees with attrition risk probabilities.

    Returns the original df with added columns:
      risk_score (0-1), risk_category (High/Medium/Low).
    """
    model = model_result["model"]
    feature_names = model_result["feature_names"]

    X, _ = _preprocess_features(df)

    # Align columns to training features (handles missing dummies)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["risk_score"] = np.round(proba, 3)
    result["risk_category"] = pd.cut(
        proba,
        bins=[-0.01, 0.4, 0.7, 1.01],
        labels=["Low", "Medium", "High"],
    )

    return result


# =========================================================================
# LAYER 2 — RESUME SCREENING (Sentence-BERT)
# =========================================================================

@st.cache_resource(show_spinner="Loading NLP model (all-MiniLM-L6-v2)...")
def load_sbert_model():
    """Load and cache the Sentence-BERT model.

    This is the ONLY function that should be cached. Scoring functions
    must call this to get the model, then compute embeddings live so
    the UI updates instantly when new text is provided.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def score_resume(resume_text: str, jd_text: str, model=None) -> dict[str, Any]:
    """Compute semantic similarity between a resume and job description.

    Uses all-MiniLM-L6-v2 (384-dim embeddings) with cosine similarity.
    This function is NOT cached — it computes live on every call.

    Args:
        resume_text: Candidate resume text.
        jd_text: Job description text.
        model: Pre-loaded SentenceTransformer. If None, loads via cache.

    Returns:
        Dict with similarity_score, recommendation, and threshold.
    """
    if model is None:
        model = load_sbert_model()
    embeddings = model.encode([resume_text, jd_text])

    # Cosine similarity
    from numpy.linalg import norm
    cos_sim = float(
        np.dot(embeddings[0], embeddings[1])
        / (norm(embeddings[0]) * norm(embeddings[1]) + 1e-8)
    )

    threshold = 0.65
    if cos_sim >= threshold:
        recommendation = "SHORTLIST"
    elif cos_sim >= 0.45:
        recommendation = "REVIEW"
    else:
        recommendation = "REJECT"

    return {
        "similarity_score": round(cos_sim, 4),
        "match_percentage": round(cos_sim * 100, 1),
        "recommendation": recommendation,
        "threshold": threshold,
    }


# =========================================================================
# SAMPLE DATA FOR DEMO
# =========================================================================

SAMPLE_JD = """Marketing Executive — Alpino Health Foods

Location: Ahmedabad, Gujarat
Department: Marketing
Experience: 2-4 years in D2C / FMCG marketing

Responsibilities:
- Plan and execute digital marketing campaigns on Meta, Instagram, and Google Ads
- Manage influencer partnerships and brand collaborations
- Analyze campaign performance metrics (CAC, ROAS, CTR, CPC) and optimize spend
- Coordinate with creative team for ad creatives and content calendar
- Monitor competitor activity and market trends in health food / nutrition space
- Drive customer acquisition through paid social, organic social, and email marketing
- Work with the inventory team to align promotional campaigns with stock levels

Requirements:
- 2+ years experience in performance marketing for D2C or FMCG brands
- Proficiency in Meta Business Suite, Google Ads, Google Analytics
- Strong analytical skills with experience in A/B testing
- Understanding of e-commerce platforms (Shopify, Amazon, Flipkart)
- Excellent communication skills in English and Hindi
- Bachelor's degree in Marketing, Business, or related field
- Experience with health/wellness/nutrition brands is a plus
"""

SAMPLE_RESUME_GOOD = """Priya Sharma
Digital Marketing Specialist | Ahmedabad

Experience:
Senior Marketing Executive, FreshBasket D2C Foods (2023-2026)
- Managed Rs. 15L monthly ad spend across Meta, Google, and Amazon PPC
- Achieved ROAS of 4.2x on Instagram campaigns, reducing CAC from Rs. 8 to Rs. 4.5
- Led influencer marketing program with 50+ micro-influencers, driving 30% brand awareness lift
- Implemented A/B testing framework resulting in 25% improvement in CTR
- Coordinated with supply chain team to prevent promotional stockouts

Marketing Associate, NutriWell Health Foods (2021-2023)
- Executed performance marketing campaigns on Meta Business Suite and Google Ads
- Analyzed weekly campaign metrics and presented ROAS/CAC reports to leadership
- Created content calendar and managed social media for 3 product lines

Education: MBA Marketing, Gujarat University (2021)
Skills: Meta Ads, Google Ads, Google Analytics, Shopify, Amazon Seller Central,
A/B Testing, Canva, Excel, SQL, Hindi, English, Gujarati
"""

SAMPLE_RESUME_POOR = """Rahul Patel
Software Developer | Pune

Experience:
Junior Developer, TechSoft Solutions (2024-2026)
- Developed REST APIs using Python Flask and PostgreSQL
- Built automated testing pipelines with pytest and Jenkins CI/CD
- Managed deployment on AWS EC2 and S3 buckets
- Participated in agile sprints and daily standups

Intern, DataMinds Analytics (2023-2024)
- Assisted in building data dashboards using Tableau
- Wrote SQL queries for data extraction and reporting
- Learned basics of machine learning with scikit-learn

Education: B.Tech Computer Science, Pune University (2023)
Skills: Python, Java, SQL, Flask, Django, AWS, Git, Docker, Linux
"""
