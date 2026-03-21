# Alpino AI Operating System v1.0

**Enterprise Decision Intelligence Platform for Alpino Health Foods**

---

## Executive Summary

The Alpino AI Operating System is an integrated analytics platform that addresses three critical operational gaps identified in Alpino Health Foods' growth trajectory:

| Problem | Impact | Solution |
|---|---|---|
| **25% invoice disparity rate** with 1-2 month reconciliation lag | ₹5-6L/yr in unrecovered revenue leakage + labor costs | 3-stage OCR extraction pipeline with deterministic validation engine |
| **19.44% early attrition** (tenure < 6 months) across 36 annual hires | ₹3-4L/yr wasted onboarding cost per cycle | XGBoost risk scoring + SBERT-powered resume screening to reduce portal dependency |
| **Campaign stockouts** during promotional spikes on Meta/Instagram | Lost ROAS on ₹13.14L paid social spend | RFM segmentation + demand-inventory alignment with predictive alerting |

**Projected Year-1 ROI:** ₹25-40L total investment with 12-18 month payback period.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Finance  │  │ HR Analytics │  │ Marketing Command │  │
│  │   OCR    │  │  Dashboard   │  │     Center        │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘  │
├───────┼────────────────┼──────────────────┼──────────────┤
│       │   modules/     │                  │              │
│  ┌────▼─────┐    ┌─────▼──────┐    ┌──────▼─────┐       │
│  │finance_  │    │hr_analytics│    │ In-page    │       │
│  │ocr.py    │    │.py         │    │ analytics  │       │
│  └────┬─────┘    └──┬─────┬───┘    └────────────┘       │
│       │             │     │                              │
│  pytesseract   XGBoost  SBERT                           │
│  (Tier 1 OCR)  (Risk)  (NLP)                            │
│       │                                                  │
│  VLM API *                                              │
│  (Tier 2 — Qwen-VL/Donut)                               │
└─────────────────────────────────────────────────────────┘
 * Tier 2 uses ground truth in dev mode; swap to VLM API for production
```

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.55+ | Multi-page dashboard with dark-mode Alpino branding |
| **OCR Tier 1** | pytesseract (Tesseract v4) | Baseline text extraction from invoice images |
| **OCR Tier 2** | transformers (HuggingFace) | Structured extraction via VLM (Qwen-VL / Donut) |
| **ML — Attrition** | XGBoost, scikit-learn | Gradient-boosted classification with risk scoring |
| **NLP — Resume** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim semantic embeddings for resume-JD matching |
| **Visualization** | Plotly | Interactive charts with Alpino brand palette |
| **Data Layer** | pandas, NumPy | Data processing, feature engineering |

---

## Directory Structure

```
alpino-ai-os/
├── app.py                              # Executive dashboard (entry point)
├── pages/
│   ├── 1_Finance_OCR.py                # Invoice pipeline + batch reconciliation
│   ├── 2_HR_Analytics.py               # Attrition model + batch resume scoring
│   └── 3_Marketing_Insights.py         # Channel economics + demand forecasting
├── modules/
│   ├── finance_ocr.py                  # OCR extraction, validation engine
│   └── hr_analytics.py                 # XGBoost model, SBERT scoring
├── scripts/
│   └── generate_data.py                # Synthetic data generation engine
├── data/
│   ├── raw/                            # Connect live Alpino data here
│   └── synthetic/
│       ├── hr/hr_data.csv              # 100 employee records (dev/QA)
│       └── finance/
│           ├── invoices/               # 50 GST invoice PNGs (dev/QA)
│           └── ground_truth/metadata.json
├── .streamlit/config.toml              # Alpino brand theme
└── requirements.txt                    # Python dependencies
```

---

## Installation Guide

### Prerequisites

- Python 3.10+
- Tesseract OCR v4+ (`sudo apt install tesseract-ocr` on Ubuntu/Debian)
- 4 GB RAM minimum (SBERT model requires ~500 MB)

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone <repository-url> alpino-ai-os
cd alpino-ai-os
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Generate Development Data

```bash
python scripts/generate_data.py
```

This produces:
- **HR Dataset** (`data/synthetic/hr/hr_data.csv`): 100 employee records with statistically enforced constraints matching Alpino's operational data.
- **Finance Dataset** (`data/synthetic/finance/`): 50 GST invoice images (PNG) with structured ground truth JSON. 25% of invoices contain deliberate errors mirroring the real-world disparity rate.

### Step 4: Launch the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`. The sidebar provides access to all three modules.

---

## Usage Guide

### Finance OCR

**Single Invoice Mode:**
1. Click **Load Random Sample** or upload a PNG/JPG invoice
2. The system extracts fields via OCR (Tier 1), loads structured data (Tier 2), and runs the 3-stage validation engine
3. Review flags: HIGH (arithmetic mismatch), MEDIUM (GST inconsistency), LOW (discount outside range)

**Batch Reconciliation Mode:**
1. Switch to **Batch Reconciliation** via the radio toggle
2. Upload multiple invoice images simultaneously
3. Each invoice is processed through the full pipeline
4. A consolidated summary table shows: File, Invoice #, Vendor, Extracted Total (₹), Discrepancy Flag, Status

### HR Analytics

**Attrition Dashboard:**
- KPIs, department-level attrition, XGBoost feature importance, and per-employee risk scores load automatically
- Upload a custom HR CSV via the sidebar to score your own workforce data

**Single Resume Screening:**
1. Paste a resume and job description
2. Click **Calculate Fit Score** — the SBERT model computes 384-dim semantic similarity
3. Results: SHORTLIST (≥65%), REVIEW (45-65%), REJECT (<45%)

**Batch Applicant Scoring:**
1. Switch to **Batch Processing** via the radio toggle
2. Upload a CSV with a `resume_text` column (optional: `name`, `email`)
3. All applicants are scored against the JD using the cached SBERT model
4. Output: ranked leaderboard sorted by match score with visual progress bars

### Marketing Command Center

- Channel economics with exact Alpino FY25 data
- RFM segmentation with budget reallocation modeling (ROAS 3.16x → 4.08x)
- 30-day demand simulation with campaign-adjusted stockout alerting

---

## Developer Notes — Production Migration

### Connecting Live Data Sources

#### Finance: UniCommerce / Tally Integration

Replace the synthetic data loader in `modules/finance_ocr.py`:

```python
# Current (dev mode):
GT_PATH = BASE_DIR / "data" / "synthetic" / "finance" / "ground_truth" / "metadata.json"

# Production: point to UniCommerce API export or Tally database
# Option A: Direct file path to live invoice metadata
GT_PATH = Path("/mnt/alpino-data/finance/invoice_metadata.json")

# Option B: Replace _load_ground_truth() with an API call
def _load_ground_truth() -> dict[str, dict]:
    response = requests.get("https://api.unicommerce.com/v1/invoices", headers=AUTH)
    return {inv["invoice_number"]: inv for inv in response.json()["data"]}
```

#### HR: HRMS Database Connection

Replace the CSV loader in `modules/hr_analytics.py`:

```python
# Current (dev mode):
HR_DATA_PATH = BASE_DIR / "data" / "synthetic" / "hr" / "hr_data.csv"

# Production: connect to Alpino's HRMS
# Option A: Direct database connection
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql://user:pass@hrms-host/alpino_hr")
def load_hr_data():
    return pd.read_sql("SELECT * FROM employees", engine)

# Option B: CSV export from HRMS (simplest)
HR_DATA_PATH = Path("/mnt/alpino-data/hr/employee_export.csv")
```

**Required columns for HR model:** `Employee_ID`, `Department`, `Tenure_Months`, `Training_Hours`, `Promotion_Received`, `Performance_Rating`, `Hiring_Source`, `Attrition_Target`

#### Marketing: Google Analytics / Meta Ads API

Channel data in `pages/3_Marketing_Insights.py` is currently hardcoded. To automate:

```python
# Replace the channel_data DataFrame with a live API call
from google.analytics.data_v1beta import BetaAnalyticsDataClient
client = BetaAnalyticsDataClient()
# ... fetch sessions, revenue by channel
```

### VLM Tier 2 Upgrade

To upgrade from ground-truth fallback to live VLM extraction, replace `load_ground_truth_for_invoice()` with a call to your deployed Qwen-VL or Donut endpoint:

```python
def extract_invoice_vlm(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://vlm-api.alpino.internal/extract",
            files={"image": f},
        )
    return response.json()
```

---

## Synthetic Data Constraints (QA Reference)

### HR Dataset (100 records)

| Metric | Target | Achieved |
|---|---|---|
| Hiring Source = Portal | 95% | 95.0% |
| Tenure Distribution | Exponential, mean=12 | mean ≈ 10-12 |
| Early Attrition Rate | 19.44% | 19.0% |
| E-commerce Attrition | 66.67% | ~63.6% |
| Sales Attrition | 53.33% | ~52.6% |
| Marketing Attrition | 43.47% | ~43.8% |
| Promotion Rate | 9.1% | 9.0% |

### Finance Dataset (50 invoices)

- 18 vendor pool with format-compliant Indian GSTINs
- 30-SKU Alpino product catalog
- Intra-state (CGST+SGST) and inter-state (IGST) GST handling
- **25% error injection rate** for validation engine testing
