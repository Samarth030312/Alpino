#!/usr/bin/env python3
"""
Synthetic Data Generation Engine
=================================

Generates two datasets that replicate Alpino Health Foods'
operational statistics for development and QA testing:

1. HR Dataset (100 records) — data/synthetic/hr/hr_data.csv
   - 95% Hiring_Source = 'Portal'
   - Exponential Tenure_Months distribution (mean=12)
   - 19.44% early attrition rate (tenure < 6 months)
   - Department-weighted attrition (E-comm=66.67%, Sales=53.33%, etc.)
   - Performance: 48.6% Outstanding, 37.1% Strong, 14.3% Satisfactory

2. Finance Dataset (50 invoices) — data/synthetic/finance/
   - Realistic Indian GST invoices rendered as PNG images
   - Ground truth metadata.json with exact extracted fields
   - 25% deliberate error injection (arithmetic/GST mismatches)
   - 15-20 vendor pool with format-compliant GSTINs
   - 2-5 SKUs per invoice with Alpino product catalog

Usage:
    python scripts/generate_data.py
"""

import json
import math
import os
import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
HR_DIR = BASE_DIR / "data" / "synthetic" / "hr"
INVOICE_DIR = BASE_DIR / "data" / "synthetic" / "finance" / "invoices"
GT_DIR = BASE_DIR / "data" / "synthetic" / "finance" / "ground_truth"

for d in (HR_DIR, INVOICE_DIR, GT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =========================================================================
# PART 1 — HR DATASET (100 RECORDS)
# =========================================================================

# Department distribution scaled from 57 → 100
DEPARTMENTS = {
    "Marketing": 32,
    "Sales": 19,
    "Finance": 12,
    "E-commerce": 11,
    "Exports": 7,
    "Purchase": 5,
    "HR": 4,
    "R&D": 4,
    "IT": 2,
    "Support": 4,
}  # total = 100

# Attrition rates per department (from Alpino operational data)
ATTRITION_RATES = {
    "Marketing": 0.4347,
    "Sales": 0.5333,
    "Finance": 0.2857,
    "E-commerce": 0.6667,
    "Exports": 0.2500,
    "Purchase": 0.2000,
    "HR": 0.5000,
    "R&D": 0.2500,
    "IT": 0.0000,
    "Support": 0.3333,
}

# Performance distribution
PERFORMANCE_WEIGHTS = {
    "Outstanding": 0.486,
    "Strong": 0.371,
    "Satisfactory": 0.143,
}

# Hiring source distribution
HIRING_SOURCES = {
    "Portal": 0.95,
    "Referral": 0.03,
    "Recruiter": 0.02,
}


def generate_hr_data(n: int = 100) -> pd.DataFrame:
    """Generate HR dataset with exact statistical constraints from Alpino operational data.

    The function enforces:
      - Department proportions matching scaled Alpino headcount
      - Exponential tenure distribution (mean=12 months, capped at 36)
      - Department-specific attrition via Bernoulli sampling
      - Exact 19.44% early attrition rate (tenure < 6 months AND attrited)
      - 95% Portal hiring source
      - Target performance rating distribution
    """
    records = []
    emp_id = 1000

    # Build flat department list
    dept_list = []
    for dept, count in DEPARTMENTS.items():
        dept_list.extend([dept] * count)
    random.shuffle(dept_list)

    # -------------------------------------------------------------------
    # Pre-compute deterministic hiring source assignments (exactly 95/3/2)
    # -------------------------------------------------------------------
    n_portal = round(n * 0.95)   # 95
    n_referral = round(n * 0.03)  # 3
    n_recruiter = n - n_portal - n_referral  # 2
    hiring_sources = (
        ["Portal"] * n_portal
        + ["Referral"] * n_referral
        + ["Recruiter"] * n_recruiter
    )
    random.shuffle(hiring_sources)

    # -------------------------------------------------------------------
    # Pre-compute deterministic department attrition counts
    # (closest integer to target rate per department)
    # -------------------------------------------------------------------
    dept_attrition_budget: dict[str, int] = {}
    for dept, count in DEPARTMENTS.items():
        target = ATTRITION_RATES[dept] * count
        dept_attrition_budget[dept] = round(target)

    # Build per-department index lists for attrition assignment
    dept_indices: dict[str, list[int]] = {d: [] for d in DEPARTMENTS}
    for i, dept in enumerate(dept_list):
        dept_indices[dept].append(i)

    # Pre-compute deterministic promotion (exactly 9 out of 100 = 9%)
    n_promo = round(n * 0.091)
    promo_flags = [1] * n_promo + [0] * (n - n_promo)
    random.shuffle(promo_flags)

    # Determine which indices get attrition = 1 (deterministic per dept)
    attrition_set: set[int] = set()
    for dept, budget in dept_attrition_budget.items():
        indices = dept_indices[dept][:]
        random.shuffle(indices)
        for idx in indices[:budget]:
            attrition_set.add(idx)

    for i, dept in enumerate(dept_list):
        emp_id += 1

        # Tenure: exponential(mean=12), capped [1, 36]
        tenure = int(np.clip(np.random.exponential(scale=12), 1, 36))

        # Performance rating (weighted)
        perf = np.random.choice(
            list(PERFORMANCE_WEIGHTS.keys()),
            p=list(PERFORMANCE_WEIGHTS.values()),
        )

        # Training hours: 0-24 range, skewed low (mean ~12)
        training_hrs = int(np.clip(np.random.normal(12, 6), 0, 48))

        records.append(
            {
                "Employee_ID": f"ALP-{emp_id}",
                "Department": dept,
                "Tenure_Months": tenure,
                "Performance_Rating": perf,
                "Hiring_Source": hiring_sources[i],
                "Training_Hours": training_hrs,
                "Promotion_Received": promo_flags[i],
                "Attrition_Target": 1 if i in attrition_set else 0,
            }
        )

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Enforce 19.44% early attrition (tenure < 6 AND Attrition_Target = 1)
    # Strategy: Among the employees already marked as attrited (via
    # department quotas), force exactly 19 to have short tenure (< 6).
    # This preserves department-level attrition counts.
    # -----------------------------------------------------------------------
    target_early = round(n * 0.1944)  # 19

    attrited_mask = df["Attrition_Target"] == 1
    attrited_indices = df.index[attrited_mask].tolist()

    # Count how many attrited employees already have tenure < 6
    early_now = df.loc[attrited_mask & (df["Tenure_Months"] < 6)].index.tolist()
    current_early = len(early_now)

    if current_early < target_early:
        # Need more attrited employees to have short tenure
        long_attrited = [
            i for i in attrited_indices if df.at[i, "Tenure_Months"] >= 6
        ]
        random.shuffle(long_attrited)
        for idx in long_attrited[: target_early - current_early]:
            df.at[idx, "Tenure_Months"] = random.randint(1, 5)
    elif current_early > target_early:
        # Too many early attritors — extend tenure for some
        random.shuffle(early_now)
        for idx in early_now[: current_early - target_early]:
            df.at[idx, "Tenure_Months"] = random.randint(6, 18)

    return df


# =========================================================================
# PART 2 — FINANCE DATASET (50 INVOICES)
# =========================================================================

# Alpino product catalog (~30 SKUs)
SKU_CATALOG = [
    ("ALB-PB-CRN-1KG", "Alpino Peanut Butter Crunch 1kg", 599.00),
    ("ALB-PB-SMO-1KG", "Alpino Peanut Butter Smooth 1kg", 579.00),
    ("ALB-PB-CHO-1KG", "Alpino Peanut Butter Chocolate 1kg", 649.00),
    ("ALB-PB-HNY-1KG", "Alpino Peanut Butter Honey 1kg", 629.00),
    ("ALB-PB-CRN-400", "Alpino Peanut Butter Crunch 400g", 299.00),
    ("ALB-PB-SMO-400", "Alpino Peanut Butter Smooth 400g", 279.00),
    ("ALB-PB-CHO-400", "Alpino Peanut Butter Chocolate 400g", 329.00),
    ("ALB-PB-NAT-1KG", "Alpino Natural Peanut Butter 1kg", 549.00),
    ("ALB-PB-NAT-400", "Alpino Natural Peanut Butter 400g", 269.00),
    ("ALB-PB-SUP-1KG", "Alpino Super Peanut Butter 1kg", 749.00),
    ("ALB-PB-CAS-250", "Alpino Cashew Butter 250g", 449.00),
    ("ALB-PB-ALM-250", "Alpino Almond Butter 250g", 499.00),
    ("ALB-PB-COC-1KG", "Alpino Coconut Peanut Butter 1kg", 679.00),
    ("ALB-CS-HAZ-350", "Alpino Chocolate Spread Hazelnut 350g", 399.00),
    ("ALB-CS-DRK-350", "Alpino Chocolate Spread Dark 350g", 379.00),
    ("ALB-CS-WHT-350", "Alpino Chocolate Spread White 350g", 389.00),
    ("ALB-PB-PRO-1KG", "Alpino Protein Peanut Butter 1kg", 799.00),
    ("ALB-PB-PRO-400", "Alpino Protein Peanut Butter 400g", 399.00),
    ("ALB-BR-PRO-60G", "Alpino Protein Bar 60g", 99.00),
    ("ALB-BR-ENR-60G", "Alpino Energy Bar 60g", 79.00),
    ("ALB-BR-OAT-40G", "Alpino Oat Bar 40g", 59.00),
    ("ALB-MG-CHL-200", "Alpino Muesli Chocolate 200g", 249.00),
    ("ALB-MG-NUT-200", "Alpino Muesli Nuts & Seeds 200g", 269.00),
    ("ALB-MG-FRT-200", "Alpino Muesli Fruits 200g", 259.00),
    ("ALB-GR-OAT-1KG", "Alpino Rolled Oats 1kg", 349.00),
    ("ALB-GR-OAT-500", "Alpino Rolled Oats 500g", 199.00),
    ("ALB-SY-MAP-320", "Alpino Maple Syrup 320ml", 449.00),
    ("ALB-SY-HNY-500", "Alpino Organic Honey 500g", 399.00),
    ("ALB-PW-WHP-1KG", "Alpino Whey Protein 1kg", 1899.00),
    ("ALB-PW-WHP-500", "Alpino Whey Protein 500g", 999.00),
]

# Indian state codes for GSTIN generation
STATE_CODES = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "29", "30", "32",
    "33", "34", "35", "36", "37",
]

CHANNELS = ["ECOMM", "D2C", "OFFLINE", "GT", "MT"]

DISCOUNT_RANGES = {
    "ECOMM": (15, 25),
    "D2C": (5, 15),
    "OFFLINE": (5, 10),
    "GT": (8, 12),
    "MT": (10, 20),
}


def generate_gstin() -> str:
    """Generate a format-compliant Indian GSTIN.

    Format: 2-digit state code + 10-char PAN-like + 1 entity + Z + check digit
    Example: 24AABCU9603R1ZP
    """
    state = random.choice(STATE_CODES)
    pan_chars = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    pan_digits = "".join(random.choices("0123456789", k=4))
    pan_last = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    entity = str(random.randint(1, 9))
    check = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return f"{state}{pan_chars}{pan_digits}{pan_last}{entity}Z{check}"


def build_vendor_pool(n: int = 18) -> list[dict]:
    """Create a pool of realistic Indian vendor entities."""
    vendors = []
    suffixes = [
        "Foods Pvt Ltd", "Enterprises", "Trading Co", "Industries",
        "Distributors", "Agro Products", "Marketing Pvt Ltd",
        "Supply Chain LLP", "Retail Solutions", "Exporters",
    ]
    cities = [
        "Ahmedabad", "Mumbai", "Delhi", "Bengaluru", "Hyderabad",
        "Chennai", "Pune", "Jaipur", "Surat", "Lucknow",
        "Kolkata", "Indore", "Nagpur", "Vadodara", "Rajkot",
        "Chandigarh", "Coimbatore", "Kochi",
    ]
    for i in range(n):
        city = cities[i % len(cities)]
        name = f"{fake.last_name()} {random.choice(suffixes)}"
        vendors.append(
            {
                "vendor_id": f"VND-{i + 1:03d}",
                "vendor_name": name,
                "gstin": generate_gstin(),
                "address": f"{fake.street_address()}, {city}",
                "city": city,
                "state_code": random.choice(STATE_CODES),
            }
        )
    return vendors


def generate_invoice(
    invoice_num: int,
    vendor: dict,
    inject_error: bool = False,
) -> dict:
    """Generate a single invoice with structured metadata.

    Args:
        invoice_num: Sequential invoice number.
        vendor: Vendor dict from the pool.
        inject_error: If True, introduce deliberate discrepancies.

    Returns:
        Dict with full invoice metadata (ground truth).
    """
    channel = random.choice(CHANNELS)
    disc_lo, disc_hi = DISCOUNT_RANGES[channel]
    num_items = random.randint(2, 5)
    items_selected = random.sample(SKU_CATALOG, num_items)

    # Alpino GSTIN (buyer) — Gujarat state code 24
    alpino_gstin = "24AABCA1234B1Z5"
    # Determine intra-state vs inter-state
    is_intra = vendor["state_code"] == "24"

    line_items = []
    for sku_code, sku_name, base_price in items_selected:
        qty = random.randint(5, 200)
        discount_pct = round(random.uniform(disc_lo, disc_hi), 2)
        unit_price = round(base_price, 2)
        taxable_value = round(unit_price * qty * (1 - discount_pct / 100), 2)

        # GST: food products typically 5% or 12%
        gst_rate = random.choice([5, 12])

        if is_intra:
            cgst = round(taxable_value * (gst_rate / 2) / 100, 2)
            sgst = round(taxable_value * (gst_rate / 2) / 100, 2)
            igst = 0.0
        else:
            cgst = 0.0
            sgst = 0.0
            igst = round(taxable_value * gst_rate / 100, 2)

        total = round(taxable_value + cgst + sgst + igst, 2)

        line_items.append(
            {
                "sku": sku_code,
                "description": sku_name,
                "qty": qty,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "taxable_value": taxable_value,
                "gst_rate": gst_rate,
                "cgst": cgst,
                "sgst": sgst,
                "igst": igst,
                "line_total": total,
            }
        )

    grand_total_computed = round(sum(item["line_total"] for item in line_items), 2)
    grand_total = grand_total_computed

    error_type = None
    if inject_error:
        error_kind = random.choice(["arithmetic", "gst_rate", "both"])
        if error_kind in ("arithmetic", "both"):
            # Offset grand total by a noticeable amount
            offset = round(random.uniform(50, 500) * random.choice([-1, 1]), 2)
            grand_total = round(grand_total + offset, 2)
            error_type = "arithmetic_mismatch"
        if error_kind in ("gst_rate", "both"):
            # Corrupt one line item's GST rate
            target = random.randint(0, len(line_items) - 1)
            original_rate = line_items[target]["gst_rate"]
            wrong_rate = 18 if original_rate != 18 else 28
            line_items[target]["gst_rate"] = wrong_rate
            error_type = (
                "gst_rate_mismatch"
                if error_type is None
                else "arithmetic_and_gst_mismatch"
            )

    inv_date = fake.date_between(start_date="-6m", end_date="today")

    return {
        "invoice_number": f"INV-{inv_date.year}-{invoice_num:04d}",
        "invoice_date": inv_date.isoformat(),
        "vendor_id": vendor["vendor_id"],
        "vendor_name": vendor["vendor_name"],
        "vendor_gstin": vendor["gstin"],
        "vendor_address": vendor["address"],
        "buyer_gstin": alpino_gstin,
        "channel": channel,
        "line_items": line_items,
        "grand_total_computed": grand_total_computed,
        "grand_total": grand_total,
        "has_error": inject_error,
        "error_type": error_type,
    }


# ---------------------------------------------------------------------------
# Invoice Image Rendering (Pillow)
# ---------------------------------------------------------------------------

def _try_load_font(size: int):
    """Attempt to load a monospace or sans-serif font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_invoice_image(invoice: dict, output_path: Path) -> None:
    """Render an invoice dict as a PNG image with slight visual noise.

    Introduces minor rotation and font-size jitter to simulate real-world
    scan variability for OCR testing.
    """
    width, height = 900, 1300
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_title = _try_load_font(22)
    font_header = _try_load_font(16)
    font_body = _try_load_font(13)
    font_small = _try_load_font(11)

    y = 30

    # --- Header ---
    draw.text((30, y), "TAX INVOICE", fill="black", font=font_title)
    y += 35
    draw.line([(30, y), (width - 30, y)], fill="black", width=2)
    y += 15

    # Vendor info
    draw.text((30, y), f"Vendor: {invoice['vendor_name']}", fill="black", font=font_header)
    y += 22
    draw.text((30, y), f"GSTIN: {invoice['vendor_gstin']}", fill="black", font=font_body)
    y += 18
    draw.text((30, y), f"Address: {invoice['vendor_address']}", fill="gray", font=font_small)
    y += 25

    # Invoice meta
    draw.text((30, y), f"Invoice #: {invoice['invoice_number']}", fill="black", font=font_body)
    draw.text((500, y), f"Date: {invoice['invoice_date']}", fill="black", font=font_body)
    y += 18
    draw.text((30, y), f"Buyer GSTIN: {invoice['buyer_gstin']}", fill="black", font=font_body)
    draw.text((500, y), f"Channel: {invoice['channel']}", fill="black", font=font_body)
    y += 25

    draw.line([(30, y), (width - 30, y)], fill="black", width=1)
    y += 10

    # --- Table header ---
    cols = [30, 80, 370, 420, 480, 540, 600, 660, 720, 790]
    headers = ["#", "Description", "Qty", "Price", "Disc%", "Taxable", "GST%", "CGST", "SGST/IGST", "Total"]
    for i, h in enumerate(headers):
        draw.text((cols[i], y), h, fill="black", font=font_small)
    y += 16
    draw.line([(30, y), (width - 30, y)], fill="gray", width=1)
    y += 8

    # --- Line items ---
    for idx, item in enumerate(invoice["line_items"], 1):
        draw.text((cols[0], y), str(idx), fill="black", font=font_small)
        # Truncate long descriptions
        desc = item["description"][:35]
        draw.text((cols[1], y), desc, fill="black", font=font_small)
        draw.text((cols[2], y), str(item["qty"]), fill="black", font=font_small)
        draw.text((cols[3], y), f"{item['unit_price']:.0f}", fill="black", font=font_small)
        draw.text((cols[4], y), f"{item['discount_pct']:.1f}", fill="black", font=font_small)
        draw.text((cols[5], y), f"{item['taxable_value']:.2f}", fill="black", font=font_small)
        draw.text((cols[6], y), f"{item['gst_rate']}%", fill="black", font=font_small)
        draw.text((cols[7], y), f"{item['cgst']:.2f}", fill="black", font=font_small)
        sgst_igst = item["sgst"] if item["sgst"] > 0 else item["igst"]
        draw.text((cols[8], y), f"{sgst_igst:.2f}", fill="black", font=font_small)
        draw.text((cols[9], y), f"{item['line_total']:.2f}", fill="black", font=font_small)
        y += 18

    y += 10
    draw.line([(30, y), (width - 30, y)], fill="black", width=2)
    y += 15

    # --- Grand total ---
    draw.text(
        (600, y),
        f"GRAND TOTAL: Rs. {invoice['grand_total']:.2f}",
        fill="black",
        font=font_header,
    )
    y += 30

    if invoice["has_error"]:
        # Subtle watermark for debugging (would not exist in production)
        draw.text(
            (30, height - 40),
            f"[DEBUG: {invoice['error_type']}]",
            fill=(220, 220, 220),
            font=font_small,
        )

    # --- Visual noise: slight rotation ---
    rotation_angle = random.uniform(-1.5, 1.5)
    img = img.rotate(rotation_angle, expand=False, fillcolor="white")

    img.save(str(output_path), "PNG")


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main() -> None:
    """Generate all synthetic datasets and verify constraints."""
    print("=" * 60)
    print("ALPINO AI — SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. HR Dataset
    # ------------------------------------------------------------------
    print("\n[1/2] Generating HR dataset (100 records)...")
    df_hr = generate_hr_data(100)
    hr_path = HR_DIR / "hr_data.csv"
    df_hr.to_csv(hr_path, index=False)
    print(f"  -> Saved to {hr_path}")

    # Verification
    total = len(df_hr)
    portal_pct = (df_hr["Hiring_Source"] == "Portal").sum() / total * 100
    avg_tenure = df_hr["Tenure_Months"].mean()
    early = df_hr[(df_hr["Tenure_Months"] < 6) & (df_hr["Attrition_Target"] == 1)]
    early_pct = len(early) / total * 100
    overall_attrition = df_hr["Attrition_Target"].mean() * 100

    print(f"\n  HR Data Verification:")
    print(f"    Total records:           {total}")
    print(f"    Portal hiring:           {portal_pct:.1f}% (target: 95%)")
    print(f"    Mean tenure (months):    {avg_tenure:.1f} (target: ~12)")
    print(f"    Early attrition rate:    {early_pct:.2f}% (target: 19.44%)")
    print(f"    Overall attrition:       {overall_attrition:.1f}%")
    print(f"    Departments:")
    for dept in sorted(DEPARTMENTS.keys()):
        dept_df = df_hr[df_hr["Department"] == dept]
        dept_attr = dept_df["Attrition_Target"].mean() * 100
        print(f"      {dept:15s}: {len(dept_df):3d} records, attrition={dept_attr:.1f}%")

    # ------------------------------------------------------------------
    # 2. Finance Dataset (50 invoices)
    # ------------------------------------------------------------------
    print("\n[2/2] Generating Finance dataset (50 invoices)...")
    vendors = build_vendor_pool(18)
    all_invoices = []

    # Exactly 25% with errors = 12 or 13 invoices
    error_count = round(50 * 0.25)
    error_indices = set(random.sample(range(50), error_count))

    for i in range(50):
        vendor = random.choice(vendors)
        inject_err = i in error_indices
        invoice = generate_invoice(i + 1, vendor, inject_error=inject_err)
        all_invoices.append(invoice)

        # Render image
        img_path = INVOICE_DIR / f"invoice_{i + 1:03d}.png"
        render_invoice_image(invoice, img_path)

    # Save ground truth
    gt_path = GT_DIR / "metadata.json"
    with open(gt_path, "w") as f:
        json.dump(all_invoices, f, indent=2, default=str)
    print(f"  -> Saved {len(all_invoices)} invoice images to {INVOICE_DIR}")
    print(f"  -> Saved ground truth to {gt_path}")

    # Verification
    err_invoices = [inv for inv in all_invoices if inv["has_error"]]
    err_pct = len(err_invoices) / len(all_invoices) * 100
    print(f"\n  Finance Data Verification:")
    print(f"    Total invoices:          {len(all_invoices)}")
    print(f"    With errors:             {len(err_invoices)} ({err_pct:.0f}%)")
    print(f"    Error types:")
    for inv in err_invoices:
        print(f"      {inv['invoice_number']}: {inv['error_type']}")

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
