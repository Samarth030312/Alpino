"""
Finance OCR Engine — Invoice Extraction & Validation
=====================================================

Two-tier extraction architecture for Alpino's invoice reconciliation pipeline.

  Tier 1 — OCR Extraction (pytesseract):
    Extracts raw text from invoice images, parses header fields (vendor,
    GSTIN, invoice number, date, channel) and grand total via regex.
    Line items are parsed with position-aware heuristics.

  Tier 2 — Structured Extraction (VLM / ground truth fallback):
    In production, this tier calls the VLM API (Qwen-VL / Donut) for
    structured line-item extraction. In dev mode, loads pre-generated
    metadata.json for deterministic testing.

The validation engine runs identical deterministic rules regardless of
extraction source.

Usage:
    from modules.finance_ocr import (
        extract_invoice_ocr,
        load_ground_truth_for_invoice,
        validate_invoice,
    )
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
GT_PATH = BASE_DIR / "data" / "synthetic" / "finance" / "ground_truth" / "metadata.json"
INVOICE_DIR = BASE_DIR / "data" / "synthetic" / "finance" / "invoices"

# Preload ground truth index
_GT_CACHE: dict[str, dict] | None = None


def _load_ground_truth() -> dict[str, dict]:
    """Load and index ground truth by filename and invoice number."""
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE
    if not GT_PATH.exists():
        _GT_CACHE = {}
        return _GT_CACHE
    with open(GT_PATH) as f:
        data = json.load(f)
    _GT_CACHE = {}
    for i, inv in enumerate(data):
        fname = f"invoice_{i + 1:03d}.png"
        _GT_CACHE[fname] = inv
        _GT_CACHE[inv["invoice_number"]] = inv
    return _GT_CACHE


# =========================================================================
# TIER 1 — OCR EXTRACTION (pytesseract)
# =========================================================================

def _parse_header_fields(raw_text: str) -> dict[str, Any]:
    """Extract header-level fields using regex on OCR raw text."""
    result: dict[str, Any] = {}

    m = re.search(r"Vendor:\s*(.+)", raw_text)
    result["vendor_name"] = m.group(1).strip() if m else ""

    m = re.search(r"GSTIN:\s*([A-Z0-9]{14,16})", raw_text)
    result["vendor_gstin"] = m.group(1).strip() if m else ""

    m = re.search(
        r"(?:Invoice|Inv|Tavoice)\s*#[\s:]*([A-Z]*-?\d{4}-\d{3,5})",
        raw_text,
        re.IGNORECASE,
    )
    result["invoice_number"] = m.group(1).strip() if m else ""

    m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
    result["invoice_date"] = m.group(1).strip() if m else ""

    m = re.search(r"Buyer\s+GSTIN:\s*([A-Z0-9]{14,16})", raw_text)
    result["buyer_gstin"] = m.group(1).strip() if m else ""

    m = re.search(r"[Cc]hannel:\s*(\w+)", raw_text)
    result["channel"] = m.group(1).strip() if m else ""

    return result


def _parse_grand_total(raw_text: str) -> float:
    """Extract grand total from OCR text."""
    m = re.search(
        r"GRAND\s+TOTAL[:\s]*Rs\.?\s*([\d,]+\.?\d*)",
        raw_text,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1).replace(",", "").strip())
        except ValueError:
            pass
    return 0.0


def _parse_line_items_ocr(raw_text: str) -> list[dict[str, Any]]:
    """Best-effort line item extraction from raw OCR text.

    Uses flexible regex to find lines containing 'Alp' (Alpino products)
    followed by numeric fields. Handles common OCR artifacts: merged
    columns, special characters, and misread digits.
    """
    items: list[dict[str, Any]] = []
    lines = raw_text.split("\n")

    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect table header (case-insensitive, tolerates OCR errors)
        lower = stripped.lower()
        if ("escription" in lower or "desc" in lower) and (
            "qty" in lower or "oty" in lower or "price" in lower
        ):
            in_table = True
            continue

        if "GRAND TOTAL" in stripped.upper():
            break

        if not in_table:
            continue

        # Skip separator lines
        if all(c in "-—_ |" for c in stripped):
            continue

        # Try to parse line items containing Alpino products
        item = _try_parse_line(stripped)
        if item is not None:
            items.append(item)

    return items


def _try_parse_line(line: str) -> dict[str, Any] | None:
    """Parse a single OCR line into a line item dict."""
    # Clean common OCR artifacts
    line = line.replace("©", "0.00").replace("|", " ").replace("—", " ")
    line = line.replace("–", " ").replace("~", " ").replace("«", " ")
    line = line.replace("*", " ").replace("=", " ")

    # Must contain 'Alp' (Alpino product)
    if not re.search(r"Alp", line, re.IGNORECASE):
        return None

    # Strategy: extract the description (Alp...until numbers dominate),
    # then parse trailing numbers right-to-left (most reliable).
    # The last number is typically the line total.

    # Find all decimal/integer numbers in the line
    all_nums = re.findall(r"\d+\.\d+|\d{2,}", line)
    float_nums = []
    for n in all_nums:
        try:
            float_nums.append(float(n))
        except ValueError:
            continue

    if len(float_nums) < 2:
        return None

    # Extract description — everything from "Alp" to where numbers start
    desc_match = re.search(r"(Alp\w+(?:\s+\w+){1,6})", line, re.IGNORECASE)
    desc = desc_match.group(1).strip() if desc_match else "Unknown"
    # Clean trailing numeric fragments from description
    desc = re.sub(r"\s+\d+[gkmls]*$", "", desc, flags=re.IGNORECASE).strip()

    item: dict[str, Any] = {
        "sku": "",
        "description": desc,
        "qty": 0,
        "unit_price": 0.0,
        "discount_pct": 0.0,
        "taxable_value": 0.0,
        "gst_rate": 0,
        "cgst": 0.0,
        "sgst": 0.0,
        "igst": 0.0,
        "line_total": float_nums[-1] if float_nums else 0.0,
    }

    return item


def extract_invoice_ocr(image_path: str | Path) -> dict[str, Any]:
    """Extract structured invoice data from an image using pytesseract.

    No preprocessing is applied (raw images yield better OCR on our
    Pillow-rendered invoices).

    Returns:
        Dict with extracted fields plus ``ocr_raw_text`` and
        ``ocr_confidence`` metadata.
    """
    image_path = Path(image_path)
    img = Image.open(image_path)

    raw_text = pytesseract.image_to_string(img)

    # Confidence calculation
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    header = _parse_header_fields(raw_text)
    line_items = _parse_line_items_ocr(raw_text)
    grand_total = _parse_grand_total(raw_text)

    grand_total_computed = round(
        sum(item.get("line_total", 0) for item in line_items), 2
    )

    return {
        "invoice_number": header.get("invoice_number", ""),
        "invoice_date": header.get("invoice_date", ""),
        "vendor_name": header.get("vendor_name", ""),
        "vendor_gstin": header.get("vendor_gstin", ""),
        "buyer_gstin": header.get("buyer_gstin", ""),
        "channel": header.get("channel", ""),
        "line_items": line_items,
        "grand_total": grand_total,
        "grand_total_computed": grand_total_computed,
        "ocr_raw_text": raw_text,
        "ocr_confidence": round(avg_conf, 1),
        "extraction_method": "pytesseract_v4",
    }


# =========================================================================
# TIER 2 — STRUCTURED EXTRACTION (Ground Truth / Simulated VLM)
# =========================================================================

def load_ground_truth_for_invoice(filename: str) -> dict[str, Any] | None:
    """Load structured ground truth for a specific invoice.

    In production, this tier calls Qwen-VL or Donut for structured
    extraction. In dev mode, loads pre-generated metadata.

    Args:
        filename: e.g. "invoice_001.png" or "INV-2026-0001".
    """
    gt = _load_ground_truth()
    return gt.get(filename)


def list_sample_invoices() -> list[str]:
    """Return sorted list of available sample invoice filenames."""
    if not INVOICE_DIR.exists():
        return []
    return sorted(f.name for f in INVOICE_DIR.iterdir() if f.suffix == ".png")


def get_invoice_image_path(filename: str) -> Path:
    """Return full path to a sample invoice image."""
    return INVOICE_DIR / filename


# =========================================================================
# VALIDATION ENGINE
# =========================================================================

VALID_GST_RATES = {0, 5, 12, 18, 28}

EXPECTED_DISCOUNT_RANGES = {
    "ECOMM": (15, 25),
    "D2C": (5, 15),
    "OFFLINE": (5, 10),
    "GT": (8, 12),
    "MT": (10, 20),
}


def validate_invoice(invoice: dict[str, Any]) -> list[dict[str, str]]:
    """Apply deterministic validation rules to extracted invoice data.

    Three-stage validation pipeline:
      1. Arithmetic Consistency (tolerance +/-0.50)
      2. GST Rate Compliance
      3. Discount Sanity Check

    Returns:
        List of flag dicts with keys: field, issue, severity, detail.
    """
    flags: list[dict[str, str]] = []
    line_items = invoice.get("line_items", [])

    # --- 1. Arithmetic Consistency ---
    grand_total = invoice.get("grand_total", 0)
    computed_total = round(
        sum(item.get("line_total", 0) for item in line_items), 2
    )

    if line_items and grand_total > 0:
        diff = round(abs(grand_total - computed_total), 2)
        if diff > 0.50:
            flags.append({
                "field": "grand_total",
                "issue": "Arithmetic mismatch",
                "severity": "HIGH",
                "detail": (
                    f"Sum of line totals = Rs. {computed_total:,.2f} | "
                    f"Invoice grand total = Rs. {grand_total:,.2f} | "
                    f"Difference = Rs. {diff:,.2f}"
                ),
            })

    # Per-line-item taxable value check
    for i, item in enumerate(line_items, 1):
        qty = item.get("qty", 0)
        price = item.get("unit_price", 0)
        disc = item.get("discount_pct", 0)
        taxable = item.get("taxable_value", 0)

        if qty > 0 and price > 0 and taxable > 0:
            expected_taxable = round(qty * price * (1 - disc / 100), 2)
            line_diff = round(abs(taxable - expected_taxable), 2)
            if line_diff > 1.0:
                flags.append({
                    "field": f"line_item_{i}.taxable_value",
                    "issue": "Line item arithmetic mismatch",
                    "severity": "MEDIUM",
                    "detail": (
                        f"Expected taxable = Rs. {expected_taxable:,.2f}, "
                        f"extracted = Rs. {taxable:,.2f} "
                        f"(Qty={qty}, Price={price}, Disc={disc}%)"
                    ),
                })

    # --- 2. GST Rate Compliance ---
    for i, item in enumerate(line_items, 1):
        gst_rate = item.get("gst_rate", 0)
        if gst_rate and gst_rate not in VALID_GST_RATES:
            flags.append({
                "field": f"line_item_{i}.gst_rate",
                "issue": "Invalid GST rate",
                "severity": "HIGH",
                "detail": (
                    f"GST rate {gst_rate}% is not a standard Indian GST slab. "
                    f"Valid rates: {sorted(VALID_GST_RATES)}"
                ),
            })

        # GST amount consistency
        taxable = item.get("taxable_value", 0)
        cgst = item.get("cgst", 0)
        sgst = item.get("sgst", 0)
        igst = item.get("igst", 0)

        if taxable > 0 and gst_rate > 0:
            if igst > 0:
                expected_igst = round(taxable * gst_rate / 100, 2)
                if round(abs(igst - expected_igst), 2) > 1.0:
                    flags.append({
                        "field": f"line_item_{i}.igst",
                        "issue": "GST amount mismatch",
                        "severity": "HIGH",
                        "detail": (
                            f"IGST expected Rs. {expected_igst:,.2f} "
                            f"at {gst_rate}%, got Rs. {igst:,.2f}"
                        ),
                    })
            elif cgst > 0 or sgst > 0:
                expected_half = round(taxable * (gst_rate / 2) / 100, 2)
                if round(abs(cgst - expected_half), 2) > 1.0:
                    flags.append({
                        "field": f"line_item_{i}.cgst",
                        "issue": "CGST amount mismatch",
                        "severity": "MEDIUM",
                        "detail": (
                            f"CGST expected Rs. {expected_half:,.2f} "
                            f"at {gst_rate}%, got Rs. {cgst:,.2f}"
                        ),
                    })

    # --- 3. Discount Sanity Check ---
    channel = invoice.get("channel", "")
    expected_range = EXPECTED_DISCOUNT_RANGES.get(channel)
    if expected_range:
        for i, item in enumerate(line_items, 1):
            disc = item.get("discount_pct", 0)
            if disc > 0:
                lo, hi = expected_range
                if disc < lo - 2 or disc > hi + 2:
                    flags.append({
                        "field": f"line_item_{i}.discount_pct",
                        "issue": "Discount outside expected range",
                        "severity": "LOW",
                        "detail": (
                            f"Discount {disc:.1f}% outside expected "
                            f"{lo}-{hi}% range for {channel} channel"
                        ),
                    })

    return flags


def compute_validation_summary(
    flags: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute a summary of validation results."""
    high = sum(1 for f in flags if f["severity"] == "HIGH")
    medium = sum(1 for f in flags if f["severity"] == "MEDIUM")
    low = sum(1 for f in flags if f["severity"] == "LOW")

    return {
        "status": "FAIL" if high > 0 else ("WARNING" if medium > 0 else "PASS"),
        "total_flags": len(flags),
        "high": high,
        "medium": medium,
        "low": low,
        "has_critical_issues": high > 0,
    }
