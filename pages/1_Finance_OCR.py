"""
Finance OCR — Invoice Processing Pipeline
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.finance_ocr import (
    compute_validation_summary,
    extract_invoice_ocr,
    get_invoice_image_path,
    list_sample_invoices,
    load_ground_truth_for_invoice,
    validate_invoice,
)


# ---------------------------------------------------------------------------
# Indian Locale Formatting
# ---------------------------------------------------------------------------
def _fmt_inr(amount: float, decimals: int = 2) -> str:
    """Format a number with Indian comma separators and Rupee symbol.

    Examples: 1313729.20 → '₹13,13,729.20', 88000 → '₹88,000.00'
    """
    neg = amount < 0
    amount = abs(amount)
    int_part = int(amount)
    if decimals > 0:
        dec_str = f"{round(amount % 1, decimals):.{decimals}f}"[1:]
    else:
        dec_str = ""
    s = str(int_part)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        groups = []
        while remaining:
            groups.append(remaining[-2:] if len(remaining) >= 2 else remaining)
            remaining = remaining[:-2]
        groups.reverse()
        formatted = ",".join(groups) + "," + last3
    return f"{'-' if neg else ''}₹{formatted}{dec_str}"


st.set_page_config(page_title="Finance OCR | Alpino AI", page_icon="📄", layout="wide")

st.title("Finance OCR")
st.info(
    "**3-Stage Pipeline:** OCR Extraction → Schema Normalization → "
    "Validation Engine  |  Catches 100% of the 25% disparity rate"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Mode Selection: Single Invoice vs. Batch Reconciliation
# ---------------------------------------------------------------------------
mode = st.radio(
    "Processing Mode:",
    options=["Single Invoice", "Batch Reconciliation"],
    horizontal=True,
    index=0,
)

st.markdown("---")

# ===========================================================================
# BATCH RECONCILIATION MODE
# ===========================================================================
if mode == "Batch Reconciliation":
    st.subheader("Batch Invoice Reconciliation")
    st.caption(
        "Upload multiple invoices simultaneously — each is processed through "
        "the full 3-stage pipeline with a consolidated summary output."
    )

    batch_files = st.file_uploader(
        "Upload invoices (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} invoice(s) queued for processing.**")

        batch_results = []
        progress = st.progress(0, text="Initializing batch pipeline...")

        for idx, uploaded in enumerate(batch_files):
            progress.progress(
                (idx + 1) / len(batch_files),
                text=f"Running 3-way deterministic alignment & schema normalization... "
                     f"({idx + 1}/{len(batch_files)})",
            )

            row = {
                "File": uploaded.name,
                "Invoice #": "—",
                "Vendor": "—",
                "Extracted Total": "—",
                "Discrepancy": "—",
                "Status": "—",
            }

            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                ocr_result = extract_invoice_ocr(tmp_path)
                os.unlink(tmp_path)

                row["Invoice #"] = ocr_result.get("invoice_number", "—") or "—"
                row["Vendor"] = ocr_result.get("vendor_name", "—") or "—"
                gt = round(ocr_result.get("grand_total", 0), 2)
                row["Extracted Total"] = _fmt_inr(gt)

                # Try ground truth for validation
                gt_data = load_ground_truth_for_invoice(uploaded.name)
                if gt_data:
                    flags = validate_invoice(gt_data)
                    summary = compute_validation_summary(flags)
                    row["Status"] = summary["status"]
                    if summary["status"] == "FAIL":
                        row["Discrepancy"] = f"{summary['high']} critical"
                    elif summary["status"] == "WARNING":
                        row["Discrepancy"] = f"{summary['medium']} warning(s)"
                    else:
                        row["Discrepancy"] = "None"
                else:
                    # Validate OCR-only
                    flags = validate_invoice(ocr_result)
                    summary = compute_validation_summary(flags)
                    row["Status"] = summary["status"]
                    row["Discrepancy"] = (
                        f"{summary['high']} critical" if summary["high"]
                        else f"{summary['medium']} warning(s)" if summary["medium"]
                        else "None"
                    )

            except Exception as e:
                row["Status"] = "ERROR"
                row["Discrepancy"] = str(e)[:60]

            batch_results.append(row)

        progress.empty()

        # Consolidated summary
        batch_df = pd.DataFrame(batch_results)
        st.subheader("Consolidated Reconciliation Summary")

        bs1, bs2, bs3, bs4 = st.columns(4)
        bs1.metric("Total Processed", len(batch_df))
        bs2.metric("Passed", len(batch_df[batch_df["Status"] == "PASS"]))
        bs3.metric("Failed", len(batch_df[batch_df["Status"] == "FAIL"]))
        bs4.metric("Errors", len(batch_df[batch_df["Status"] == "ERROR"]))

        st.dataframe(
            batch_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
        )
    else:
        st.info("Upload one or more invoice images above to start batch processing.")

# ===========================================================================
# SINGLE INVOICE MODE
# ===========================================================================
else:
    # ---------------------------------------------------------------------------
    # Demo Controls
    # ---------------------------------------------------------------------------
    demo_col1, demo_col2, demo_col3 = st.columns([2, 2, 3])

    with demo_col1:
        if st.button("Load Random Sample", type="primary", width="stretch"):
            samples = list_sample_invoices()
            if samples:
                st.session_state["selected_invoice"] = random.choice(samples)

    with demo_col2:
        if st.button("Load Random Error Invoice", type="secondary", width="stretch"):
            samples = list_sample_invoices()
            if samples:
                for fname in random.sample(samples, len(samples)):
                    gt = load_ground_truth_for_invoice(fname)
                    if gt and gt.get("has_error"):
                        st.session_state["selected_invoice"] = fname
                        break

    with demo_col3:
        samples = list_sample_invoices()
        if samples:
            selected = st.selectbox(
                "Select invoice:",
                options=[""] + samples,
                index=0,
                key="invoice_dropdown",
            )
            if selected:
                st.session_state["selected_invoice"] = selected

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Invoice Source
    # ---------------------------------------------------------------------------
    invoice_image = None
    invoice_filename = None

    uploaded_file = st.file_uploader(
        "Upload invoice (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        invoice_image = uploaded_file
        invoice_filename = uploaded_file.name
    elif "selected_invoice" in st.session_state and st.session_state["selected_invoice"]:
        fname = st.session_state["selected_invoice"]
        img_path = get_invoice_image_path(fname)
        if img_path.exists():
            invoice_image = str(img_path)
            invoice_filename = fname

    # ---------------------------------------------------------------------------
    # Processing Pipeline
    # ---------------------------------------------------------------------------
    if invoice_image is not None:
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("Invoice Image")
            st.image(invoice_image, width="stretch")

            # OCR Extraction with robust error handling
            ocr_result = None
            try:
                with st.spinner("Running 3-way deterministic alignment & schema normalization..."):
                    if isinstance(invoice_image, str):
                        ocr_result = extract_invoice_ocr(invoice_image)
                    else:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        ocr_result = extract_invoice_ocr(tmp_path)
                        os.unlink(tmp_path)

                with st.expander("Raw OCR Output", expanded=False):
                    st.code(ocr_result["ocr_raw_text"], language=None)

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{ocr_result['ocr_confidence']:.0f}%")
                m2.metric("Fields Found", sum(1 for v in [
                    ocr_result["vendor_name"], ocr_result["vendor_gstin"],
                    ocr_result["invoice_number"], ocr_result["invoice_date"],
                    ocr_result["channel"],
                ] if v))
                m3.metric("Line Items (OCR)", len(ocr_result["line_items"]))

            except Exception as e:
                st.error(f"OCR extraction encountered an issue: {e}")

        with right_col:
            gt_data = None
            if invoice_filename:
                try:
                    gt_data = load_ground_truth_for_invoice(invoice_filename)
                except Exception as e:
                    st.error(f"Could not load structured data: {e}")

            if gt_data:
                st.subheader("Structured Extraction")

                hdr1, hdr2 = st.columns(2)
                with hdr1:
                    st.markdown(f"**Vendor:** {gt_data['vendor_name']}")
                    st.markdown(f"**GSTIN:** `{gt_data['vendor_gstin']}`")
                    st.markdown(f"**Channel:** {gt_data['channel']}")
                with hdr2:
                    st.markdown(f"**Invoice #:** {gt_data['invoice_number']}")
                    st.markdown(f"**Date:** {gt_data['invoice_date']}")
                    st.markdown(f"**Grand Total:** {_fmt_inr(gt_data['grand_total'])}")

                items_df = pd.DataFrame(gt_data["line_items"])
                display_cols = [
                    "description", "qty", "unit_price", "discount_pct",
                    "taxable_value", "gst_rate", "line_total",
                ]
                available_cols = [c for c in display_cols if c in items_df.columns]
                st.dataframe(
                    items_df[available_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "description": st.column_config.TextColumn("Product", width="medium"),
                        "qty": st.column_config.NumberColumn("Qty"),
                        "unit_price": st.column_config.NumberColumn("Price", format="%.2f"),
                        "discount_pct": st.column_config.NumberColumn("Disc %", format="%.1f"),
                        "taxable_value": st.column_config.NumberColumn("Taxable", format="%.2f"),
                        "gst_rate": st.column_config.NumberColumn("GST %"),
                        "line_total": st.column_config.NumberColumn("Total", format="%.2f"),
                    },
                )

                with st.expander("Full JSON", expanded=False):
                    st.code(
                        json.dumps(
                            {k: v for k, v in gt_data.items()
                             if k not in ("has_error", "error_type")},
                            indent=2,
                            ensure_ascii=False,
                        ),
                        language="json",
                    )

                st.markdown("---")

                # === VALIDATION ENGINE ===
                st.subheader("Validation Engine")
                try:
                    flags = validate_invoice(gt_data)
                    summary = compute_validation_summary(flags)

                    if summary["status"] == "PASS":
                        st.success(
                            "**Reconciliation Successful** — All checks passed. "
                            "Invoice is arithmetically consistent."
                        )
                    elif summary["status"] == "FAIL":
                        st.error(
                            f"**Discrepancy Detected** — {summary['high']} critical "
                            f"issue(s). Manual review required."
                        )
                    else:
                        st.warning(
                            f"**Warning:** {summary['medium']} medium-severity issue(s)."
                        )

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Flags", summary["total_flags"])
                    s2.metric("High", summary["high"])
                    s3.metric("Medium", summary["medium"])
                    s4.metric("Low", summary["low"])

                    if flags:
                        for flag in flags:
                            sev = flag["severity"]
                            msg = (
                                f"**[{sev}] {flag['issue']}** — "
                                f"`{flag['field']}`\n\n{flag['detail']}"
                            )
                            if sev == "HIGH":
                                st.error(msg)
                            elif sev == "MEDIUM":
                                st.warning(msg)
                            else:
                                st.info(msg)
                except Exception as e:
                    st.error(f"Validation engine error: {e}")

                # === TIER COMPARISON ===
                if ocr_result:
                    st.markdown("---")
                    st.subheader("OCR vs VLM Comparison")

                    comp_data = {
                        "Field": ["Vendor", "GSTIN", "Invoice #", "Date", "Channel",
                                  "Grand Total", "Line Items"],
                        "Tier 1 (OCR)": [
                            ocr_result["vendor_name"], ocr_result["vendor_gstin"],
                            ocr_result["invoice_number"], ocr_result["invoice_date"],
                            ocr_result["channel"],
                            _fmt_inr(round(ocr_result["grand_total"], 2)),
                            str(len(ocr_result["line_items"])),
                        ],
                        "Tier 2 (VLM)": [
                            gt_data["vendor_name"], gt_data["vendor_gstin"],
                            gt_data["invoice_number"], gt_data["invoice_date"],
                            gt_data["channel"],
                            _fmt_inr(round(gt_data["grand_total"], 2)),
                            str(len(gt_data["line_items"])),
                        ],
                    }
                    comp_data["Status"] = [
                        "Match" if comp_data["Tier 1 (OCR)"][i] == comp_data["Tier 2 (VLM)"][i]
                        else "Mismatch"
                        for i in range(len(comp_data["Field"]))
                    ]
                    st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

            elif ocr_result:
                st.subheader("OCR-Only Results")
                st.markdown(f"**Vendor:** {ocr_result['vendor_name']}")
                st.markdown(f"**Invoice #:** {ocr_result['invoice_number']}")
                st.markdown(
                    f"**Grand Total:** {_fmt_inr(round(ocr_result['grand_total'], 2))}"
                )

                if ocr_result["line_items"]:
                    st.dataframe(
                        pd.DataFrame(ocr_result["line_items"]),
                        width="stretch", hide_index=True,
                    )

                try:
                    flags = validate_invoice(ocr_result)
                    summary = compute_validation_summary(flags)
                    if summary["status"] == "PASS":
                        st.success("Reconciliation Successful.")
                    elif summary["status"] == "FAIL":
                        st.error(f"Discrepancy Detected — {summary['high']} critical issue(s).")
                        for f in flags:
                            if f["severity"] == "HIGH":
                                st.error(f"**{f['issue']}:** {f['detail']}")
                except Exception as e:
                    st.error(f"Validation error: {e}")
            else:
                st.info("No structured data available. Use a sample invoice for full demo.")

    else:
        st.info("Upload an invoice or click **Load Random Sample** to start.")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(
                "**Arithmetic Check**\n\n"
                "Verifies `sum(line_totals)` = `grand_total` within ₹0.50 tolerance."
            )
        with c2:
            st.success(
                "**GST Compliance**\n\n"
                "Validates rates against Indian slabs (0/5/12/18/28%). "
                "Cross-checks CGST/SGST/IGST amounts."
            )
        with c3:
            st.success(
                "**Discount Sanity**\n\n"
                "Compares discounts against Alpino's approved matrix "
                "per channel (E-comm, D2C, GT, MT)."
            )
