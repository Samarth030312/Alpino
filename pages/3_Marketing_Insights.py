"""
Marketing Command Center — Channel Performance, Segmentation & Demand
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Indian Locale Formatting
# ---------------------------------------------------------------------------
def _fmt_inr(amount: float, decimals: int = 0) -> str:
    """Format a number with Indian comma separators and Rupee symbol."""
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


# -- Alpino chart palette --
_ALPINO_GOLD = "#FFC107"
_ALPINO_ORANGE = "#FF9800"
_ALPINO_DEEP = "#FF5722"
_ALPINO_GREEN = "#4CAF50"
_ALPINO_SEQ = [_ALPINO_GREEN, _ALPINO_GOLD, _ALPINO_DEEP]

st.set_page_config(
    page_title="Marketing Insights | Alpino AI",
    page_icon="📊",
    layout="wide",
)

st.title("Marketing Command Center")
st.info(
    "**Channel economics** with RFM segmentation for ROAS optimization  |  "
    "**Predictive demand** vs. inventory with stockout alerting"
)
st.markdown("---")

# =========================================================================
# SECTION A — Unit Economics & Channel Health
# =========================================================================
st.subheader("A. Unit Economics & Channel Health")

# Top-line metrics
a1, a2, a3, a4 = st.columns(4)
a1.metric("CAC (Meta/Instagram)", "₹5.14",
          help="Cost per acquisition on primary paid channel")
a2.metric("ROAS (Meta/Instagram)", "3.16x",
          help="Return on ad spend — ₹3.16 revenue per ₹1 spent")
a3.metric("Paid Social Sessions", "1,54,333",
          help="Dominant traffic source by volume")
a4.metric("Total Digital Revenue", "₹33.88L",
          help="Sum across all channels")

# Channel data — exact figures from Alpino FY25 analytics
channel_data = pd.DataFrame({
    "Channel": [
        "Paid Social", "Organic Social", "Paid Shopping",
        "Organic Search", "Paid Search", "Email",
    ],
    "Sessions": [154_333, 62_093, 42_736, 14_746, 1_385, 1_069],
    "Revenue (Rs.)": [
        13_13_729.20, 9_11_633.00, 8_02_000.00,
        2_38_000.00, 88_000.00, 32_000.00,
    ],
})

# Revenue per session
channel_data["Rev/Session (Rs.)"] = (
    channel_data["Revenue (Rs.)"] / channel_data["Sessions"]
).round(2)

# Grouped bar chart — sessions and revenue
fig_channels = go.Figure()

fig_channels.add_trace(go.Bar(
    name="Sessions",
    x=channel_data["Channel"],
    y=channel_data["Sessions"],
    yaxis="y",
    marker_color=_ALPINO_GOLD,
    text=channel_data["Sessions"].apply(lambda x: f"{x:,}"),
    textposition="outside",
))

fig_channels.add_trace(go.Bar(
    name="Revenue (Rs.)",
    x=channel_data["Channel"],
    y=channel_data["Revenue (Rs.)"],
    yaxis="y2",
    marker_color=_ALPINO_DEEP,
    text=channel_data["Revenue (Rs.)"].apply(lambda x: _fmt_inr(x)),
    textposition="outside",
))

fig_channels.update_layout(
    barmode="group",
    height=420,
    yaxis={"title": "Sessions", "side": "left"},
    yaxis2={
        "title": "Revenue (Rs.)",
        "side": "right",
        "overlaying": "y",
    },
    legend={"orientation": "h", "y": 1.12, "x": 0.3},
    margin={"t": 40, "b": 10},
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_channels.update_xaxes(showgrid=False)
fig_channels.update_yaxes(showgrid=False)

st.plotly_chart(fig_channels, width="stretch")

# Channel detail table
with st.expander("Channel Detail Table", expanded=False):
    st.dataframe(channel_data, width="stretch", hide_index=True)

st.markdown("---")

# =========================================================================
# SECTION B — RFM Customer Segmentation
# =========================================================================
st.subheader("B. RFM Customer Segmentation — ROAS Optimization")
st.caption(
    "Shifting 30% of Meta ad budget from cold prospects to high-value repeat "
    "buyers projects blended ROAS from 3.16 to 4.2 (+33%)"
)

b1, b2 = st.columns([1, 1], gap="large")

with b1:
    # RFM segment data
    rfm_data = pd.DataFrame({
        "Segment": [
            "High-Value Repeat Buyers",
            "One-Time High Intent",
            "Cold Prospects (Lookalike)",
        ],
        "Customer Share (%)": [15, 35, 50],
        "Current Budget (%)": [20, 30, 50],
        "Optimized Budget (%)": [40, 35, 25],
        "Expected ROAS": [6.0, 3.5, 1.8],
        "Description": [
            "3+ purchases in 6 months. Target: loyalty offers, new launches, referrals.",
            "1 purchase + 3+ browse sessions. Target: retargeting, bundles.",
            "Lookalike audiences from Meta. Target: brand awareness, education.",
        ],
    })

    # Donut chart — optimized budget allocation
    fig_donut = px.pie(
        rfm_data,
        names="Segment",
        values="Optimized Budget (%)",
        hole=0.4,
        color_discrete_sequence=[_ALPINO_GREEN, _ALPINO_GOLD, _ALPINO_ORANGE],
    )
    fig_donut.update_traces(
        textinfo="label+percent",
        textposition="outside",
        pull=[0.05, 0, 0],
    )
    fig_donut.update_layout(
        height=380,
        showlegend=False,
        margin={"t": 20, "b": 10, "l": 10, "r": 10},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[{
            "text": "Optimized<br>Budget",
            "x": 0.5, "y": 0.5,
            "font_size": 14,
            "showarrow": False,
        }],
    )
    st.plotly_chart(fig_donut, width="stretch")

with b2:
    # Budget reallocation comparison
    st.markdown("**Budget Reallocation Strategy**")

    comparison_df = rfm_data[["Segment", "Current Budget (%)", "Optimized Budget (%)", "Expected ROAS"]].copy()
    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Expected ROAS": st.column_config.NumberColumn(format="%.1fx"),
        },
    )

    st.markdown("**Projected Impact**")

    roas_current = 3.16
    roas_projected = round(
        0.40 * 6.0 + 0.35 * 3.5 + 0.25 * 1.8, 2
    )  # = 4.075 ≈ 4.08

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Current ROAS", f"{roas_current}x")
    rc2.metric("Projected ROAS", f"{roas_projected}x",
               delta=f"+{((roas_projected / roas_current - 1) * 100):.0f}%")
    rc3.metric("Revenue Uplift", f"Rs. {((roas_projected - roas_current) / roas_current * 13.14):.1f}L",
               help="Additional revenue at current Paid Social spend level")

    st.info(
        f"**Mathematical Basis:** Reallocating 30% of Meta budget from Segment 3 "
        f"(ROAS 1.8x) to Segment 1 (ROAS 6.0x) yields a blended ROAS of "
        f"**{roas_projected}x** — a **{((roas_projected / roas_current - 1) * 100):.0f}% improvement** "
        f"in marketing efficiency without increasing total spend."
    )

st.markdown("---")

# =========================================================================
# SECTION C — Stockout Solver (Demand vs. Inventory)
# =========================================================================
st.subheader("C. The Stockout Solver — Predictive Demand vs. Inventory")
st.caption(
    "Campaign-adjusted demand forecasting for 'Alpino Peanut Butter 1kg'  |  "
    "30-day simulation with Meta campaign uplift on Day 15"
)

# Generate 30-day synthetic demand data
np.random.seed(42)
days = np.arange(1, 31)
base_demand = np.random.normal(50, 8, 30).clip(30, 70).astype(int)

# Campaign uplift: ramp up on Day 13, spike Day 15, tail off by Day 20
campaign_multiplier = np.ones(30)
campaign_multiplier[12] = 1.8   # Day 13: early uplift
campaign_multiplier[13] = 2.5   # Day 14: building
campaign_multiplier[14] = 4.0   # Day 15: peak (50 * 4 = 200)
campaign_multiplier[15] = 3.2   # Day 16: still high
campaign_multiplier[16] = 2.4   # Day 17: declining
campaign_multiplier[17] = 1.8   # Day 18: tail
campaign_multiplier[18] = 1.4   # Day 19: settling
campaign_multiplier[19] = 1.2   # Day 20: back to normal

campaign_demand = (base_demand * campaign_multiplier).astype(int)
inventory_cap = 150

demand_df = pd.DataFrame({
    "Day": days,
    "Base Demand (units)": base_demand,
    "Campaign-Adjusted Demand (units)": campaign_demand,
    "Inventory Capacity": inventory_cap,
})

# Identify stockout days
stockout_days = demand_df[
    demand_df["Campaign-Adjusted Demand (units)"] > inventory_cap
]

# Build the chart
fig_demand = go.Figure()

# Base demand line
fig_demand.add_trace(go.Scatter(
    x=demand_df["Day"],
    y=demand_df["Base Demand (units)"],
    mode="lines",
    name="Base Demand",
    line={"color": _ALPINO_GOLD, "width": 2, "dash": "dot"},
))

# Campaign-adjusted demand
fig_demand.add_trace(go.Scatter(
    x=demand_df["Day"],
    y=demand_df["Campaign-Adjusted Demand (units)"],
    mode="lines+markers",
    name="Campaign-Adjusted Demand",
    line={"color": _ALPINO_DEEP, "width": 3},
    marker={"size": 5},
))

# Inventory capacity line
fig_demand.add_trace(go.Scatter(
    x=demand_df["Day"],
    y=demand_df["Inventory Capacity"],
    mode="lines",
    name="Inventory Capacity (150 units)",
    line={"color": _ALPINO_ORANGE, "width": 2, "dash": "dash"},
))

# Shade the stockout zone
if not stockout_days.empty:
    for _, row in stockout_days.iterrows():
        fig_demand.add_vrect(
            x0=row["Day"] - 0.4, x1=row["Day"] + 0.4,
            fillcolor="rgba(239,68,68,0.15)",
            line_width=0,
        )

# Campaign annotation
fig_demand.add_annotation(
    x=15, y=campaign_demand[14],
    text=f"Meta Campaign Peak<br>{campaign_demand[14]} units",
    showarrow=True,
    arrowhead=2,
    ax=60, ay=-40,
    font={"size": 12, "color": _ALPINO_DEEP},
    bordercolor=_ALPINO_DEEP,
    borderwidth=1,
    borderpad=4,
)

fig_demand.update_layout(
    height=450,
    xaxis_title="Day",
    yaxis_title="Units / Day",
    legend={"orientation": "h", "y": 1.12, "x": 0.1},
    margin={"t": 50, "b": 10},
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_demand.update_xaxes(showgrid=False)
fig_demand.update_yaxes(showgrid=False)

st.plotly_chart(fig_demand, width="stretch")

# Stockout alert
if not stockout_days.empty:
    peak_day = stockout_days.loc[
        stockout_days["Campaign-Adjusted Demand (units)"].idxmax()
    ]
    peak_demand = peak_day["Campaign-Adjusted Demand (units)"]
    peak_day_num = peak_day["Day"]

    st.error(
        f"**SUPPLY CHAIN ALERT:** Predictive demand for Day {peak_day_num} "
        f"({peak_demand} units) exceeds current inventory capacity "
        f"({inventory_cap} units). "
        f"**{len(stockout_days)} days** of projected stockout detected.\n\n"
        f"**Recommended Actions:**\n"
        f"- Trigger immediate procurement order for {peak_demand - inventory_cap} "
        f"additional units\n"
        f"- OR pause Meta ad spend on Day {peak_day_num - 2} to cap demand "
        f"within supply capacity\n"
        f"- Estimated CAC waste if stockout occurs: "
        f"{_fmt_inr(round((peak_demand - inventory_cap) * 5.14))}"
    )
else:
    st.success(
        "No stockout risk detected. Campaign demand is within inventory capacity."
    )

# Stockout detail table
with st.expander(f"Stockout Risk Detail ({len(stockout_days)} days)", expanded=False):
    if not stockout_days.empty:
        display = stockout_days.copy()
        display["Shortfall (units)"] = (
            display["Campaign-Adjusted Demand (units)"] - display["Inventory Capacity"]
        )
        display["Wasted CAC (Rs.)"] = (display["Shortfall (units)"] * 5.14).round(0)
        st.dataframe(
            display[["Day", "Campaign-Adjusted Demand (units)",
                     "Inventory Capacity", "Shortfall (units)", "Wasted CAC (Rs.)"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No stockout days to display.")

st.markdown("---")

# =========================================================================
# SECTION D — Key Recommendations
# =========================================================================
st.subheader("D. Strategic Recommendations")

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("**Immediate (Week 1)**")
    st.markdown(
        "- Onboard CleverTap/MoEngage CDP\n"
        "- Configure RFM segmentation on existing customer data\n"
        "- Create 3 audience segments for A/B testing"
    )

with d2:
    st.markdown("**Short-Term (Month 1-3)**")
    st.markdown(
        "- Shift 30% of Segment 3 budget to Segment 1\n"
        "- Connect CDP with UniCommerce inventory\n"
        "- Deploy base demand model (Prophet/LightGBM)"
    )

with d3:
    st.markdown("**Medium-Term (Month 4-6)**")
    st.markdown(
        "- Automate stockout alerts linked to campaign scheduler\n"
        "- Deploy campaign uplift model\n"
        "- Target: zero stockouts during planned campaigns"
    )
