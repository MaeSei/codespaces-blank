import json
import math
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page / global CSS
# ----------------------------
st.set_page_config(page_title="Product Scenario Tool", layout="wide")

st.markdown(
    """
    <style>
      /* Make inputs and sliders less cramped on small screens */
      div[data-testid="stNumberInput"] input { min-width: 0px; }
      div[data-testid="stSlider"] { padding-top: 0.25rem; padding-bottom: 0.25rem; }
      .small-caption { font-size: 0.85rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Data (updated)
# ----------------------------
BASE_DATA = [
    {"Product":"Ready-made library sequencing","Contribution/unit (SEK)":9825,"Base units":6881,"Price/unit (SEK)":14000,"Cost/unit (SEK)":4175},
    {"Product":"Human WGS (library + sequencing)","Contribution/unit (SEK)":1563,"Base units":0,"Price/unit (SEK)":4200,"Cost/unit (SEK)":2637},
    {"Product":"Human WGS bulk pricing","Contribution/unit (SEK)":2403,"Base units":0,"Price/unit (SEK)":4200,"Cost/unit (SEK)":1797},
    {"Product":"Human Whole Exome Sequencing 150x","Contribution/unit (SEK)":1730,"Base units":0,"Price/unit (SEK)":2400,"Cost/unit (SEK)":670},
    {"Product":"Human Whole Exome Sequencing 250x","Contribution/unit (SEK)":1788,"Base units":0,"Price/unit (SEK)":2800,"Cost/unit (SEK)":1012},
    {"Product":"Bacterial transcriptome","Contribution/unit (SEK)":1200,"Base units":0,"Price/unit (SEK)":2200,"Cost/unit (SEK)":1000},
    {"Product":"Metagenome – Bulk +10% Price","Contribution/unit (SEK)":685,"Base units":0,"Price/unit (SEK)":990,"Cost/unit (SEK)":305},
    {"Product":"Metagenome – +20% Price","Contribution/unit (SEK)":670,"Base units":0,"Price/unit (SEK)":1080,"Cost/unit (SEK)":410},
    {"Product":"Metagenome – Bulk Pricing","Contribution/unit (SEK)":595,"Base units":0,"Price/unit (SEK)":900,"Cost/unit (SEK)":305},
    {"Product":"Metagenome – +10% Price","Contribution/unit (SEK)":580,"Base units":0,"Price/unit (SEK)":990,"Cost/unit (SEK)":410},
    {"Product":"Metagenome – Base (Optimized)","Contribution/unit (SEK)":490,"Base units":991,"Price/unit (SEK)":900,"Cost/unit (SEK)":410},
    {"Product":"FFPE extraction","Contribution/unit (SEK)":300,"Base units":0,"Price/unit (SEK)":350,"Cost/unit (SEK)":50},
    {"Product":"Bacterial DNA extraction","Contribution/unit (SEK)":100,"Base units":0,"Price/unit (SEK)":150,"Cost/unit (SEK)":50},
    {"Product":"DNA extraction","Contribution/unit (SEK)":60,"Base units":0,"Price/unit (SEK)":100,"Cost/unit (SEK)":40},
]
BASE_DF = pd.DataFrame(BASE_DATA)

GROUPS = {
    "Human sequencing": [
        "Ready-made library sequencing",
        "Human WGS (library + sequencing)",
        "Human WGS bulk pricing",
        "Human Whole Exome Sequencing 150x",
        "Human Whole Exome Sequencing 250x",
    ],
    "Transcriptomics": ["Bacterial transcriptome"],
    "Metagenomics": [
        "Metagenome – Bulk +10% Price",
        "Metagenome – +20% Price",
        "Metagenome – Bulk Pricing",
        "Metagenome – +10% Price",
        "Metagenome – Base (Optimized)",
    ],
    "Extraction": ["FFPE extraction", "Bacterial DNA extraction", "DNA extraction"],
}

FIXED_COST_DEFAULT = 4_800_000

def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue (SEK)"] = df["Units"] * df["Price/unit (SEK)"]
    df["Variable Cost (SEK)"] = df["Units"] * df["Cost/unit (SEK)"]
    df["Contribution Profit (SEK)"] = df["Units"] * df["Contribution/unit (SEK)"]
    df["Margin %"] = (df["Contribution Profit (SEK)"] / df["Revenue (SEK)"]).fillna(0.0)
    return df

def safe_key(p: str) -> str:
    return f"p_{abs(hash(p))}"

def clamp_nonneg_int(x) -> int:
    try:
        return max(0, int(x))
    except Exception:
        return 0

def short_label(s: str, n: int = 22) -> str:
    return s if len(s) <= n else (s[: n - 1] + "…")

# ----------------------------
# Session state
# ----------------------------
if "units" not in st.session_state:
    st.session_state.units = {r["Product"]: int(r["Base units"]) for _, r in BASE_DF.iterrows()}

if "fixed_cost" not in st.session_state:
    st.session_state.fixed_cost = FIXED_COST_DEFAULT

if "show_breakeven_view" not in st.session_state:
    st.session_state.show_breakeven_view = True

if "compact_mode" not in st.session_state:
    st.session_state.compact_mode = False  # user can toggle

# ----------------------------
# Slider/number sync callbacks
# ----------------------------
def on_slider_change(product: str, slider_k: str, num_k: str):
    v = clamp_nonneg_int(st.session_state[slider_k])
    st.session_state.units[product] = v
    st.session_state[num_k] = v

def on_num_change(product: str, slider_k: str, num_k: str):
    v = clamp_nonneg_int(st.session_state[num_k])
    st.session_state.units[product] = v
    st.session_state[slider_k] = v

# ----------------------------
# Header
# ----------------------------
st.title("Product Scenario Tool")
st.caption("Adjust units per product and instantly see Revenue, Costs, Contribution, and Net Profit after fixed costs.")

# A user-controlled compact mode (works on phone reliably)
st.session_state.compact_mode = st.toggle(
    "Compact/mobile mode",
    value=bool(st.session_state.compact_mode),
    help="Optimizes layout for phones: stacked layout, shorter labels, top-N charts.",
)

# If compact mode, constrain layout visually
if st.session_state.compact_mode:
    st.markdown(
        """
        <style>
          section.main > div { max-width: 820px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Controls renderer
# ----------------------------
def render_controls():
    st.subheader("Controls")

    st.session_state.fixed_cost = st.number_input(
        "Fixed costs (SEK)",
        min_value=0,
        value=int(st.session_state.fixed_cost),
        step=50_000,
        help="Breakeven threshold: Contribution Profit must exceed Fixed costs for Net Profit to be positive.",
    )

    st.session_state.show_breakeven_view = st.toggle(
        "Show breakeven view (Gauge + Cumulative)",
        value=bool(st.session_state.show_breakeven_view),
        help="ON: shows fixed-cost coverage gauge + cumulative contribution chart. OFF: shows profit by product.",
    )

    st.divider()

    search = st.text_input("Search products", value="", placeholder="Type to filter…")
    group_choice = st.selectbox("Group", ["All"] + list(GROUPS.keys()), index=0)

    products = list(BASE_DF["Product"])
    if group_choice != "All":
        products = [p for p in products if p in GROUPS[group_choice]]
    if search.strip():
        s = search.strip().lower()
        products = [p for p in products if s in p.lower()]

    st.markdown('<div class="small-caption">Use the slider for quick changes; use the number box for exact values (synced).</div>', unsafe_allow_html=True)

    def slider_max(baseline: int, current: int) -> int:
        anchor = max(baseline, current, 50)
        return int(max(200, anchor * 2))

    def render_product_control(p: str):
        baseline = int(BASE_DF.loc[BASE_DF["Product"] == p, "Base units"].iloc[0])
        current = int(st.session_state.units.get(p, baseline))

        sk = f"sl_{safe_key(p)}"
        nk = f"num_{safe_key(p)}"

        if sk not in st.session_state:
            st.session_state[sk] = current
        if nk not in st.session_state:
            st.session_state[nk] = current

        if st.session_state.compact_mode:
            # stacked (better on phone)
            st.slider(
                p,
                min_value=0,
                max_value=slider_max(baseline, current),
                value=int(st.session_state[sk]),
                step=1,
                key=sk,
                on_change=on_slider_change,
                kwargs={"product": p, "slider_k": sk, "num_k": nk},
            )
            st.number_input(
                f"Units – {p}",
                min_value=0,
                value=int(st.session_state[nk]),
                step=1,
                key=nk,
                on_change=on_num_change,
                kwargs={"product": p, "slider_k": sk, "num_k": nk},
            )
        else:
            # desktop two-column control
            c1, c2 = st.columns([0.72, 0.28], vertical_alignment="center")
            with c1:
                st.slider(
                    p,
                    min_value=0,
                    max_value=slider_max(baseline, current),
                    value=int(st.session_state[sk]),
                    step=1,
                    key=sk,
                    on_change=on_slider_change,
                    kwargs={"product": p, "slider_k": sk, "num_k": nk},
                )
            with c2:
                st.number_input(
                    "Units",
                    min_value=0,
                    value=int(st.session_state[nk]),
                    step=1,
                    key=nk,
                    label_visibility="collapsed",
                    on_change=on_num_change,
                    kwargs={"product": p, "slider_k": sk, "num_k": nk},
                )

        st.session_state.units[p] = clamp_nonneg_int(st.session_state[sk])

    if group_choice == "All" and not search.strip():
        for g, plist in GROUPS.items():
            with st.expander(g, expanded=not st.session_state.compact_mode):
                for p in plist:
                    render_product_control(p)
    else:
        if not products:
            st.info("No products match your filters.")
        else:
            for p in products:
                render_product_control(p)

    st.divider()

    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Reset to baseline", use_container_width=True):
            st.session_state.units = {r["Product"]: int(r["Base units"]) for _, r in BASE_DF.iterrows()}
            for p, v in st.session_state.units.items():
                st.session_state[f"sl_{safe_key(p)}"] = v
                st.session_state[f"num_{safe_key(p)}"] = v
            st.rerun()

    with cB:
        payload = [{"Product": p, "Units": int(u)} for p, u in st.session_state.units.items()]
        st.download_button(
            "Download scenario",
            data=json.dumps(payload, indent=2),
            file_name="scenario_units.json",
            mime="application/json",
            use_container_width=True,
        )

    with cC:
        uploaded = st.file_uploader("Load scenario", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                m = {x["Product"]: int(x["Units"]) for x in loaded}
                for p in list(st.session_state.units.keys()):
                    if p in m:
                        st.session_state.units[p] = clamp_nonneg_int(m[p])
                for p, v in st.session_state.units.items():
                    st.session_state[f"sl_{safe_key(p)}"] = v
                    st.session_state[f"num_{safe_key(p)}"] = v
                st.rerun()
            except Exception as e:
                st.error(f"Could not load scenario: {e}")

# ----------------------------
# Charts/KPIs renderer
# ----------------------------
def render_results():
    fixed_cost = float(st.session_state.fixed_cost)

    df = BASE_DF.copy()
    df["Units"] = df["Product"].map(lambda p: int(st.session_state.units.get(p, 0)))
    df_calc = compute(df)

    total_revenue = float(df_calc["Revenue (SEK)"].sum())
    total_var_cost = float(df_calc["Variable Cost (SEK)"].sum())
    total_contrib = float(df_calc["Contribution Profit (SEK)"].sum())
    net_profit = total_contrib - fixed_cost
    contrib_margin = (total_contrib / total_revenue) if total_revenue else 0.0
    net_margin = (net_profit / total_revenue) if total_revenue else 0.0
    breakeven = total_contrib >= fixed_cost
    shortfall = max(0.0, fixed_cost - total_contrib)

    # KPIs: fewer columns on mobile
    if st.session_state.compact_mode:
        st.metric("Total Revenue (SEK)", f"{total_revenue:,.0f}")
        st.metric("Variable Cost (SEK)", f"{total_var_cost:,.0f}")
        st.metric("Contribution Profit (SEK)", f"{total_contrib:,.0f}")
        st.metric("Net Profit after Fixed (SEK)", f"{net_profit:,.0f}", delta=("Above breakeven" if breakeven else "Below breakeven"))
        st.caption(f"Net margin: **{net_margin:.1%}** · Contribution margin: **{contrib_margin:.1%}**")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Revenue (SEK)", f"{total_revenue:,.0f}")
        k2.metric("Variable Cost (SEK)", f"{total_var_cost:,.0f}")
        k3.metric("Contribution Profit (SEK)", f"{total_contrib:,.0f}", delta=f"Breakeven: {fixed_cost:,.0f}")
        k4.metric("Net Profit after Fixed (SEK)", f"{net_profit:,.0f}", delta=("Above breakeven" if breakeven else "Below breakeven"))
        st.caption(f"Net margin: **{net_margin:.1%}** · Contribution margin: **{contrib_margin:.1%}**")

    st.divider()

    chart_df = df_calc.sort_values("Contribution Profit (SEK)", ascending=False).copy()

    # In compact mode: show top N to avoid unreadable x-axis labels
    top_n = 8 if st.session_state.compact_mode else len(chart_df)
    chart_df_n = chart_df.head(top_n).copy()
    chart_df_n["Label"] = chart_df_n["Product"].apply(lambda s: short_label(s, 22 if st.session_state.compact_mode else 40))

    if st.session_state.show_breakeven_view:
        coverage_ratio = (total_contrib / fixed_cost) if fixed_cost > 0 else 0.0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=coverage_ratio * 100,
            number={"suffix": "%"},
            title={"text": "Fixed Cost Coverage"},
            gauge={"axis": {"range": [0, 150]}, "threshold": {"line": {"width": 4}, "value": 100}},
        ))
        fig_gauge.update_layout(height=210 if st.session_state.compact_mode else 220, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        cum_df = chart_df[chart_df["Contribution Profit (SEK)"] > 0].copy()
        cum_df["Cumulative Contribution (SEK)"] = cum_df["Contribution Profit (SEK)"].cumsum()

        # For compact: use horizontal bars + cumulative line (still readable)
        if st.session_state.compact_mode:
            cum_df = cum_df.head(10).copy()
            cum_df["Label"] = cum_df["Product"].apply(lambda s: short_label(s, 22))
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Bar(
                y=cum_df["Label"],
                x=cum_df["Contribution Profit (SEK)"],
                name="Contribution Profit (SEK)",
                orientation="h"
            ))
            fig_cum.add_trace(go.Scatter(
                y=cum_df["Label"],
                x=cum_df["Cumulative Contribution (SEK)"],
                name="Cumulative Contribution (SEK)",
                xaxis="x2"
            ))
            fig_cum.update_layout(
                title="Cumulative Contribution (Top 10)",
                height=520,
                margin=dict(l=20, r=20, t=60, b=40),
                xaxis=dict(title="SEK (per product)"),
                xaxis2=dict(title="Cumulative SEK", overlaying="x", side="top"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        else:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Bar(
                x=cum_df["Product"],
                y=cum_df["Contribution Profit (SEK)"],
                name="Contribution Profit (SEK)",
            ))
            fig_cum.add_trace(go.Scatter(
                x=cum_df["Product"],
                y=cum_df["Cumulative Contribution (SEK)"],
                name="Cumulative Contribution (SEK)",
                yaxis="y2",
            ))
            cross = cum_df.index[cum_df["Cumulative Contribution (SEK)"] >= fixed_cost].tolist()
            if cross:
                first_idx = cross[0]
                be_prod = cum_df.loc[first_idx, "Product"]
                be_val = float(cum_df.loc[first_idx, "Cumulative Contribution (SEK)"])
                xvals = list(cum_df["Product"])
                be_pos = xvals.index(be_prod)
                fig_cum.add_vrect(x0=be_pos - 0.5, x1=be_pos + 0.5, fillcolor="lightgreen", opacity=0.18, line_width=0)
                fig_cum.add_annotation(
                    x=be_prod, y=be_val, xref="x", yref="y2",
                    text=f"Breakeven reached at: {be_prod}",
                    showarrow=True, arrowhead=2, ax=20, ay=-40
                )
            fig_cum.update_layout(
                title="Cumulative Contribution (clean breakeven view)",
                height=440,
                margin=dict(l=20, r=20, t=60, b=120),
                xaxis=dict(tickangle=-35),
                yaxis=dict(title="SEK (per product)"),
                yaxis2=dict(title="Cumulative SEK", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cum, use_container_width=True)
    else:
        # Profit by product (compact uses horizontal top-N)
        if st.session_state.compact_mode:
            fig_profit = px.bar(
                chart_df_n.sort_values("Contribution Profit (SEK)", ascending=True),
                x="Contribution Profit (SEK)",
                y="Label",
                orientation="h",
                title=f"Contribution Profit by Product (Top {top_n})",
            )
            fig_profit.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=40))
        else:
            fig_profit = px.bar(chart_df, x="Product", y="Contribution Profit (SEK)", title="Contribution Profit by Product")
            fig_profit.update_layout(xaxis_tickangle=-35, height=440, margin=dict(l=20, r=20, t=60, b=120))
        st.plotly_chart(fig_profit, use_container_width=True)

    # Waterfall: still fine on mobile but reduce height
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue", "Variable Cost", "Fixed Cost", "Net Profit"],
        y=[total_revenue, -total_var_cost, -fixed_cost, net_profit],
        connector={"line": {"width": 1}},
    ))
    fig_wf.update_layout(
        title="Waterfall: Revenue → Variable Cost → Fixed Cost → Net Profit",
        height=320 if st.session_state.compact_mode else 360,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
        yaxis_title="SEK",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Units needed table
    with st.expander("Units needed to reach breakeven (by product)", expanded=not st.session_state.compact_mode):
        st.metric("Breakeven shortfall (SEK)", f"{shortfall:,.0f}")

        units_needed_df = BASE_DF[["Product", "Contribution/unit (SEK)"]].copy()

        def units_needed(contrib_per_unit: float, shortfall_: float):
            if shortfall_ <= 0:
                return 0
            if contrib_per_unit <= 0:
                return None
            return int(math.ceil(shortfall_ / contrib_per_unit))

        units_needed_df["Units needed to breakeven"] = units_needed_df["Contribution/unit (SEK)"].apply(
            lambda c: units_needed(float(c), shortfall)
        )

        units_needed_df["Units needed (sort)"] = units_needed_df["Units needed to breakeven"].apply(
            lambda x: x if x is not None else 10**18
        )
        show_units = units_needed_df.sort_values("Units needed (sort)", ascending=True).drop(columns=["Units needed (sort)"])

        # Compact: show fewer columns first
        if st.session_state.compact_mode:
            show_units2 = show_units.copy()
            show_units2["Product"] = show_units2["Product"].apply(lambda s: short_label(s, 26))
            st.dataframe(show_units2, use_container_width=True, hide_index=True)
        else:
            st.dataframe(show_units, use_container_width=True, hide_index=True)

        csv = show_units.to_csv(index=False).encode("utf-8")
        st.download_button("Download table (CSV)", data=csv, file_name="units_needed_to_breakeven.csv", mime="text/csv")

    # Pareto: on mobile show top 10 only
    pareto = chart_df[["Product", "Contribution Profit (SEK)"]].copy()
    pareto = pareto[pareto["Contribution Profit (SEK)"] > 0].sort_values("Contribution Profit (SEK)", ascending=False)
    if len(pareto):
        if st.session_state.compact_mode:
            pareto = pareto.head(10).copy()
            pareto["Label"] = pareto["Product"].apply(lambda s: short_label(s, 22))
            pareto["Cumulative Profit (SEK)"] = pareto["Contribution Profit (SEK)"].cumsum()
            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(y=pareto["Label"], x=pareto["Contribution Profit (SEK)"], name="Contribution Profit (SEK)", orientation="h"))
            fig_par.add_trace(go.Scatter(y=pareto["Label"], x=pareto["Cumulative Profit (SEK)"], name="Cumulative Profit (SEK)", xaxis="x2"))
            fig_par.update_layout(
                title="Pareto (Top 10)",
                height=520,
                margin=dict(l=20, r=20, t=60, b=40),
                xaxis=dict(title="SEK (per product)"),
                xaxis2=dict(title="Cumulative SEK", overlaying="x", side="top"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_par, use_container_width=True)
        else:
            pareto["Cumulative Profit (SEK)"] = pareto["Contribution Profit (SEK)"].cumsum()
            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=pareto["Product"], y=pareto["Contribution Profit (SEK)"], name="Contribution Profit (SEK)"))
            fig_par.add_trace(go.Scatter(x=pareto["Product"], y=pareto["Cumulative Profit (SEK)"], name="Cumulative Profit (SEK)", yaxis="y2"))
            fig_par.update_layout(
                title="Pareto: Profit Drivers",
                height=420,
                margin=dict(l=20, r=20, t=60, b=120),
                xaxis=dict(tickangle=-35),
                yaxis=dict(title="SEK (per product)"),
                yaxis2=dict(title="Cumulative SEK", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_par, use_container_width=True)
    else:
        st.info("Pareto chart needs at least one product with positive contribution profit.")

    with st.expander("Details table", expanded=False):
        show = df_calc[[
            "Product","Units","Price/unit (SEK)","Cost/unit (SEK)",
            "Revenue (SEK)","Variable Cost (SEK)","Contribution Profit (SEK)","Margin %"
        ]].copy()
        show["Margin %"] = (show["Margin %"] * 100).round(1)
        if st.session_state.compact_mode:
            show["Product"] = show["Product"].apply(lambda s: short_label(s, 26))
        st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------------------
# Layout: desktop (two columns) vs compact (stacked)
# ----------------------------
if st.session_state.compact_mode:
    # stacked: controls in an expander, then results
    with st.expander("Controls", expanded=True):
        render_controls()
    render_results()
else:
    left, right = st.columns([0.44, 0.56], gap="large")
    with left:
        render_controls()
    with right:
        render_results()
