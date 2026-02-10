import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Product Scenario Tool", layout="wide")

# ----------------------------
# Data
# ----------------------------
BASE_DATA = [
    {"Product":"Ready-made library sequencing","Contribution/unit (SEK)":9825,"Base units":4,"Price/unit (SEK)":14000,"Cost/unit (SEK)":4175},
    {"Product":"Human WGS (library + sequencing)","Contribution/unit (SEK)":1563,"Base units":180,"Price/unit (SEK)":4200,"Cost/unit (SEK)":2637},
    {"Product":"Bacterial transcriptome","Contribution/unit (SEK)":1200,"Base units":0,"Price/unit (SEK)":2200,"Cost/unit (SEK)":1000},
    {"Product":"Metagenome – Bulk +10% Price","Contribution/unit (SEK)":685,"Base units":0,"Price/unit (SEK)":990,"Cost/unit (SEK)":305},
    {"Product":"Metagenome – +20% Price","Contribution/unit (SEK)":670,"Base units":0,"Price/unit (SEK)":1080,"Cost/unit (SEK)":410},
    {"Product":"Metagenome – Bulk Pricing","Contribution/unit (SEK)":595,"Base units":0,"Price/unit (SEK)":900,"Cost/unit (SEK)":305},
    {"Product":"Metagenome – +10% Price","Contribution/unit (SEK)":580,"Base units":0,"Price/unit (SEK)":990,"Cost/unit (SEK)":410},
    {"Product":"Metagenome – Base (Optimized)","Contribution/unit (SEK)":490,"Base units":1061,"Price/unit (SEK)":900,"Cost/unit (SEK)":410},
    {"Product":"FFPE extraction","Contribution/unit (SEK)":300,"Base units":0,"Price/unit (SEK)":350,"Cost/unit (SEK)":50},
    {"Product":"Bacterial DNA extraction","Contribution/unit (SEK)":100,"Base units":0,"Price/unit (SEK)":150,"Cost/unit (SEK)":50},
    {"Product":"DNA extraction","Contribution/unit (SEK)":60,"Base units":0,"Price/unit (SEK)":100,"Cost/unit (SEK)":40},
]
BASE_DF = pd.DataFrame(BASE_DATA)

GROUPS = {
    "Sequencing": ["Ready-made library sequencing", "Human WGS (library + sequencing)"],
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

FIXED_COST_DEFAULT = 2_000_000

def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue (SEK)"] = df["Units"] * df["Price/unit (SEK)"]
    df["Variable Cost (SEK)"] = df["Units"] * df["Cost/unit (SEK)"]
    df["Contribution Profit (SEK)"] = df["Units"] * df["Contribution/unit (SEK)"]
    df["Margin %"] = (df["Contribution Profit (SEK)"] / df["Revenue (SEK)"]).fillna(0.0)
    return df

def scenario_units(multiplier: float) -> dict:
    d = {}
    for _, r in BASE_DF.iterrows():
        d[r["Product"]] = int(round(r["Base units"] * multiplier))
    return d

def clamp_nonneg_int(x) -> int:
    try:
        v = int(x)
        return max(0, v)
    except Exception:
        return 0

# ----------------------------
# Session state
# ----------------------------
if "units" not in st.session_state:
    st.session_state.units = {r["Product"]: int(r["Base units"]) for _, r in BASE_DF.iterrows()}

if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "Base"

if "fixed_cost" not in st.session_state:
    st.session_state.fixed_cost = FIXED_COST_DEFAULT

if "show_breakeven_view" not in st.session_state:
    st.session_state.show_breakeven_view = True

# ----------------------------
# Header
# ----------------------------
st.title("Product Scenario Tool")
st.caption("Adjust units per product and instantly see Revenue, Costs, Contribution, and Net Profit after fixed costs.")

left, right = st.columns([0.44, 0.56], gap="large")

# ----------------------------
# Left: Controls
# ----------------------------
with left:
    st.subheader("Controls")

    # Fixed cost threshold (breakeven)
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
        help="ON: shows fixed-cost coverage gauge and cumulative contribution chart. OFF: shows profit by product.",
    )

    st.divider()

    # Scenario tabs
    tab_base, tab_cons, tab_aggr = st.tabs(["Base", "Conservative", "Aggressive"])

    def apply_scenario(name: str, mult: float):
        st.session_state.units = scenario_units(mult)
        st.session_state.active_scenario = name

    with tab_base:
        st.write("Baseline units from your table.")
        if st.button("Apply Base", use_container_width=True):
            apply_scenario("Base", 1.0)
            st.rerun()

    with tab_cons:
        st.write("Example: 80% of baseline units (adjust in code if you want different logic).")
        if st.button("Apply Conservative (80%)", use_container_width=True):
            apply_scenario("Conservative", 0.8)
            st.rerun()

    with tab_aggr:
        st.write("Example: 120% of baseline units (adjust in code if you want different logic).")
        if st.button("Apply Aggressive (120%)", use_container_width=True):
            apply_scenario("Aggressive", 1.2)
            st.rerun()

    st.divider()

    # Search + group filter
    search = st.text_input("Search products", value="", placeholder="Type to filter…")
    group_choice = st.selectbox("Group", ["All"] + list(GROUPS.keys()), index=0)

    products = list(BASE_DF["Product"])
    if group_choice != "All":
        products = [p for p in products if p in GROUPS[group_choice]]
    if search.strip():
        s = search.strip().lower()
        products = [p for p in products if s in p.lower()]

    st.caption("Use sliders for fast changes, +/- for small tweaks, and the number box for exact values.")

    def slider_max(baseline: int, current: int) -> int:
        anchor = max(baseline, current, 50)
        return int(max(200, anchor * 2))

    def render_product_control(p: str):
        baseline = int(BASE_DF.loc[BASE_DF["Product"] == p, "Base units"].iloc[0])
        current = int(st.session_state.units.get(p, baseline))

        c1, c2, c3, c4 = st.columns([0.52, 0.18, 0.15, 0.15], vertical_alignment="center")
        with c1:
            new_slider = st.slider(
                p,
                min_value=0,
                max_value=slider_max(baseline, current),
                value=current,
                step=1,
                key=f"sl_{p}",
            )
        with c2:
            new_num = st.number_input(
                "Units",
                min_value=0,
                value=int(new_slider),
                step=1,
                key=f"num_{p}",
                label_visibility="collapsed",
            )
        with c3:
            if st.button("−", key=f"minus_{p}", use_container_width=True):
                new_num = max(0, int(new_num) - 1)
        with c4:
            if st.button("+", key=f"plus_{p}", use_container_width=True):
                new_num = int(new_num) + 1

        st.session_state.units[p] = clamp_nonneg_int(new_num)

    if group_choice == "All" and not search.strip():
        for g, plist in GROUPS.items():
            with st.expander(g, expanded=True):
                for p in plist:
                    render_product_control(p)
    else:
        if not products:
            st.info("No products match your filters.")
        else:
            for p in products:
                render_product_control(p)

    st.divider()

    # Save/load scenarios
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Reset to baseline", use_container_width=True):
            apply_scenario("Base", 1.0)
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
                st.session_state.active_scenario = "Custom"
                st.rerun()
            except Exception as e:
                st.error(f"Could not load scenario: {e}")

# ----------------------------
# Right: KPIs + Charts
# ----------------------------
with right:
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue (SEK)", f"{total_revenue:,.0f}")
    k2.metric("Variable Cost (SEK)", f"{total_var_cost:,.0f}")
    k3.metric("Contribution Profit (SEK)", f"{total_contrib:,.0f}", delta=f"Breakeven: {fixed_cost:,.0f}")
    k4.metric(
        "Net Profit after Fixed (SEK)",
        f"{net_profit:,.0f}",
        delta=("Above breakeven" if breakeven else "Below breakeven"),
    )

    st.caption(
        f"Scenario: **{st.session_state.active_scenario}** · "
        f"Net margin: **{net_margin:.1%}** · Contribution margin: **{contrib_margin:.1%}**"
    )

    st.divider()

    chart_df = df_calc.sort_values("Contribution Profit (SEK)", ascending=False).copy()

    if st.session_state.show_breakeven_view:
        # --- Gauge: fixed-cost coverage ---
        coverage_ratio = (total_contrib / fixed_cost) if fixed_cost > 0 else 0.0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=coverage_ratio * 100,
            number={'suffix': "%"},
            title={'text': "Fixed Cost Coverage"},
            gauge={
                'axis': {'range': [0, 150]},
                'threshold': {'line': {'width': 4}, 'value': 100},
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- Cumulative contribution (clean) + breakeven highlight (no dashed line) ---
        cum_df = chart_df[chart_df["Contribution Profit (SEK)"] > 0].copy()
        cum_df["Cumulative Contribution (SEK)"] = cum_df["Contribution Profit (SEK)"].cumsum()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Bar(
            x=cum_df["Product"],
            y=cum_df["Contribution Profit (SEK)"],
            name="Contribution Profit (SEK)"
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum_df["Product"],
            y=cum_df["Cumulative Contribution (SEK)"],
            name="Cumulative Contribution (SEK)",
            yaxis="y2"
        ))

        # Highlight breakeven product (first point where cumulative >= fixed cost)
        cross = cum_df.index[cum_df["Cumulative Contribution (SEK)"] >= fixed_cost].tolist()
        if cross:
            first_idx = cross[0]
            be_prod = cum_df.loc[first_idx, "Product"]
            be_val = float(cum_df.loc[first_idx, "Cumulative Contribution (SEK)"])

            # Draw a subtle vertical highlight at breakeven product
            xvals = list(cum_df["Product"])
            be_pos = xvals.index(be_prod)
            fig_cum.add_vrect(
                x0=be_pos - 0.5, x1=be_pos + 0.5,
                fillcolor="lightgreen", opacity=0.18,
                line_width=0
            )
            fig_cum.add_annotation(
                x=be_prod, y=be_val,
                xref="x", yref="y2",
                text=f"Breakeven reached at: {be_prod}",
                showarrow=True, arrowhead=2, ax=20, ay=-40
            )
        else:
            # If not reached, annotate shortfall
            shortfall = fixed_cost - total_contrib
            fig_cum.add_annotation(
                x=cum_df["Product"].iloc[-1] if len(cum_df) else 0,
                y=cum_df["Cumulative Contribution (SEK)"].iloc[-1] if len(cum_df) else 0,
                xref="x", yref="y2",
                text=f"Breakeven not reached. Shortfall: {shortfall:,.0f} SEK",
                showarrow=False,
                xanchor="right"
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
        # Simple profit by product
        fig_profit = px.bar(
            chart_df,
            x="Product",
            y="Contribution Profit (SEK)",
            title="Contribution Profit by Product",
        )
        fig_profit.update_layout(xaxis_tickangle=-35, height=440, margin=dict(l=20, r=20, t=60, b=120))
        st.plotly_chart(fig_profit, use_container_width=True)

    # Waterfall: Revenue -> Variable Cost -> Fixed Cost -> Net Profit
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue", "Variable Cost", "Fixed Cost", "Net Profit"],
        y=[total_revenue, -total_var_cost, -fixed_cost, net_profit],
        connector={"line": {"width": 1}},
    ))
    fig_wf.update_layout(
        title="Waterfall: Revenue → Variable Cost → Fixed Cost → Net Profit",
        height=360,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
        yaxis_title="SEK",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Pareto: Bars = contribution profit, Line = cumulative profit (SEK) - keep clean (no dashed line)
    pareto = chart_df[["Product", "Contribution Profit (SEK)"]].copy()
    pareto = pareto[pareto["Contribution Profit (SEK)"] > 0].sort_values("Contribution Profit (SEK)", ascending=False)

    if len(pareto) == 0:
        st.info("Pareto chart needs at least one product with positive contribution profit.")
    else:
        pareto["Cumulative Profit (SEK)"] = pareto["Contribution Profit (SEK)"].cumsum()

        fig_par = go.Figure()
        fig_par.add_trace(go.Bar(
            x=pareto["Product"],
            y=pareto["Contribution Profit (SEK)"],
            name="Contribution Profit (SEK)",
        ))
        fig_par.add_trace(go.Scatter(
            x=pareto["Product"],
            y=pareto["Cumulative Profit (SEK)"],
            name="Cumulative Profit (SEK)",
            yaxis="y2",
        ))

        # Highlight breakeven product (if reached)
        cross = pareto.index[pareto["Cumulative Profit (SEK)"] >= fixed_cost].tolist()
        if cross:
            first_idx = cross[0]
            be_prod = pareto.loc[first_idx, "Product"]
            be_val = float(pareto.loc[first_idx, "Cumulative Profit (SEK)"])
            xvals = list(pareto["Product"])
            be_pos = xvals.index(be_prod)
            fig_par.add_vrect(
                x0=be_pos - 0.5, x1=be_pos + 0.5,
                fillcolor="lightgreen", opacity=0.18,
                line_width=0
            )
            fig_par.add_annotation(
                x=be_prod,
                y=be_val,
                xref="x",
                yref="y2",
                text=f"Breakeven at: {be_prod}",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40,
            )

        fig_par.update_layout(
            title="Pareto: Profit Drivers (clean) + breakeven highlight",
            height=420,
            margin=dict(l=20, r=20, t=60, b=120),
            xaxis=dict(tickangle=-35),
            yaxis=dict(title="SEK (per product)"),
            yaxis2=dict(title="Cumulative SEK", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_par, use_container_width=True)

    with st.expander("Details table", expanded=False):
        show = df_calc[[
            "Product","Units","Price/unit (SEK)","Cost/unit (SEK)",
            "Revenue (SEK)","Variable Cost (SEK)","Contribution Profit (SEK)","Margin %"
        ]].copy()
        show["Margin %"] = (show["Margin %"] * 100).round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)
