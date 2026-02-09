import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Product Scenario Tool", layout="wide")

# ---- Data ----
DATA = [
    {"Product":"Ready-made library sequencing","Contribution/unit (SEK)":9825,"Base units":6881,"Price/unit (SEK)":14000,"Cost/unit (SEK)":4175},
    {"Product":"Human WGS (library + sequencing)","Contribution/unit (SEK)":1563,"Base units":0,"Price/unit (SEK)":4200,"Cost/unit (SEK)":2637},
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

def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue (SEK)"] = df["Base units"] * df["Price/unit (SEK)"]
    df["Cost (SEK)"] = df["Base units"] * df["Cost/unit (SEK)"]
    df["Profit (SEK)"] = df["Base units"] * df["Contribution/unit (SEK)"]
    df["Margin %"] = (df["Profit (SEK)"] / df["Revenue (SEK)"]).fillna(0.0)
    return df

# ---- State ----
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(DATA)

st.title("Product Scenario Tool")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Adjust base units")
    st.caption("Use sliders for quick scenario testing. All numbers update instantly.")

    df = st.session_state.df.copy()

    # UX: group products
    groups = {
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

    # Slider range strategy: default up to 2x baseline (min 1000 for usability)
    def slider_max(b):
        return max(1000, int(max(b * 2, 200)))

    for g, products in groups.items():
        with st.expander(g, expanded=True):
            for p in products:
                idx = df.index[df["Product"] == p][0]
                base = int(df.loc[idx, "Base units"])
                new_val = st.slider(
                    p,
                    min_value=0,
                    max_value=slider_max(base),
                    value=base,
                    step=1,
                    key=f"slider_{p}",
                )
                df.loc[idx, "Base units"] = new_val

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset to baseline", use_container_width=True):
            st.session_state.df = pd.DataFrame(DATA)
            st.rerun()
    with c2:
        # Save scenario
        scenario = df[["Product", "Base units"]].to_dict(orient="records")
        st.download_button(
            "Download scenario (.json)",
            data=json.dumps(scenario, indent=2),
            file_name="scenario_base_units.json",
            mime="application/json",
            use_container_width=True,
        )
    with c3:
        # Load scenario
        uploaded = st.file_uploader("Load scenario", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                m = {x["Product"]: int(x["Base units"]) for x in loaded}
                df2 = st.session_state.df.copy()
                df2["Base units"] = df2["Product"].map(lambda x: m.get(x, int(df2.loc[df2["Product"]==x, "Base units"].iloc[0])))
                st.session_state.df = df2
                st.rerun()
            except Exception as e:
                st.error(f"Could not load scenario: {e}")

    # Persist edited df
    st.session_state.df = df

with right:
    df_calc = compute(st.session_state.df)

    total_revenue = float(df_calc["Revenue (SEK)"].sum())
    total_cost = float(df_calc["Cost (SEK)"].sum())
    total_profit = float(df_calc["Profit (SEK)"].sum())
    overall_margin = (total_profit / total_revenue) if total_revenue else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue (SEK)", f"{total_revenue:,.0f}")
    k2.metric("Total Cost (SEK)", f"{total_cost:,.0f}")
    k3.metric("Total Profit (SEK)", f"{total_profit:,.0f}")
    k4.metric("Overall Margin", f"{overall_margin:.1%}")

    st.divider()

    # Charts
    chart_df = df_calc.sort_values("Profit (SEK)", ascending=False)

    fig_profit = px.bar(
        chart_df,
        x="Product",
        y="Profit (SEK)",
        title="Profit by Product",
    )
    fig_profit.update_layout(xaxis_tickangle=-35, height=420, margin=dict(l=20, r=20, t=60, b=120))
    st.plotly_chart(fig_profit, use_container_width=True)

    fig_rev_cost = px.bar(
        df_calc.melt(id_vars=["Product"], value_vars=["Revenue (SEK)", "Cost (SEK)"]),
        x="Product",
        y="value",
        color="variable",
        barmode="group",
        title="Revenue vs Cost by Product",
    )
    fig_rev_cost.update_layout(xaxis_tickangle=-35, height=420, margin=dict(l=20, r=20, t=60, b=120))
    st.plotly_chart(fig_rev_cost, use_container_width=True)

    with st.expander("Details table", expanded=False):
        show = df_calc[[
            "Product","Base units","Price/unit (SEK)","Cost/unit (SEK)",
            "Revenue (SEK)","Cost (SEK)","Profit (SEK)","Margin %"
        ]].copy()
        show["Margin %"] = (show["Margin %"]*100).round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)
