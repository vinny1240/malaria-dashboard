from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model_core import Params, simulate, summarize, make_overlay, temperature_overlay

st.set_page_config(
    page_title="Malaria STELLA Dashboard",
    page_icon="🦟",
    layout="wide",
)

st.title("🦟 Malaria Transmission Model Dashboard")
st.caption(
    "Python/RK4 dashboard reconstructed from the STELLA midterm model. "
    "It is designed for interactive demonstration, not as an official STELLA publisher replacement."
)


def line_plot(df: pd.DataFrame, y_cols, title: str, color=None):
    if color:
        fig = px.line(df, x="time", y=y_cols, color=color, title=title)
    else:
        fig = px.line(df, x="time", y=y_cols, title=title)
    fig.update_layout(legend_title_text="Variable", height=420, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def overlay_plot(df: pd.DataFrame, y_col: str, title: str):
    hover = ["parameter_value"] if "parameter_value" in df.columns else None
    fig = px.line(df, x="time", y=y_col, color="run", title=title, hover_data=hover)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def metric_cards(summary: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S peak", f"{summary['S_peak']:.2f}")
    c2.metric("S time to peak", f"{summary['S_time_to_peak']:.3f} y")
    c3.metric("I peak", f"{summary['I_peak']:.2f}")
    c4.metric("I time to peak", f"{summary['I_time_to_peak']:.3f} y")


def base_param_controls(prefix: str = "") -> Params:
    with st.sidebar.expander("Baseline parameters", expanded=False):
        c = st.slider(f"{prefix}Bite rate c", 0.0, 1.5, 0.6, 0.01)
        z = st.slider(f"{prefix}Transmission probability z", 0.0, 1.0, 0.9, 0.01)
        i = st.slider(f"{prefix}Malaria death rate i", 0.0, 1.0, 0.5, 0.01)
        r = st.slider(f"{prefix}Recovery rate r", 0.0, 12.0, 0.25, 0.05)
        e = st.slider(f"{prefix}Mosquito birth rate e", 0.0, 1.0, 0.202, 0.001)
        g = st.slider(f"{prefix}Mosquito death rate g", 0.0, 5.0, 0.2, 0.01)
    return Params(c=c, z=z, i=i, r=r, e=e, g=g)


task = st.sidebar.radio(
    "Choose task",
    [
        "Task 1: Baseline",
        "Task 2.1: Policy sensitivity",
        "Task 2.2: Pesticide",
        "Task 2.3: Temperature",
        "Task 2.4: Parameter uncertainty",
    ],
)

dt = st.sidebar.selectbox(
    "DT",
    [1 / 64, 1 / 128, 1 / 32],
    index=0,
    format_func=lambda x: f"{x:.6f}",
)


if task == "Task 1: Baseline":
    st.header("Task 1｜Baseline model implementation")
    st.write("Four stocks: healthy villagers H, sick villagers S, healthy mosquitoes M, infected mosquitoes I.")

    params = base_param_controls()
    t_end = st.sidebar.slider("Time horizon", 5, 100, 100, 5)

    df = simulate(params=params, t_end=float(t_end), dt=dt)

    metric_cards(summarize(df))

    col1, col2 = st.columns(2)
    col1.plotly_chart(line_plot(df, ["H", "S"], "Human populations"), use_container_width=True)
    col2.plotly_chart(line_plot(df, ["M", "I"], "Mosquito populations"), use_container_width=True)

    st.info(
        "Take-home message: under the high-transmission and high-mortality baseline, "
        "the system tends toward host depletion rather than endemic equilibrium."
    )


elif task == "Task 2.1: Policy sensitivity":
    st.header("Task 2.1｜Policy sensitivity: r, g, c")
    st.write("Compare recovery rate r, mosquito death rate g, and bite rate c under the original model structure.")

    option = st.sidebar.selectbox("Sensitivity parameter", ["r", "g", "c"])
    t_end = st.sidebar.slider("Time horizon", 5, 100, 20, 5)

    if option == "r":
        values = np.linspace(0, 12, 10)
        label = "Recovery rate r"
    elif option == "g":
        values = np.linspace(0.2, 5.0, 10)
        label = "Mosquito death rate g"
    else:
        values = np.linspace(0.0, 1.0, 10)
        label = "Bite rate c"

    overlay_df, summary_df = make_overlay(option, values, t_end=float(t_end), dt=dt)

    st.subheader(f"Overlay sensitivity for {label}")
    col1, col2 = st.columns(2)
    col1.plotly_chart(overlay_plot(overlay_df, "S", "Sick villagers S"), use_container_width=True)
    col2.plotly_chart(overlay_plot(overlay_df, "I", "Infected mosquitoes I"), use_container_width=True)

    st.dataframe(summary_df.round(4), use_container_width=True)

    st.info(
        "Take-home message: vector-side interventions, especially reducing biting or increasing mosquito mortality, "
        "suppress transmission more directly than increasing recovery alone."
    )


elif task == "Task 2.2: Pesticide":
    st.header("Task 2.2｜Pesticide component")
    st.latex(r"P(t)=P_0 e^{-kt}, \quad g_{new}(t)=g+P(t)")

    params = Params()
    p0 = st.sidebar.slider("Initial pesticide P0", 0.0, 3.0, 1.5, 0.05)
    k = st.sidebar.slider("Decay constant k", 0.1, 3.0, 1.4, 0.05)
    t_end = st.sidebar.slider("Time horizon", 5, 100, 100, 5)

    base_df = simulate(params=params, t_end=float(t_end), dt=dt, pesticide_p0=0.0, pesticide_k=k)
    pest_df = simulate(params=params, t_end=float(t_end), dt=dt, pesticide_p0=p0, pesticide_k=k)

    base_df["scenario"] = "Baseline P0=0"
    pest_df["scenario"] = f"Pesticide P0={p0:.2f}"
    comp = pd.concat([base_df, pest_df], ignore_index=True)

    st.subheader("Baseline vs pesticide")

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.line(comp, x="time", y="M", color="scenario", title="Healthy mosquitoes M"),
        use_container_width=True,
    )
    col2.plotly_chart(
        px.line(comp, x="time", y="H", color="scenario", title="Healthy villagers H"),
        use_container_width=True,
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(
        px.line(comp, x="time", y="I", color="scenario", title="Infected mosquitoes I"),
        use_container_width=True,
    )
    col4.plotly_chart(
        px.line(comp, x="time", y="S", color="scenario", title="Sick villagers S"),
        use_container_width=True,
    )

    with st.expander("Show pesticide effect and dynamic mosquito death rate"):
        st.plotly_chart(
            px.line(
                pest_df,
                x="time",
                y=["Pesticide_Effect", "g_new"],
                title="Pesticide effect and dynamic mosquito death rate",
            ),
            use_container_width=True,
        )

    st.subheader("Summary")
    summary_table = pd.DataFrame(
        [
            {"scenario": "Baseline", **summarize(base_df)},
            {"scenario": "Pesticide", **summarize(pest_df)},
        ]
    )
    st.dataframe(summary_table.round(4), use_container_width=True)

    st.info(
        "Take-home message: pesticide strongly suppresses mosquitoes early, but under high transmission it does not "
        "substantially reduce the first human infection peak unless paired with contact reduction."
    )


elif task == "Task 2.3: Temperature":
    st.header("Task 2.3｜Temperature / global warming component")
    st.latex(r"Current\_Temp=22+Warming\_dC")
    st.latex(r"c(T)=\max(0,0.8-0.0125(T-26)^2), \quad z(T)=\max(0,0.99-0.01(T-25)^2)")

    mode = st.sidebar.radio("Mode", ["Single scenario", "10-run overlay"])
    t_end = st.sidebar.slider("Time horizon", 6, 100, 40, 2)

    if mode == "Single scenario":
        w = st.sidebar.slider("Warming_dC", 0.0, 10.0, 0.0, 0.1)
        df = simulate(t_end=float(t_end), dt=dt, warming_dc=w)

        metric_cards(summarize(df))

        col1, col2 = st.columns(2)
        col1.plotly_chart(
            line_plot(df, ["H", "S"], f"Human populations, T={22 + w:.1f}°C"),
            use_container_width=True,
        )
        col2.plotly_chart(
            line_plot(df, ["M", "I"], f"Mosquito populations, T={22 + w:.1f}°C"),
            use_container_width=True,
        )

        temps = np.linspace(15, 35, 300)
        thermal = pd.DataFrame(
            {
                "Temperature": temps,
                "c(T)": np.maximum(0, 0.8 - 0.0125 * (temps - 26) ** 2),
                "z(T)": np.maximum(0, 0.99 - 0.01 * (temps - 25) ** 2),
            }
        )
        st.plotly_chart(
            px.line(thermal, x="Temperature", y=["c(T)", "z(T)"], title="Thermal response curves"),
            use_container_width=True,
        )

    else:
        values = np.linspace(0, 10, 10)
        overlay_df, summary_df = temperature_overlay(values, t_end=float(t_end), dt=dt)
        overlay_df["parameter_value"] = overlay_df["Warming_dC"]

        col1, col2 = st.columns(2)
        col1.plotly_chart(overlay_plot(overlay_df, "S", "Sick villagers S"), use_container_width=True)
        col2.plotly_chart(overlay_plot(overlay_df, "I", "Infected mosquitoes I"), use_container_width=True)

        col3, col4 = st.columns(2)
        col3.plotly_chart(overlay_plot(overlay_df, "M", "Healthy mosquitoes M"), use_container_width=True)
        col4.plotly_chart(
            px.line(
                summary_df,
                x="Warming_dC",
                y=["S_peak", "I_peak"],
                markers=True,
                title="Peak values under warming scenarios",
            ),
            use_container_width=True,
        )

        st.dataframe(summary_df.round(4), use_container_width=True)

    st.info(
        "Take-home message: transmission is highest near the thermal optimum around 25–26°C; "
        "extreme warming can reduce transmission efficiency."
    )


elif task == "Task 2.4: Parameter uncertainty":
    st.header("Task 2.4｜Sensitivity analysis: z vs e")

    option = st.sidebar.selectbox("Sensitivity parameter", ["z", "e"])

    if option == "e":
        st.sidebar.caption("For e, the display focuses on the early 20-year outbreak window.")
        t_end = st.sidebar.slider("Time horizon", 5, 100, 20, 5)
    else:
        t_end = st.sidebar.slider("Time horizon", 5, 100, 20, 5)

    values = np.linspace(0.0, 1.0, 10)
    overlay_df, summary_df = make_overlay(option, values, t_end=float(t_end), dt=dt)

    st.subheader(f"Overlay sensitivity for {option}")

    col1, col2 = st.columns(2)

    fig_s = overlay_plot(overlay_df, "S", "Sick villagers S")
    if option == "e":
        fig_s.update_yaxes(range=[0, 100])
        fig_s.update_layout(
            title="Sick villagers S｜early outbreak window",
            yaxis_title="S"
        )

    col1.plotly_chart(fig_s, use_container_width=True)

    fig_i = overlay_plot(overlay_df, "I", "Infected mosquitoes I")
    col2.plotly_chart(fig_i, use_container_width=True)

    col3, col4 = st.columns(2)

    fig_peak = px.line(
        summary_df,
        x="parameter_value",
        y="S_peak",
        markers=True,
        title="S peak vs parameter"
    )
    if option == "e":
        fig_peak.update_yaxes(range=[0, 100])

    col3.plotly_chart(fig_peak, use_container_width=True)

    col4.plotly_chart(
        px.line(
            summary_df,
            x="parameter_value",
            y="S_time_to_peak",
            markers=True,
            title="S time to peak vs parameter"
        ),
        use_container_width=True
    )

    if option == "e":
        st.warning(
            "Display note: for e sensitivity, the dashboard focuses on the early human outbreak window "
            "and applies a host-extinction cutoff to avoid numerical reinfection caused by extremely small "
            "residual human values multiplied by explosive mosquito growth. This matches the report's "
            "interpretation that e mainly affects long-term mosquito abundance, not the early S curve."
        )

    st.dataframe(summary_df.round(4), use_container_width=True)

    st.info(
        "Take-home message: z controls early human outbreak size and timing, while e mainly affects long-term mosquito "
        "growth in this vector-abundant setting."
    )
