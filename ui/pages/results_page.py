import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown("# 📊 Results")
    result = st.session_state.get("pipeline_result")
    if not result:
        st.info("Run the pipeline first from 🤖 Pipeline.")
        return

    # Support both local CodeAct and deep agent results
    if "deep_agent_output" in result:
        st.markdown("### Deep Agent Output")
        st.markdown(result["deep_agent_output"])
        return

    profile = result.get("profile", {})
    model_results = result.get("model_results", [])
    best = result.get("best_model", {})

    tab1, tab2, tab3 = st.tabs(["📊 Profile", "🤖 Models", "📄 Report"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{profile.get('n_rows', 0):,}")
        with c2:
            st.metric("Columns", profile.get("n_cols", 0))
        with c3:
            st.metric("Nulls", profile.get("total_nulls", 0))
        with c4:
            st.metric("Task", profile.get("task_type", "N/A"))

        st.markdown("**Target:** `{} ` | ** Columns:** {}".format(profile.get("target", "?"), ", ".join(profile.get("columns", [])[:10])))

        # Show profile json
        with st.expander("Raw profile.json"):
            st.json(profile)

    with tab2:
        if model_results:
            df = pd.DataFrame([{"Model": r["model"], **r["metrics"], "Time": r.get("time", 0)} for r in model_results])
            st.dataframe(df, use_container_width=True)

            if len(model_results) > 1:
                key = "f1" if profile.get("task_type") == "classification" else "r2" if profile.get("task_type") == "regression" else "silhouette"
                fig = go.Figure(go.Bar(
                    x=[r["model"] for r in model_results],
                    y=[r["metrics"].get(key, 0) for r in model_results],
                    marker_color=["#00FF88" if r["model"] == best.get("model") else "#00D4FF" for r in model_results],
                    text=[f"{r['metrics'].get(key, 0):.3f}" for r in model_results],
                    textposition="outside",
                ))
                fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e0e0e0"}, yaxis=dict(gridcolor="#1a2a3a"))
                st.plotly_chart(fig, use_container_width=True)

            if best.get("fi"):
                st.markdown("**Feature Importance (Best Model):**")
                fig = go.Figure(go.Bar(x=list(best["fi"].values()), y=list(best["fi"].keys()), orientation="h", marker_color="#00D4FF"))
                fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e0e0e0"}, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model results yet.")

    with tab3:
        txt = result.get("report_text", "")
        if txt:
            st.code(txt, language="text")
        else:
            st.info("No report found. Check outputs/reports/")
