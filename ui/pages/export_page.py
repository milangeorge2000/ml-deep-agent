import streamlit as st
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown("# 📥 Export")
    result = st.session_state.get("pipeline_result", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📄 Report")
        txt = result.get("report_text", "")
        if txt:
            st.download_button("Download report.txt", txt, "ml_report.txt", "text/plain", use_container_width=True)
        else:
            reports = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "reports", "*.txt"))
            if reports:
                latest = max(reports, key=os.path.getmtime)
                st.download_button("Download latest report", open(latest, encoding="utf-8").read(), os.path.basename(latest), "text/plain", use_container_width=True)
            else:
                st.info("No report yet.")

    with c2:
        st.markdown("### 🤖 Model")
        model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.joblib")
        if os.path.exists(model_path):
            st.download_button("Download best_model.joblib", open(model_path, "rb").read(), "best_model.joblib", "application/octet-stream", use_container_width=True)
        else:
            st.info("No model saved yet.")

    with c3:
        st.markdown("### 📊 Data")
        feat_path = os.path.join(PROJECT_ROOT, "outputs", "code", "features.csv")
        if os.path.exists(feat_path):
            st.download_button("Download features.csv", open(feat_path, "rb").read(), "features.csv", "text/csv", use_container_width=True)
        else:
            st.info("No transformed data yet.")

    st.markdown("---")
    st.markdown("### 💻 All Generated Code")
    code_dir = os.path.join(PROJECT_ROOT, "outputs", "code")
    if os.path.exists(code_dir):
        for f in sorted(glob.glob(os.path.join(code_dir, "*"))):
            rel = os.path.relpath(f, PROJECT_ROOT)
            size = os.path.getsize(f)
            st.markdown(f"- `{rel}` ({size} bytes)")
