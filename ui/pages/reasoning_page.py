import streamlit as st
import os
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown("# 📝 Reasoning & Evidence")
    st.markdown("Every imputation, transform, and model choice was made by **code that documents its reasoning**.")

    result = st.session_state.get("pipeline_result", {})
    if not result:
        st.info("Run the pipeline first.")
        return

    # Show literature & reasoning from code files
    code_dir = os.path.join(PROJECT_ROOT, "outputs", "code")

    # Profile reasoning
    if result.get("profile"):
        st.markdown("### 1. Data Understanding (01_profile.py)")
        st.markdown(f"<div class='reasoning-card'>Profile: {result['profile'].get('n_rows')} rows, {result['profile'].get('n_cols')} cols, target={result['profile'].get('target')}, task={result['profile'].get('task_type')}</div>", unsafe_allow_html=True)

    # Show code-level reasoning for each step
    for fname in ["03_impute.py", "04_features.py", "05_train.py"]:
        fpath = os.path.join(code_dir, fname)
        if os.path.exists(fpath):
            with st.expander(f"📖 {fname} — inline reasoning", expanded=(fname == "03_impute.py")):
                content = open(fpath, encoding="utf-8").read()
                # Extract comments and reasoning prints
                st.code(content[:4000], language="python")

    # Literature
    st.markdown("### 📚 Literature Cited")
    st.markdown("""
<div class='literature-card'>
<b>Imputation</b><br>
• Troyanskaya et al. 2001 — KNN for &lt;20% missing (Bioinformatics)<br>
• van Buuren & Groothuis-Oudshoorn 2011 — MICE for 20-60% MAR data<br>
• Little & Rubin 2019 — drop when &gt;60% missing
</div>
<div class='literature-card'>
<b>Transforms & Scaling</b><br>
• Box & Cox 1964 — power transforms for skewed data<br>
• Bishop 2006 — StandardScaler for distance-based models<br>
• Hyndman & Athanasopoulos 2021 — datetime decomposition
</div>
<div class='literature-card'>
<b>Models & Evaluation</b><br>
• Breiman 2001 — RandomForest<br>
• Chawla et al. 2002 — SMOTE for imbalance<br>
• Kohavi 1995 — cross-validation
</div>
""", unsafe_allow_html=True)

    # Report
    if result.get("report_text"):
        st.markdown("### 📄 Full Report")
        st.code(result["report_text"], language="text")
