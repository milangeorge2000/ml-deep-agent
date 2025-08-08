import streamlit as st
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown("# 💻 Generated Code")
    st.markdown("Every pipeline step **wrote a Python file to `outputs/code/` and executed it**. This is CodeAct — not static functions.")

    code_dir = os.path.join(PROJECT_ROOT, "outputs", "code")
    if not os.path.exists(code_dir):
        st.info("No code generated yet. Run the pipeline first.")
        return

    py_files = sorted(glob.glob(os.path.join(code_dir, "*.py")))
    if not py_files:
        st.info("No Python files in outputs/code/ yet.")
        return

    st.markdown(f"**{len(py_files)} files** in `outputs/code/`:")
    for f in py_files:
        st.markdown(f"- `{os.path.basename(f)}` ({os.path.getsize(f)} bytes)")

    st.markdown("---")
    chosen = st.selectbox("View file", [os.path.basename(f) for f in py_files])
    fpath = os.path.join(code_dir, chosen)
    content = open(fpath, encoding="utf-8").read()
    st.code(content, language="python")

    # Show execution output if available
    result = st.session_state.get("pipeline_result", {})
    outputs = result.get("outputs", {})
    if chosen in outputs:
        with st.expander(f"Execution output: {chosen}", expanded=True):
            st.code(outputs[chosen], language="text")
