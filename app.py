import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ML Deep Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp {
    background: linear-gradient(135deg, #080a10 0%, #0f1a2a 50%, #0a1628 100%);
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 { color: #00D4FF !important; }
.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0088CC 100%) !important;
    color: #000 !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,212,255,0.4); }
.code-card {
    background: rgba(15,25,35,0.95); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px; padding: 14px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #00D4FF;
}
.reasoning-card {
    background: rgba(0,212,255,0.05); border-left: 3px solid #00D4FF;
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 10px 0;
}
.step-indicator {
    display: inline-block; background: linear-gradient(135deg, #00D4FF, #0088CC);
    color: #000; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700;
}
.metric-box {
    background: rgba(15,25,35,0.9); border: 1px solid rgba(0,212,255,0.3);
    border-radius: 10px; padding: 18px; text-align: center;
}
.metric-value { font-size: 26px; font-weight: 700; color: #00D4FF; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.literature-card {
    background: rgba(0,200,100,0.05); border: 1px solid rgba(0,200,100,0.3);
    border-radius: 10px; padding: 14px; margin: 8px 0;
}
.agent-running { border-left: 3px solid #FFB800; animation: pulse 2s infinite; }
.agent-done { border-left: 3px solid #00FF88; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
div[data-testid="stExpander"] { background: rgba(15,25,35,0.7) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0;'>
    <span style='font-size:28px;'>🤖</span><br>
    <span style='font-size:17px; font-weight:700; color:#00D4FF;'>ML Deep Agent</span><br>
    <span style='font-size:10px; color:#666;'>DeepAgents + CodeAct</span><br>
    <span style='font-size:10px; color:#888;'>writes code → executes → fixes → iterates</span>
</div>
<hr style='border-color:#1a2a3a;'>
""", unsafe_allow_html=True)

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = []

page = st.sidebar.radio("Navigate", ["🤖 Pipeline", "📊 Results", "💻 Generated Code", "📝 Reasoning", "📥 Export"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding:10px; background:rgba(0,212,255,0.06); border-radius:8px; border:1px solid rgba(0,212,255,0.2);'>
<span style='font-size:10px; color:#00D4FF;'>
<b>How it works</b><br>
Agent writes Python files to <code>outputs/code/</code>, executes them, reads output/errors, fixes, and iterates. No static hand-coded steps — every decision is dynamic code.
</span>
</div>
""", unsafe_allow_html=True)

from ui.pages import pipeline_page, results_page, reasoning_page, export_page
import ui.pages.code_page as code_page

if page == "🤖 Pipeline":
    pipeline_page.render()
elif page == "📊 Results":
    results_page.render()
elif page == "💻 Generated Code":
    code_page.render()
elif page == "📝 Reasoning":
    reasoning_page.render()
elif page == "📥 Export":
    export_page.render()
