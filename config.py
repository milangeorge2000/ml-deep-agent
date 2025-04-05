import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LLM_MODEL = "openai:gpt-4o"
LLM_TEMPERATURE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CODE_DIR = os.path.join(OUTPUT_DIR, "code")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

for d in [OUTPUT_DIR, CODE_DIR, MODEL_DIR, REPORT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
