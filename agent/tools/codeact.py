import subprocess
import sys
import os
import traceback
from langchain_core.tools import tool


@tool
def write_and_execute_python(code: str, filename: str, description: str = "") -> str:
    """Write Python code to a file in outputs/code/ and execute it. Returns stdout, stderr, and any errors.

    Use this to write code for any ML pipeline step — EDA, imputation, feature engineering, model training.
    The code is saved and executed. If it errors, you will see the traceback and can fix it.

    Args:
        code: Complete Python code to write and execute
        filename: Filename to save as (e.g., '01_eda.py', '02_impute.py')
        description: Brief description of what this code does
    """
    from config import CODE_DIR

    os.makedirs(CODE_DIR, exist_ok=True)
    filepath = os.path.join(CODE_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(CODE_DIR)),
        )
        output = f"=== Executed: {filename} ===\n"
        if description:
            output += f"Description: {description}\n"
        output += f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"\n--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}\n"
        if result.returncode != 0:
            output += "\n*** CODE FAILED - fix and retry ***\n"
        else:
            output += "\n*** CODE SUCCEEDED ***\n"
        return output
    except subprocess.TimeoutExpired:
        return f"=== {filename} TIMED OUT (120s) ==="
    except Exception as e:
        return f"=== Execution error for {filename}: {e}\n{traceback.format_exc()} ==="


@tool
def execute_python_file(filename: str) -> str:
    """Execute an existing Python file in outputs/code/ and return its output."""

    from config import CODE_DIR

    filepath = os.path.join(CODE_DIR, filename)
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"

    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(CODE_DIR)),
        )
        output = f"=== Executed: {filename} ===\nExit code: {result.returncode}\n"
        if result.stdout:
            output += f"\n--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}\n"
        return output
    except subprocess.TimeoutExpired:
        return f"=== {filename} TIMED OUT ==="
    except Exception as e:
        return f"=== Error: {e} ==="


@tool
def read_file_content(filepath: str) -> str:
    """Read the contents of any file (code, reports, data summaries)."""

    if not os.path.exists(filepath):
        abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filepath)
        if os.path.exists(abs_path):
            filepath = abs_path
        else:
            return f"File not found: {filepath}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            return content[:8000] + f"\n... (truncated, total {len(content)} chars)"
        return content
    except Exception as e:
        return f"Error reading {filepath}: {e}"


@tool
def list_output_files(directory: str = "") -> str:
    """List files in the outputs directory. Use to see what the pipeline has produced."""

    from config import OUTPUT_DIR

    target = os.path.join(OUTPUT_DIR, directory) if directory else OUTPUT_DIR
    if not os.path.exists(target):
        return f"Directory not found: {target}"

    files = []
    for root, dirs, filenames in os.walk(target):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, OUTPUT_DIR)
            size = os.path.getsize(fpath)
            files.append(f"{rel} ({size} bytes)")

    if not files:
        return "No output files yet."

    return "\n".join(sorted(files))
