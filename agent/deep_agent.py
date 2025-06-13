import os
from deepagents import create_deep_agent
from agent.tools.codeact import write_and_execute_python, execute_python_file, read_file_content, list_output_files
from agent.tools.web_search import search_literature

ML_SYSTEM_PROMPT = """You are an autonomous ML engineering agent. You solve ML tasks by WRITING AND EXECUTING PYTHON CODE.

You have access to:
- write_and_execute_python: Write a .py file to outputs/code/ and execute it
- execute_python_file: Re-run an existing file
- read_file_content: Read any file that was created
- list_output_files: See what outputs exist
- search_literature: Search for peer-reviewed backing for methods

Your skill (agent/skills/ml-pipeline/SKILL.md) defines the full workflow. Follow it step by step.

CRITICAL RULES:
1. You MUST write code to files and execute them. Never claim analysis is done without running code.
2. After each execution, READ the output. If it failed, FIX the code and re-execute.
3. Document your reasoning in code comments and in prints.
4. Each step depends on the previous — imputation needs profile, features need imputed data, etc.
5. Cite literature for imputation, transformations, and model choices.
6. The dataset path will be given to you. Discover everything about it by writing code.
"""

SUBAGENTS = [
    {
        "name": "data-explorer",
        "description": "Explores datasets by writing and executing Python code to profile, analyze distributions, and find patterns. Use for data understanding and EDA.",
        "system_prompt": (
            "You are a data exploration specialist. You understand datasets by WRITING PYTHON CODE and EXECUTING it. "
            "Given a dataset path, you write profiling and EDA scripts to outputs/code/, execute them, "
            "read the output, and summarize findings. You handle CSV, Excel, JSON, Parquet. "
            "You check dtypes, nulls, distributions, correlations, outliers, and class balance. "
            "Every conclusion must be backed by executed code output."
        ),
    },
    {
        "name": "researcher",
        "description": "Searches for peer-reviewed literature and best practices to back ML decisions. Use for imputation methods, transformations, encoding choices, and model selection.",
        "system_prompt": (
            "You are a research specialist. You find peer-reviewed citations for ML decisions. "
            "Given a data issue (e.g. '15% missing in numerical column with skewed distribution'), "
            "you search literature and return the recommended method WITH citation. "
            "Key papers: Troyanskaya 2001 (KNN imputation), van Buuren 2011 (MICE), "
            "Little & Rubin 2019 (missing data theory), Box & Cox 1964 (transforms), "
            "Chawla 2002 (SMOTE), Breiman 2001 (RandomForest)."
        ),
    },
    {
        "name": "code-engineer",
        "description": "Writes, executes, and debugs Python code for ML pipeline steps. Use for imputation, feature engineering, and model training code.",
        "system_prompt": (
            "You are a code engineering specialist. You write Python files to outputs/code/ and execute them. "
            "You handle imputation (KNN/median/mode/drop with literature backing), "
            "feature engineering (datetime decomposition, log transforms, encoding, scaling), "
            "and model training (multiple models, CV, metrics, saving best). "
            "You ALWAYS execute code after writing, read errors, fix, and retry until success. "
            "Every file must be runnable standalone. Document reasoning in comments and prints."
        ),
    },
    {
        "name": "evaluator",
        "description": "Evaluates model results, generates reports, and validates pipeline outputs. Use for final evaluation and documentation.",
        "system_prompt": (
            "You are an evaluation specialist. You validate model results, compare metrics, "
            "generate comprehensive reports with reasoning traces, and verify all outputs. "
            "You write report generation code to outputs/code/, execute it, and produce "
            "outputs/reports/report.txt with every decision documented."
        ),
    },
]

TOOLS = [
    write_and_execute_python,
    execute_python_file,
    read_file_content,
    list_output_files,
    search_literature,
]


def get_deep_ml_agent(model: str = None):
    """Create the deep ML agent with CodeAct execution, subagents, and skills."""
    from config import LLM_MODEL

    model_name = model or LLM_MODEL

    agent = create_deep_agent(
        model=model_name,
        tools=TOOLS,
        system_prompt=ML_SYSTEM_PROMPT,
        subagents=SUBAGENTS,
    )
    return agent


def run_ml_pipeline(dataset_path: str, target: str = "", task: str = "") -> str:
    """Invoke the deep agent to run the full ML pipeline on a dataset.

    The agent will dynamically write and execute code for each step.
    """
    agent = get_deep_ml_agent()

    task_hint = f" Target column is '{target}'." if target else " Auto-detect the target column."
    task_type_hint = f" Task type is '{task}'." if task else " Auto-detect task type from target."

    prompt = (
        f"Run the full ML pipeline on the dataset at: {dataset_path}."
        f"{task_hint}{task_type_hint}\n\n"
        f"Follow your skill workflow in agent/skills/ml-pipeline/SKILL.md step by step. "
        f"At each step, write a Python file to outputs/code/ and execute it. "
        f"Delegate to subagents where appropriate:\n"
        f"- data-explorer for profiling and EDA\n"
        f"- researcher for literature-backed method selection\n"
        f"- code-engineer for imputation, feature engineering, and model training\n"
        f"- evaluator for final report\n\n"
        f"Document every decision with literature citations and reasoning. "
        f"Save the best model to outputs/models/ and report to outputs/reports/."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    # Extract final message
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    return str(result)
