import os
from langchain_core.tools import tool


@tool
def search_literature(query: str) -> str:
    """Search the web for peer-reviewed literature, best practices, and documentation on ML methods.

    Use this to find citations and evidence for imputation methods, feature engineering,
    model selection, etc. Returns summarized findings with sources.

    Args:
        query: Search query, e.g. 'best imputation method for <20% missing numerical data'
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if tavily_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            resp = client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
            result = f"=== Web Search: {query} ===\n\n"
            if resp.get("answer"):
                result += f"Answer: {resp['answer']}\n\n"
            for r in resp.get("results", []):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("content", "")[:500]
                result += f"- **{title}**\n  URL: {url}\n  {snippet}\n\n"
            return result
        except Exception as e:
            return f"Tavily search error: {e}"

    # Fallback: curated knowledge base
    knowledge = {
        "imputation": (
            "Imputation methods (literature):\n"
            "- KNN Imputation (Troyanskaya et al. 2001, Bioinformatics) - best for <20% missing, preserves local structure\n"
            "- MICE/Multivariate Imputation (van Buuren 2011) - iterative, gold standard for 20-60% MAR data\n"
            "- Mean imputation - simple, only for normally distributed data with <5% missing\n"
            "- Median imputation - robust for skewed data, WHO guidelines 2020\n"
            "- Drop column - when >60% missing (Little & Rubin 2019)\n"
        ),
        "feature engineering": (
            "Feature engineering (literature):\n"
            "- Log/Box-Cox transform (Box & Cox 1964) for skewed features\n"
            "- One-hot vs Label encoding (Buitinck et al. 2013, Pedregosa 2011)\n"
            "- StandardScaler for distance-based models (Bishop 2006)\n"
            "- datetime decomposition (Hyndman 2021)\n"
        ),
        "model": (
            "Model selection:\n"
            "- RandomForest robust default (Breiman 2001)\n"
            "- GradientBoosting often best for tabular (Friedman 2001)\n"
            "- Cross-validation k=5 standard (Kohavi 1995)\n"
            "- SMOTE for imbalance (Chawla 2002)\n"
        ),
    }

    lower = query.lower()
    for key, val in knowledge.items():
        if key in lower:
            return f"=== Knowledge Base: {query} ===\n{val}"

    return (
        f"=== Search: {query} ===\n"
        "No live search available (set TAVILY_API_KEY for web search).\n"
        "Falling back to general ML knowledge:\n"
        "- For imputation: consider KNN (<20% missing), MICE (20-60%), drop (>60%)\n"
        "- For skewed features: log transform\n"
        "- For categoricals: one-hot for nominal, label for ordinal\n"
    )
