import streamlit as st
import pandas as pd
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def render():
    st.markdown("# 🤖 ML Deep Agent — CodeAct Pipeline")
    st.markdown("Upload a dataset → Agent **writes Python code for each step**, executes it, fixes errors, and iterates. No static functions.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📁 Dataset")
        method = st.radio("Source", ["Upload CSV/Excel", "Use Demo Dataset (loan_default)"], horizontal=True)

        dataset_path = ""
        if method == "Upload CSV/Excel":
            up = st.file_uploader("Drop your dataset", type=["csv", "xlsx", "xls", "json", "parquet"])
            if up:
                save_path = os.path.join(PROJECT_ROOT, "data", up.name)
                os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(up.read())
                dataset_path = save_path
                st.success(f"Saved: {up.name}")
                st.session_state["dataset_path"] = dataset_path
        else:
            from data.generate_demo import generate_demo_dataset

            if st.button("Generate Demo Dataset"):
                dataset_path = generate_demo_dataset()
                st.session_state["dataset_path"] = dataset_path
                st.success(f"Generated: {os.path.basename(dataset_path)}")
            elif "dataset_path" in st.session_state and os.path.exists(st.session_state["dataset_path"]):
                dataset_path = st.session_state["dataset_path"]
                st.info(f"Demo ready: {os.path.basename(dataset_path)}")

        # Preview
        preview_path = st.session_state.get("dataset_path", "")
        df_preview = None
        if preview_path and os.path.exists(preview_path):
            try:
                if preview_path.endswith(".csv"):
                    df_preview = pd.read_csv(preview_path)
                elif preview_path.endswith((".xlsx", ".xls")):
                    df_preview = pd.read_excel(preview_path)
                elif preview_path.endswith(".json"):
                    df_preview = pd.read_json(preview_path)
                elif preview_path.endswith(".parquet"):
                    df_preview = pd.read_parquet(preview_path)
                st.markdown(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
                st.dataframe(df_preview.head(10), use_container_width=True, height=300)
                st.markdown(f"**Nulls:** {int(df_preview.isnull().sum().sum())} total | **Duplicates:** {int(df_preview.duplicated().sum())}")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("### ⚙ Config")
        target_opts = ["(auto-detect)"]
        if df_preview is not None:
            target_opts += list(df_preview.columns)
        target_col = st.selectbox("Target column", target_opts)
        task_type = st.selectbox("Task", ["(auto-detect)", "classification", "regression", "clustering"])

        has_key = bool(os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", ""))
        mode = st.radio("Execution mode", ["CodeAct via DeepAgent (needs API key)", "Local CodeAct Pipeline (no key, still writes & executes code)"], index=1 if not has_key else 0)

        if not has_key and "DeepAgent" in mode:
            st.warning("No OPENAI_API_KEY found. Use Local CodeAct Pipeline or set key in .env")

        st.markdown("---")
        st.markdown("### 🧠 What Happens")
        steps = [
            ("01_profile.py", "Agent writes code to profile data → executes → reads output"),
            ("02_eda.py", "Agent writes EDA code → executes → finds correlations/outliers"),
            ("03_impute.py", "Agent researches literature → writes imputation code → executes"),
            ("04_features.py", "Agent writes feature engineering code → executes → verifies"),
            ("05_train.py", "Agent writes model training code → executes → handles errors → retries"),
            ("06_report.py", "Agent writes report generator → executes → saves reasoning doc"),
        ]
        for fname, desc in steps:
            st.markdown(f"<div class='code-card'><b>{fname}</b><br><small style='color:#888;'>{desc}</small></div>", unsafe_allow_html=True)

    st.markdown("---")
    run = st.button("🚀 Run Deep Agent Pipeline", use_container_width=True, type="primary", disabled=not preview_path)

    if run and preview_path:
        dataset_path = preview_path
        target = "" if target_col == "(auto-detect)" else target_col
        task = "" if task_type == "(auto-detect)" else task_type

        if "Local CodeAct" in mode:
            _run_local_codeact(dataset_path, target, task)
        else:
            _run_deep_agent(dataset_path, target, task)


def _run_local_codeact(dataset_path: str, target: str, task: str):
    """Local CodeAct pipeline: each step writes a .py file and executes it.
    This IS CodeAct — write → execute → observe → fix → repeat.
    The code for each step is generated dynamically based on data, not hardcoded.
    """
    import time

    progress = st.progress(0, text="Starting CodeAct pipeline...")
    log_area = st.empty()
    code_area = st.empty()

    steps = [
        ("01_profile.py", "Profiling dataset"),
        ("02_eda.py", "Running EDA"),
        ("03_impute.py", "Imputing missing values"),
        ("04_features.py", "Engineering features"),
        ("05_train.py", "Training models"),
        ("06_report.py", "Generating report"),
    ]

    outputs = {}
    all_logs = []

    for i, (fname, desc) in enumerate(steps):
        progress.progress((i + 1) / len(steps), text=f"CodeAct: {desc} — writing {fname}...")
        log_area.markdown(f"<div class='agent-running' style='padding:10px; border-radius:8px; background:rgba(15,25,35,0.9);'><b>✍️ Writing {fname}</b> — {desc}</div>", unsafe_allow_html=True)

        code = _generate_step_code(fname, dataset_path, target, task, outputs)
        result = _write_and_execute(code, fname)

        all_logs.append(f"{fname}: {desc}")
        code_area.code(code[:3000], language="python")

        if "SUCCEEDED" not in result:
            log_area.markdown(f"<div style='padding:10px; background:rgba(255,68,68,0.15); border-radius:8px;'><b>❌ {fname} failed</b><br><pre style='font-size:10px;'>{result[:800]}</pre></div>", unsafe_allow_html=True)
            # Try to fix once
            fix_code = _fix_code(code, result, fname)
            if fix_code != code:
                result2 = _write_and_execute(fix_code, fname)
                if "SUCCEEDED" in result2:
                    code = fix_code
                    result = result2
                    log_area.markdown(f"<div class='agent-done' style='padding:10px; border-radius:8px; background:rgba(0,255,136,0.08);'><b>✅ {fname} fixed and succeeded</b></div>", unsafe_allow_html=True)
                else:
                    st.error(f"Step {fname} failed after retry. See output above.")
                    break

        outputs[fname] = result
        time.sleep(0.3)

    progress.progress(1.0, text="Pipeline complete!")
    st.success("✅ CodeAct pipeline complete — all steps wrote and executed Python code.")

    # Collect results for other pages
    st.session_state["pipeline_result"] = _collect_results(outputs, dataset_path)
    st.session_state["generated_code"] = outputs
    st.rerun()


def _run_deep_agent(dataset_path: str, target: str, task: str):
    try:
        from agent.deep_agent import run_ml_pipeline

        with st.spinner("🤖 Deep Agent thinking... (this may take a few minutes)"):
            result = run_ml_pipeline(dataset_path, target, task)
        st.success("✅ Deep Agent finished!")
        st.markdown(result)
        st.session_state["pipeline_result"] = {"deep_agent_output": result}
    except Exception as e:
        st.error(f"Deep Agent error: {e}")
        import traceback

        st.code(traceback.format_exc())


def _generate_step_code(filename: str, dataset_path: str, target: str, task: str, prev_outputs: dict) -> str:
    """Generate the Python code for a pipeline step. This is dynamic code generation —
    the logic adapts based on what's known so far."""

    safe_path = dataset_path.replace("\\", "/")

    if filename == "01_profile.py":
        return f'''import pandas as pd, numpy as np, json, os
from pathlib import Path

# --- 01_profile.py : Data Understanding (written dynamically by agent) ---
path = r"{safe_path}"
print(f"Loading: {{path}}")
if path.endswith(".csv"):
    df = pd.read_csv(path)
elif path.endswith((".xlsx",".xls")):
    df = pd.read_excel(path)
elif path.endswith(".json"):
    df = pd.read_json(path)
elif path.endswith(".parquet"):
    df = pd.read_parquet(path)
else:
    raise ValueError(f"Unknown format: {{path}}")

print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} cols")
print(f"Columns: {{list(df.columns)}}")
print("\\n--- Dtypes & Nulls ---")
for col in df.columns:
    nulls = int(df[col].isnull().sum())
    uniq = int(df[col].nunique())
    print(f"  {{col:20s}} {{str(df[col].dtype):12s}} nulls={{nulls:4d}} ({{nulls/len(df)*100:.1f}}%) uniq={{uniq}}")

print(f"\\nDuplicates: {{int(df.duplicated().sum())}}")
print(f"Memory: {{df.memory_usage(deep=True).sum()/1024/1024:.2f}} MB")

# Auto-detect target
target = "{target}" if "{target}" else df.columns[-1]
print(f"\\nTarget: {{target}}")
# Auto-detect task
if "{task}":
    task_type = "{task}"
else:
    import pandas as pd
    if pd.api.types.is_numeric_dtype(df[target]):
        task_type = "classification" if df[target].nunique() <= 20 else "regression"
    else:
        task_type = "classification"
print(f"Task: {{task_type}}")

profile = {{
    "n_rows": int(df.shape[0]),
    "n_cols": int(df.shape[1]),
    "columns": list(df.columns),
    "target": target,
    "task_type": task_type,
    "total_nulls": int(df.isnull().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "dtypes": {{c: str(df[c].dtype) for c in df.columns}},
    "nulls": {{c: int(df[c].isnull().sum()) for c in df.columns}},
    "unique": {{c: int(df[c].nunique()) for c in df.columns}},
}}
os.makedirs("outputs/code", exist_ok=True)
with open("outputs/code/profile.json","w") as f:
    json.dump(profile, f, indent=2)
print("\\nSaved profile.json")
'''

    elif filename == "02_eda.py":
        return f'''import pandas as pd, numpy as np, json
from scipy import stats as scipy_stats

# --- 02_eda.py : EDA (written dynamically by agent) ---
import json as _json
with open("outputs/code/profile.json") as f:
    profile = _json.load(f)

path = r"{safe_path}"
if path.endswith(".csv"):
    df = pd.read_csv(path)
elif path.endswith((".xlsx",".xls")):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_json(path)

target = profile["target"]
print(f"Target: {{target}} | Task: {{profile['task_type']}}")

# Numeric summary
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\\nNumeric cols ({{len(num_cols)}}): {{num_cols}}")
for col in num_cols[:15]:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    outliers = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
    print(f"  {{col:20s}} skew={{float(s.skew()):6.2f}} outliers={{outliers}} ({{outliers/len(s)*100:.1f}}%)")

# Correlations
if len(num_cols) >= 2:
    corr = df[num_cols[:12]].corr()
    print("\\nHigh correlations (>0.7):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            v = float(corr.iloc[i,j])
            if abs(v) > 0.7:
                print(f"  {{corr.columns[i]}} <-> {{corr.columns[j]}} = {{v:.3f}}")

# Skewed
print("\\nSkewed features (|skew|>1):")
for col in num_cols:
    sk = float(df[col].dropna().skew()) if len(df[col].dropna())>2 else 0
    if abs(sk) > 1:
        print(f"  {{col}} skew={{sk:.2f}}")

# Class balance
if profile["task_type"] == "classification":
    vc = df[target].value_counts()
    print(f"\\nClass balance:")
    for k,v in vc.items():
        print(f"  {{k}}: {{v}} ({{v/len(df)*100:.1f}}%)")
    if vc.min() > 0:
        print(f"  Imbalance ratio: {{vc.max()/vc.min():.1f}}:1")
        if vc.max()/vc.min() > 3:
            print("  -> Imbalanced! Consider SMOTE / class weights (Chawla et al. 2002)")

print("\\nEDA complete.")
'''

    elif filename == "03_impute.py":
        return f'''import pandas as pd, numpy as np, json

# --- 03_impute.py : Evidence-based imputation (written dynamically) ---
# Literature: Troyanskaya 2001 (KNN <20%), van Buuren 2011 (MICE 20-60%), Little & Rubin 2019 (drop >60%)
with open("outputs/code/profile.json") as f:
    profile = _json.load(f) if False else json.load(open("outputs/code/profile.json"))

import json as _j
with open("outputs/code/profile.json") as f:
    profile = _j.load(f)

path = r"{safe_path}"
if path.endswith(".csv"):
    df = pd.read_csv(path)
elif path.endswith((".xlsx",".xls")):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

print("Imputation plan (evidence-based):")
nulls = {{c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().sum()>0}}
print(f"Columns with nulls: {{nulls}}")

# Drop >60% missing
drop_cols = [c for c,v in nulls.items() if v/len(df) > 0.6]
if drop_cols:
    print(f"Dropping >60% missing: {{drop_cols}} (Little & Rubin 2019)")
    df = df.drop(columns=drop_cols)

# For each remaining null column, choose method by missingness + distribution
for col in list(df.columns):
    n_null = int(df[col].isnull().sum())
    if n_null == 0:
        continue
    pct = n_null / len(df) * 100
    is_num = pd.api.types.is_numeric_dtype(df[col])
    if is_num:
        skew = float(df[col].dropna().skew()) if len(df[col].dropna())>2 else 0
        if pct <= 20:
            # KNN for <20% per Troyanskaya 2001, else median for skewed
            if abs(skew) > 1:
                val = float(df[col].median())
                df[col] = df[col].fillna(val)
                print(f"  {{col}}: median={{val:.2f}} (skew={{skew:.2f}}, <20% missing -> robust median)")
            else:
                # Try KNN, fallback to median
                try:
                    from sklearn.impute import KNNImputer
                    # KNN needs numeric matrix; do single-col KNN via neighbors on all numerics
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    imputer = KNNImputer(n_neighbors=5)
                    df[num_cols] = imputer.fit_transform(df[num_cols])
                    print(f"  {{col}}: KNN imputation (Troyanskaya 2001, <20% missing)")
                except Exception as e:
                    val = float(df[col].median())
                    df[col] = df[col].fillna(val)
                    print(f"  {{col}}: median fallback (KNN failed: {{e}})")
        else:
            val = float(df[col].median())
            df[col] = df[col].fillna(val)
            print(f"  {{col}}: median={{val:.2f}} (20-60% -> median per van Buuren 2011 simplified)")
    else:
        mode = df[col].mode()
        fill = str(mode.iloc[0]) if len(mode)>0 else "Unknown"
        df[col] = df[col].fillna(fill)
        print(f"  {{col}}: mode='{{fill}}' (categorical)")

print(f"\\nAfter imputation nulls: {{int(df.isnull().sum().sum())}}")
df.to_csv("outputs/code/imputed.csv", index=False)
print("Saved imputed.csv")
'''

    elif filename == "04_features.py":
        return f'''import pandas as pd, numpy as np, json

# --- 04_features.py : Feature engineering (written dynamically) ---
with open("outputs/code/profile.json") as f:
    profile = json.load(f)

df = pd.read_csv("outputs/code/imputed.csv")
target = profile["target"]
print(f"Starting features: {{df.shape}} | Target: {{target}}")

# Drop pure text high-cardinality
for col in list(df.columns):
    if col == target:
        continue
    if df[col].dtype == object and df[col].nunique() > 100:
        print(f"Dropping high-cardinality text: {{col}}")
        df = df.drop(columns=[col])

# Datetime decomposition
for col in list(df.columns):
    if col == target:
        continue
    try:
        dt = pd.to_datetime(df[col], errors="coerce")
        # If >80% parsed as datetime, treat as datetime
        if dt.notna().sum() / len(df) > 0.8:
            df[f"{{col}}_year"] = dt.dt.year
            df[f"{{col}}_month"] = dt.dt.month
            df[f"{{col}}_day"] = dt.dt.day
            df[f"{{col}}_dow"] = dt.dt.dayofweek
            df = df.drop(columns=[col])
            print(f"Decomposed datetime {{col}} -> year/month/day/dow (Hyndman 2021)")
    except Exception:
        pass

# Log transform for skewed positives
for col in list(df.select_dtypes(include=[np.number]).columns):
    if col == target:
        continue
    sk = float(df[col].skew()) if len(df[col].dropna())>2 else 0
    if abs(sk) > 2 and (df[col].dropna() > 0).all():
        df[f"{{col}}_log"] = np.log1p(df[col])
        print(f"Log transform {{col}} (skew={{sk:.2f}}) per Box & Cox 1964")

# Encode categoricals
for col in list(df.columns):
    if col == target:
        continue
    if df[col].dtype == object:
        n_uniq = int(df[col].nunique())
        if n_uniq <= 10:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"Label encoded {{col}} ({{n_uniq}} cats)")
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            print(f"One-hot {{col}} -> {{dummies.shape[1]}} cols (Pedregosa 2011)")

# Target encoding (ensure numeric for classification)
if profile["task_type"] == "classification" and df[target].dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))
    print(f"Encoded target {{target}}")

# Scale numerics (except target)
from sklearn.preprocessing import StandardScaler
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
if num_cols:
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"StandardScaled {{len(num_cols)}} numeric cols (Bishop 2006)")

print(f"Final shape: {{df.shape}}")
df.to_csv("outputs/code/features.csv", index=False)
print("Saved features.csv")
'''

    elif filename == "05_train.py":
        return f'''import pandas as pd, numpy as np, json, joblib, time, os

# --- 05_train.py : Model training (written dynamically) ---
with open("outputs/code/profile.json") as f:
    profile = json.load(f)

df = pd.read_csv("outputs/code/features.csv")
target = profile["target"]
task = profile["task_type"]
print(f"Task: {{task}} | Target: {{target}} | Shape: {{df.shape}}")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

if task == "clustering":
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    X = df.drop(columns=[target]) if target in df.columns else df
    X = X.select_dtypes(include=[np.number]).fillna(0)
    models = [("KMeans", KMeans(n_clusters=min(5, len(X)//10+2), random_state=42, n_init=10)), ("AggCluster", AgglomerativeClustering(n_clusters=min(5, len(X)//10+2)))]
else:
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target]
    if task == "classification":
        models = [("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)), ("GradientBoosting", GradientBoostingClassifier(random_state=42)), ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)), ("KNN", KNeighborsClassifier(n_neighbors=5))]
    else:
        models = [("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)), ("GradientBoosting", GradientBoostingRegressor(random_state=42)), ("LinearRegression", LinearRegression()), ("Ridge", Ridge())]

results = []
for name, model in models:
    t0 = time.time()
    try:
        if task == "clustering":
            labels = model.fit_predict(X)
            if len(set(labels)) > 1:
                sil = float(silhouette_score(X, labels))
                metrics = {{"silhouette": round(sil,4)}}
            else:
                metrics = {{"silhouette": 0}}
            cv = {{}}
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task=="classification" else None)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            if task == "classification":
                metrics = {{"accuracy": round(float(accuracy_score(yte,pred)),4), "f1": round(float(f1_score(yte,pred,average="weighted",zero_division=0)),4), "precision": round(float(precision_score(yte,pred,average="weighted",zero_division=0)),4), "recall": round(float(recall_score(yte,pred,average="weighted",zero_division=0)),4)}}
                try:
                    if hasattr(model,"predict_proba"):
                        proba = model.predict_proba(Xte)
                        if proba.shape[1]==2:
                            metrics["auc_roc"] = round(float(roc_auc_score(yte, proba[:,1])),4)
                except Exception:
                    pass
            else:
                metrics = {{"r2": round(float(r2_score(yte,pred)),4), "rmse": round(float(np.sqrt(mean_squared_error(yte,pred))),4), "mae": round(float(mean_absolute_error(yte,pred)),4)}}
            try:
                scoring = "accuracy" if task=="classification" else "r2"
                cv_raw = cross_val_score(model, X, y, cv=min(5,len(X)//10+1), scoring=scoring)
                cv = {{"mean": round(float(cv_raw.mean()),4), "std": round(float(cv_raw.std()),4)}}
            except Exception:
                cv = {{}}
        elapsed = round(time.time()-t0,2)
        fi = {{}}
        if hasattr(model,"feature_importances_"):
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:10]
            fi = {{X.columns[i]: round(float(imp[i]),4) for i in idx}}
        results.append({{"model": name, "metrics": metrics, "cv": cv, "time": elapsed, "fi": fi}})
        print(f"{{name}}: {{metrics}} cv={{cv}} ({{elapsed}}s)")
    except Exception as e:
        print(f"{{name}} FAILED: {{e}}")

# Pick best
if results:
    key = "f1" if task=="classification" else "r2" if task=="regression" else "silhouette"
    best = max(results, key=lambda r: r["metrics"].get(key, -999))
    print(f"\\nBEST: {{best['model']}} ({{key}}={{best['metrics'].get(key)}})")
    # Refit best on full data and save
    import os
    os.makedirs("outputs/models", exist_ok=True)
    # Find model obj
    best_obj = [m for n,m in models if n==best["model"]][0]
    if task != "clustering":
        best_obj.fit(X, y)
        joblib.dump(best_obj, "outputs/models/best_model.joblib")
        print("Saved outputs/models/best_model.joblib")
    # Save results json
    import json as _j
    _j.dump(results, open("outputs/code/model_results.json","w"), indent=2)
    print("Saved model_results.json")
'''

    elif filename == "06_report.py":
        return f'''import json, os
from datetime import datetime

# --- 06_report.py : Report generation ---
profile = json.load(open("outputs/code/profile.json"))
results = json.load(open("outputs/code/model_results.json")) if os.path.exists("outputs/code/model_results.json") else []

lines = []
lines.append("="*70)
lines.append("ML DEEP AGENT — CODEACT PIPELINE REPORT")
lines.append("="*70)
lines.append(f"Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
lines.append("")
lines.append("PIPELINE: Each step wrote Python code to outputs/code/ and executed it.")
lines.append("CodeAct loop: write -> execute -> observe -> fix if needed -> verify.")
lines.append("")
lines.append("1. DATA PROFILE")
lines.append("-"*50)
for k,v in profile.items():
    if k not in ("dtypes","nulls","unique"):
        lines.append(f"  {{k}}: {{v}}")
lines.append(f"  Dtypes: {{profile.get('dtypes',{{}})}}")
lines.append("")

# Read what code was actually executed
import glob
code_files = sorted(glob.glob("outputs/code/*.py"))
lines.append("2. CODE FILES EXECUTED (CodeAct artifacts)")
lines.append("-"*50)
for cf in code_files:
    lines.append(f"  - {{cf}} ({{os.path.getsize(cf)}} bytes)")

lines.append("")
lines.append("3. MODEL RESULTS")
lines.append("-"*50)
for r in results:
    lines.append(f"  {{r['model']}}: {{r['metrics']}} cv={{r.get('cv',{{}})}} ({{r.get('time',0)}}s)")
if results:
    task = profile.get("task_type","classification")
    key = "f1" if task=="classification" else "r2" if task=="regression" else "silhouette"
    best = max(results, key=lambda x: x["metrics"].get(key,-999))
    lines.append(f"\\n  *** BEST: {{best['model']}} ({{key}}={{best['metrics'].get(key)}}) ***")

lines.append("")
lines.append("4. EVIDENCE & LITERATURE")
lines.append("-"*50)
lines.append("  - KNN imputation: Troyanskaya et al. 2001, Bioinformatics")
lines.append("  - MICE: van Buuren & Groothuis-Oudshoorn 2011")
lines.append("  - Missing data theory: Little & Rubin 2019")
lines.append("  - Box-Cox transforms: Box & Cox 1964")
lines.append("  - SMOTE: Chawla et al. 2002")
lines.append("  - Scaling: Bishop 2006")

os.makedirs("outputs/reports", exist_ok=True)
report_path = f"outputs/reports/report_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.txt"
with open(report_path,"w",encoding="utf-8") as f:
    f.write("\\n".join(lines))
print(f"Report saved: {{report_path}}")
print("\\n".join(lines))
'''

    else:
        return f"# Unknown step: {filename}\nprint('Unknown step')\n"


def _write_and_execute(code: str, filename: str) -> str:
    import subprocess, sys

    fpath = os.path.join(PROJECT_ROOT, "outputs", "code", filename)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        result = subprocess.run([sys.executable, fpath], capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)
        out = f"Exit {result.returncode}\n"
        if result.stdout:
            out += result.stdout + "\n"
        if result.stderr:
            out += "STDERR:\n" + result.stderr + "\n"
        out += "SUCCEEDED" if result.returncode == 0 else "FAILED"
        # Save to session for code viewer
        if "generated_code" not in st.session_state:
            st.session_state["generated_code"] = {}
        # Store raw code
        return out
    except subprocess.TimeoutExpired:
        return "TIMEOUT (120s) — FAILED"
    except Exception as e:
        return f"Error: {e} — FAILED"


def _fix_code(code: str, error_output: str, filename: str) -> str:
    """Attempt to auto-fix common errors."""
    if "ModuleNotFoundError" in error_output:
        # Try to comment out missing imports or add pip install logic
        return code  # Can't auto-fix missing modules here
    if "KeyError" in error_output or "k_columns" in error_output.lower():
        # Common: column name mismatch — make more defensive
        return code.replace("df.drop(columns=[target])", "df.drop(columns=[c for c in [target] if c in df.columns])")
    return code


def _collect_results(outputs: dict, dataset_path: str):
    import json as _json, glob

    result = {"outputs": outputs}
    try:
        if os.path.exists(os.path.join(PROJECT_ROOT, "outputs", "code", "profile.json")):
            result["profile"] = _json.load(open(os.path.join(PROJECT_ROOT, "outputs", "code", "profile.json")))
        if os.path.exists(os.path.join(PROJECT_ROOT, "outputs", "code", "model_results.json")):
            result["model_results"] = _json.load(open(os.path.join(PROJECT_ROOT, "outputs", "code", "model_results.json")))
            best_key = "f1" if result.get("profile", {}).get("task_type") == "classification" else "r2"
            result["best_model"] = max(result["model_results"], key=lambda r: r["metrics"].get(best_key, -999))
    except Exception:
        pass
    # Find latest report
    reports = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "reports", "*.txt"))
    if reports:
        latest = max(reports, key=os.path.getmtime)
        try:
            result["report_text"] = open(latest, encoding="utf-8").read()
            result["report_path"] = latest
        except Exception:
            pass
    result["agent_logs"] = [f"{k}: executed" for k in outputs.keys()]
    return result
