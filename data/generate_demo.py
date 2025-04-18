import pandas as pd
import numpy as np
import os


def generate_demo_dataset():
    np.random.seed(42)
    n = 500

    age = np.random.normal(40, 12, n).clip(18, 80).astype(int)
    income = np.random.lognormal(10.5, 0.8, n).astype(int)
    credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)
    loan_amount = np.random.uniform(1000, 50000, n).astype(int)
    monthsEmployed = np.random.exponential(36, n).clip(0, 360).astype(int)
    interest_rate = np.random.uniform(3.5, 25.0, n).round(2)

    education = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.4, 0.2, 0.1])
    employment = np.random.choice(["Employed", "Self-Employed", "Unemployed", "Retired"], n, p=[0.6, 0.15, 0.15, 0.1])
    marital_status = np.random.choice(["Single", "Married", "Divorced"], n, p=[0.4, 0.45, 0.15])

    risk_score = (
        0.3 * (age / 80) +
        0.2 * (income / income.max()) +
        0.25 * (credit_score / 850) +
        0.15 * (monthsEmployed / 360) -
        0.1 * (loan_amount / 50000) -
        0.05 * (interest_rate / 25)
    )
    default_prob = 1 / (1 + np.exp(-(risk_score - 0.5) * 5))
    default = (np.random.random(n) < (1 - default_prob)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "months_employed": monthsEmployed,
        "interest_rate": interest_rate,
        "education": education,
        "employment_status": employment,
        "marital_status": marital_status,
        "loan_default": default,
    })

    null_indices = np.random.choice(n, size=int(n * 0.08), replace=False)
    df.loc[null_indices[:20], "credit_score"] = np.nan
    df.loc[null_indices[20:35], "income"] = np.nan
    df.loc[null_indices[35:], "months_employed"] = np.nan

    df.loc[np.random.choice(n, 3, replace=False), "age"] = 200
    df.loc[np.random.choice(n, 2, replace=False), "income"] = -5000

    duplicates = df.sample(5, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "loan_default_demo.csv")
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    path = generate_demo_dataset()
    print(f"Demo dataset saved to: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"Target distribution:\n{df['loan_default'].value_counts()}")
