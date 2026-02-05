import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from openai import OpenAI

warnings.filterwarnings('ignore')



def genai_summary(df):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment")
        return None

    client = OpenAI(api_key=api_key)

    info = f"{df.shape[0]} rows x {df.shape[1]} columns. First columns: {list(df.columns[:5])}"
    prompt = f"Analyze this dataset: {info}. Provide a sharp 2-sentence business insight."

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=80
        )
        return response.output_text.strip()

    except Exception as e:
        print("GenAI Error:", e)
        return None


# =========================
# EDA FUNCTION
# =========================
def comprehensive_eda(df, use_genai=False):
    print("\n## Dataset Overview")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Total Missing Values:", df.isnull().sum().sum())

    # Detect types
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Basic stats
    print("\n## Stats")
    if cat_cols:
        print(f"\nTop categories in '{cat_cols[0]}':")
        print(df[cat_cols[0]].value_counts().head())

    if num_cols:
        print(f"\nStats for '{num_cols[0]}':")
        print(df[num_cols[0]].describe())

    # =========================
    # Visualizations
    # =========================
    print("\n## Visualizations")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    i = 0

    if num_cols:
        axes[i].hist(df[num_cols[0]].dropna(), bins=20)
        axes[i].set_title(num_cols[0])
        i += 1

    if len(num_cols) > 1:
        df.boxplot(column=num_cols[1], ax=axes[i])
        axes[i].set_title(num_cols[1])
        i += 1

    if cat_cols:
        df[cat_cols[0]].value_counts().head(8).plot.bar(ax=axes[i])
        axes[i].set_title(cat_cols[0])
        i += 1

    if len(num_cols) >= 2:
        axes[i].scatter(df[num_cols[0]], df[num_cols[1]])
        axes[i].set_title(f"{num_cols[0]} vs {num_cols[1]}")
        i += 1

    if len(num_cols) >= 2:
        sns.heatmap(df[num_cols].corr(), ax=axes[i], annot=True)
        axes[i].set_title("Correlation Heatmap")
        i += 1

    for j in range(i, 6):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # =========================
    # GenAI Summary
    # =========================
    if use_genai:
        print("\nGenerating AI Summary...")
        ai_summary = genai_summary(df)
        if ai_summary:
            print("\nAI Summary:\n", ai_summary)
        else:
            print("AI summary failed.")


# =========================
# MAIN LOOP
# =========================
print("Automated EDA System")
print("========================")

while True:
    path = input("CSV path (quit=exit): ").strip()
    if path.lower() == 'quit':
        break

    if not os.path.exists(path):
        print("File not found")
        continue

    # Try multiple encodings
    df = None
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded: {df.shape} (encoding: {enc})")
            break
        except Exception:
            pass

    if df is None:
        print("Could not read file")
        continue

    
    df.columns = df.columns.str.strip()

    use_genai = input("GenAI summary? (y/n): ").strip().lower() == 'y'

    comprehensive_eda(df, use_genai)

    print("\n" + "=" * 50 + "\n")
