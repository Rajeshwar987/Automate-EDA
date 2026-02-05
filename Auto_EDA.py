import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def comprehensive_eda(df):
    """Complete EDA system matching all specifications"""
    print("## Dataset Overview")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Column names and inferred types:")
    print(df.dtypes.to_string())
    print("\nMissing values summary:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].to_string() if missing.sum() > 0 else "No missing values")
    
    # Quality checks
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates: {duplicates}")
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Columns with one value: {constant_cols}")
    high_missing_cols = missing_df[missing_df['Missing %'] > 50]['Missing %'].index.tolist()
    print(f"Columns with >50% missing: {high_missing_cols}")
    
    # Column types
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Descriptive Statistics
    print("\n## Descriptive Statistics")
    if cat_cols:
        example_cat = cat_cols[0]
        freq = df[example_cat].value_counts()
        freq_pct = (freq / len(df)) * 100
        cat_stats = pd.DataFrame({'Count': freq, 'Percentage': freq_pct.round(2)})
        print(f"\nCategorical - {example_cat}:")
        print(cat_stats.to_string())
    
    if len(num_cols) >= 2:
        print("\nNumeric columns (first two):")
        for col in num_cols[:2]:
            data = df[col].dropna()
            if len(data) > 0:
                min_val, max_val = data.min(), data.max()
                mean_val, median_val = data.mean(), data.median()
                mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
                std_val = data.std()
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
                print(f"\n{col}:")
                print(f"  Min/Max: {min_val:.2f}/{max_val:.2f}")
                print(f"  Mean/Median/Mode: {mean_val:.2f}/{median_val:.2f}/{mode_val:.2f}")
                print(f"  Std/IQR: {std_val:.2f}/{iqr:.2f}")
                print(f"  Outliers (1.5x IQR): {outliers}")
    
    # 5 Visualizations
    print("\n## Visualizations")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    plot_idx = 0
    
    if num_cols:
        axes[plot_idx].hist(df[num_cols[0]].dropna(), bins=30, edgecolor='black')
        axes[plot_idx].set_title(f'Histogram: {num_cols[0]}')
        axes[plot_idx].set_xlabel(num_cols[0])
        axes[plot_idx].set_ylabel('Frequency')
        plot_idx += 1
    
    if len(num_cols) > 1:
        df.boxplot(column=num_cols[1], ax=axes[plot_idx])
        axes[plot_idx].set_title(f'Boxplot: {num_cols[1]}')
        plot_idx += 1
    
    if cat_cols:
        df[cat_cols[0]].value_counts().plot(kind='bar', ax=axes[plot_idx])
        axes[plot_idx].set_title(f'Bar Chart: {cat_cols[0]}')
        axes[plot_idx].set_xlabel(cat_cols[0])
        axes[plot_idx].set_ylabel('Count')
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    if len(num_cols) >= 2:
        axes[plot_idx].scatter(df[num_cols[0]], df[num_cols[1]])
        axes[plot_idx].set_title(f'Scatter: {num_cols[0]} vs {num_cols[1]}')
        axes[plot_idx].set_xlabel(num_cols[0])
        axes[plot_idx].set_ylabel(num_cols[1])
        plot_idx += 1
    
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[plot_idx])
        axes[plot_idx].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Insights
    print("\n## Insights")
    insights = []
    if missing.sum() > 0:
        insights.append(f"- Missing values in {len(missing[missing > 0])} columns (max {missing_pct.max():.1f}%)")
    if duplicates > 0:
        insights.append(f"- {duplicates} duplicate rows found")
    if constant_cols:
        insights.append(f"- Constant columns: {', '.join(constant_cols[:3])}")
    if high_missing_cols:
        insights.append(f"- High missing: {', '.join(high_missing_cols)}")
    
    if num_cols:
        skewness = df[num_cols].skew().abs()
        if skewness.max() > 1:
            skewed = skewness[skewness > 1].index.tolist()
            insights.append(f"- Skewed columns: {', '.join(skewed)}")
    
    if len(num_cols) >= 2 and df[num_cols].corr().abs().max().max() > 0.7:
        insights.append("- High correlations detected")
    
    insights.extend(["- Ready for modeling after cleaning", "- Check outliers before training"])
    for insight in insights[:10]:
        print(f"- {insight}")
    
    print("\n## Limitations")
    print("- Missingness patterns unknown")
    print("- Sampling bias not assessed")
    print("- Domain context needed for outliers")

# ========================================
# FULL INTERACTIVE RUNTIME INPUT - NO DEFAULTS
# ========================================
print("Interactive EDA System")
print("========================")

while True:
    url = input("\n Enter CSV URL (or 'quit' to exit): ").strip()
    
    if url.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not url:
        print("ERROR: No URL provided")
        print("Enter a valid CSV URL (e.g., raw.githubusercontent.com link)")
        continue
    
    print(f"Loading from: {url}")
    print("-" * 50)
    
    try:
        df = pd.read_csv(url)
        print(f" Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print("-" * 50)
        comprehensive_eda(df)
        print("\n" + "="*50 + "\n")
        
    except Exception as e:
        print(f" ERROR loading URL: {str(e)}")
        print(" Common fixes:")
        print("   • Use RAW GitHub links (raw.githubusercontent.com)")
        print("   • Check URL is direct CSV download")
        print("   • Verify URL accessibility")
        print()
