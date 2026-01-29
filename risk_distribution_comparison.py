import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


def get_latest_quarters(df, time_column='time_'):
    """Extract the two most recent quarters from the dataframe."""

    def parse_quarter(q):
        quarter, year = q.split('-')
        quarter_num = int(quarter[1])
        return (int(year), quarter_num)

    quarters = df[time_column].unique()
    sorted_quarters = sorted(quarters, key=parse_quarter, reverse=True)

    if len(sorted_quarters) < 2:
        raise ValueError("Need at least 2 quarters for comparison")

    return sorted_quarters[0], sorted_quarters[1]


def compare_risk_distributions(df, risk_column='risk_category', time_column='time_'):
    """Compare risk distributions between latest and previous quarter."""

    latest_q, previous_q = get_latest_quarters(df, time_column)

    df_latest = df[df[time_column] == latest_q]
    df_previous = df[df[time_column] == previous_q]

    categories = ['accept', 'moderate', 'block']

    counts_previous = df_previous[risk_column].value_counts().reindex(categories, fill_value=0)
    counts_latest = df_latest[risk_column].value_counts().reindex(categories, fill_value=0)

    contingency_table = pd.DataFrame({previous_q: counts_previous, latest_q: counts_latest})

    chi2, p_value, dof, expected = chi2_contingency(contingency_table.T)

    pct_previous = (counts_previous / counts_previous.sum() * 100).round(2)
    pct_latest = (counts_latest / counts_latest.sum() * 100).round(2)

    summary = pd.DataFrame({
        f'{previous_q} Count': counts_previous,
        f'{previous_q} %': pct_previous,
        f'{latest_q} Count': counts_latest,
        f'{latest_q} %': pct_latest,
        'Diff %': (pct_latest - pct_previous).round(2)
    })

    print("=" * 60)
    print(f"RISK DISTRIBUTION COMPARISON: {previous_q} vs {latest_q}")
    print("=" * 60)
    print(f"\nSample sizes: {previous_q} = {len(df_previous)}, {latest_q} = {len(df_latest)}")
    print("\n" + summary.to_string())
    print("\n" + "-" * 60)
    print("CHI-SQUARE TEST FOR HOMOGENEITY")
    print("-" * 60)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print("-" * 60)

    alpha = 0.05
    if p_value < alpha:
        print(f"Result: SIGNIFICANT difference (p < {alpha})")
    else:
        print(f"Result: NO significant difference (p >= {alpha})")

    return {
        'chi2': chi2,
        'p_value': p_value,
        'summary': summary,
        'latest_quarter': latest_q,
        'previous_quarter': previous_q
    }


def plot_risk_comparison(df, risk_column='risk_category', time_column='time_'):
    """Visualize risk distribution comparison between latest and previous quarter."""

    latest_q, previous_q = get_latest_quarters(df, time_column)

    df_latest = df[df[time_column] == latest_q]
    df_previous = df[df[time_column] == previous_q]

    categories = ['accept', 'moderate', 'block']

    pct_previous = df_previous[risk_column].value_counts(normalize=True).reindex(categories, fill_value=0) * 100
    pct_latest = df_latest[risk_column].value_counts(normalize=True).reindex(categories, fill_value=0) * 100

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, pct_previous, width, label=previous_q, color='steelblue')
    bars2 = ax.bar(x + width/2, pct_latest, width, label=latest_q, color='coral')

    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Risk Category')
    ax.set_title(f'Risk Distribution Comparison: {previous_q} vs {latest_q}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f%%', padding=2)
    ax.bar_label(bars2, fmt='%.1f%%', padding=2)

    plt.tight_layout()
    plt.show()


# =============================================================================
# SAMPLE DATA FOR TESTING
# =============================================================================

def generate_sample_data():
    """Generate sample customer risk data across multiple quarters."""

    np.random.seed(42)

    quarters_config = {
        'Q1-2024': {'size': 800,  'probs': [0.60, 0.30, 0.10]},
        'Q2-2024': {'size': 850,  'probs': [0.58, 0.31, 0.11]},
        'Q3-2024': {'size': 900,  'probs': [0.55, 0.33, 0.12]},
        'Q4-2024': {'size': 950,  'probs': [0.52, 0.35, 0.13]},
        'Q1-2025': {'size': 1000, 'probs': [0.50, 0.36, 0.14]},
        'Q2-2025': {'size': 1100, 'probs': [0.48, 0.37, 0.15]},
        'Q3-2025': {'size': 1150, 'probs': [0.45, 0.38, 0.17]},
        'Q4-2025': {'size': 1200, 'probs': [0.42, 0.40, 0.18]},  # previous quarter
        'Q1-2026': {'size': 1300, 'probs': [0.38, 0.42, 0.20]},  # latest quarter
    }

    categories = ['accept', 'moderate', 'block']

    data_frames = []
    for quarter, config in quarters_config.items():
        df_quarter = pd.DataFrame({
            'customer_id': range(config['size']),
            'risk_category': np.random.choice(categories, size=config['size'], p=config['probs']),
            'time_': quarter
        })
        data_frames.append(df_quarter)

    return pd.concat(data_frames, ignore_index=True)


# =============================================================================
# RUN TEST
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()

    # Preview data
    print("Sample data preview:")
    print(df.head(10))
    print(f"\nTotal records: {len(df)}")
    print(f"Quarters available: {sorted(df['time_'].unique())}")
    print("\n")

    # Run statistical comparison
    results = compare_risk_distributions(df)

    # Generate plot
    plot_risk_comparison(df)
