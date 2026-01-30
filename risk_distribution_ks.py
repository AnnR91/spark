import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
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


def get_aggregated_counts(df, time_column='time_', risk_column='risk_category', count_column='count'):
    """Aggregate counts by quarter and risk category."""

    latest_q, previous_q = get_latest_quarters(df, time_column)

    df_latest = df[df[time_column] == latest_q]
    df_previous = df[df[time_column] == previous_q]

    categories = ['accept', 'moderate', 'block']

    counts_previous = df_previous.groupby(risk_column)[count_column].sum().reindex(categories, fill_value=0)
    counts_latest = df_latest.groupby(risk_column)[count_column].sum().reindex(categories, fill_value=0)

    return counts_previous, counts_latest, previous_q, latest_q


def expand_counts_to_samples(counts, categories):
    """
    Expand aggregated counts to individual samples for KS test.
    Maps categories to numeric values: accept=1, moderate=2, block=3
    """
    category_map = {'accept': 1, 'moderate': 2, 'block': 3}
    samples = []
    for cat in categories:
        count = counts[cat]
        samples.extend([category_map[cat]] * count)
    return np.array(samples)


def compare_risk_distributions(df, risk_column='risk_category', time_column='time_', count_column='count'):
    """
    Compare risk distributions between latest and previous quarter using Kolmogorov-Smirnov test.

    Note: KS test is designed for continuous distributions. For categorical data (like risk categories),
    it tests whether the cumulative distributions differ. The categories are mapped to ordinal values:
    accept=1, moderate=2, block=3 (assuming risk severity order).
    """

    counts_previous, counts_latest, previous_q, latest_q = get_aggregated_counts(
        df, time_column, risk_column, count_column
    )

    categories = ['accept', 'moderate', 'block']

    # Expand counts to samples for KS test
    sample_previous = expand_counts_to_samples(counts_previous, categories)
    sample_latest = expand_counts_to_samples(counts_latest, categories)

    # Two-sample Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(sample_previous, sample_latest)

    total_previous = counts_previous.sum()
    total_latest = counts_latest.sum()

    pct_previous = (counts_previous / total_previous * 100).round(2)
    pct_latest = (counts_latest / total_latest * 100).round(2)

    # Calculate cumulative distributions
    cumulative_previous = (counts_previous.cumsum() / total_previous * 100).round(2)
    cumulative_latest = (counts_latest.cumsum() / total_latest * 100).round(2)

    summary = pd.DataFrame({
        f'{previous_q} Count': counts_previous,
        f'{previous_q} %': pct_previous,
        f'{previous_q} Cumul %': cumulative_previous,
        f'{latest_q} Count': counts_latest,
        f'{latest_q} %': pct_latest,
        f'{latest_q} Cumul %': cumulative_latest,
        'Diff %': (pct_latest - pct_previous).round(2)
    })

    print("=" * 70)
    print(f"RISK DISTRIBUTION COMPARISON: {previous_q} vs {latest_q}")
    print("=" * 70)
    print(f"\nTotal counts: {previous_q} = {total_previous}, {latest_q} = {total_latest}")
    print("\n" + summary.to_string())
    print("\n" + "-" * 70)
    print("KOLMOGOROV-SMIRNOV TEST")
    print("-" * 70)
    print(f"KS Statistic: {ks_statistic:.6f}")
    print(f"P-value: {p_value:.6f}")
    print("-" * 70)

    alpha = 0.05
    is_significant = p_value < alpha
    if is_significant:
        print(f"Result: SIGNIFICANT difference (p < {alpha})")
        interpretation = "SIGNIFICANT - distributions differ"
    else:
        print(f"Result: NO significant difference (p >= {alpha})")
        interpretation = "NOT SIGNIFICANT - distributions similar"

    print("\nNote: KS test measures max difference between cumulative distributions.")
    print("      Categories mapped to ordinal scale: accept=1, moderate=2, block=3")

    # Create export-ready DataFrame with all results
    results_df = summary.copy()
    results_df['ks_statistic'] = ks_statistic
    results_df['p_value'] = p_value
    results_df['is_significant'] = is_significant
    results_df['interpretation'] = interpretation
    results_df['comparison'] = f"{previous_q} vs {latest_q}"
    results_df['total_previous'] = total_previous
    results_df['total_latest'] = total_latest
    results_df.index.name = 'risk_category'
    results_df = results_df.reset_index()

    return {
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'interpretation': interpretation,
        'summary': summary,
        'latest_quarter': latest_q,
        'previous_quarter': previous_q,
        'results_df': results_df
    }


def plot_risk_comparison(df, risk_column='risk_category', time_column='time_', count_column='count'):
    """Visualize risk distribution comparison with both bar chart and cumulative distribution."""

    counts_previous, counts_latest, previous_q, latest_q = get_aggregated_counts(
        df, time_column, risk_column, count_column
    )

    categories = ['accept', 'moderate', 'block']

    pct_previous = (counts_previous / counts_previous.sum() * 100).reindex(categories, fill_value=0)
    pct_latest = (counts_latest / counts_latest.sum() * 100).reindex(categories, fill_value=0)

    cumul_previous = pct_previous.cumsum()
    cumul_latest = pct_latest.cumsum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pct_previous, width, label=previous_q, color='steelblue')
    bars2 = ax1.bar(x + width/2, pct_latest, width, label=latest_q, color='coral')

    ax1.set_ylabel('Percentage (%)')
    ax1.set_xlabel('Risk Category')
    ax1.set_title(f'Distribution Comparison: {previous_q} vs {latest_q}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.1f%%', padding=2)
    ax1.bar_label(bars2, fmt='%.1f%%', padding=2)

    # Cumulative distribution (step plot for KS visualization)
    x_cumul = np.arange(len(categories))

    ax2.step(x_cumul, cumul_previous, where='mid', label=previous_q, color='steelblue', linewidth=2)
    ax2.step(x_cumul, cumul_latest, where='mid', label=latest_q, color='coral', linewidth=2)
    ax2.scatter(x_cumul, cumul_previous, color='steelblue', s=50, zorder=5)
    ax2.scatter(x_cumul, cumul_latest, color='coral', s=50, zorder=5)

    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_xlabel('Risk Category')
    ax2.set_title(f'Cumulative Distribution (KS Test): {previous_q} vs {latest_q}')
    ax2.set_xticks(x_cumul)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# =============================================================================
# SAMPLE DATA FOR TESTING
# =============================================================================

def generate_sample_data():
    """Generate sample aggregated customer risk data across multiple quarters."""

    np.random.seed(42)

    group_1_values = ['region_A', 'region_B', 'region_C']
    group_2_values = ['segment_1', 'segment_2']
    group_3_values = ['channel_X', 'channel_Y', 'channel_Z']
    categories = ['accept', 'moderate', 'block']

    quarters_config = {
        'Q1-2024': [0.60, 0.30, 0.10],
        'Q2-2024': [0.58, 0.31, 0.11],
        'Q3-2024': [0.55, 0.33, 0.12],
        'Q4-2024': [0.52, 0.35, 0.13],
        'Q1-2025': [0.50, 0.36, 0.14],
        'Q2-2025': [0.48, 0.37, 0.15],
        'Q3-2025': [0.45, 0.38, 0.17],
        'Q4-2025': [0.42, 0.40, 0.18],
        'Q1-2026': [0.38, 0.42, 0.20],
    }

    rows = []
    for quarter, probs in quarters_config.items():
        for g1 in group_1_values:
            for g2 in group_2_values:
                for g3 in group_3_values:
                    base_count = np.random.randint(50, 200)
                    for i, risk_cat in enumerate(categories):
                        count = int(base_count * probs[i] * np.random.uniform(0.8, 1.2))
                        rows.append({
                            'group_1': g1,
                            'group_2': g2,
                            'group_3': g3,
                            'time_': quarter,
                            'risk_category': risk_cat,
                            'count': count
                        })

    return pd.DataFrame(rows)


# =============================================================================
# RUN TEST
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()

    # Preview data
    print("Sample data preview:")
    print(df.head(15))
    print(f"\nTotal rows: {len(df)}")
    print(f"Quarters available: {sorted(df['time_'].unique())}")
    print(f"Total customer count: {df['count'].sum()}")
    print("\n")

    # Run statistical comparison
    results = compare_risk_distributions(df)

    # Get export-ready DataFrame
    results_df = results['results_df']

    print("\n" + "=" * 70)
    print("EXPORT-READY DATAFRAME:")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save to Excel
    output_path = 'risk_comparison_ks_results.xlsx'
    results_df.to_excel(output_path, index=False, sheet_name='Risk Comparison')
    print(f"\nResults saved to: {output_path}")

    # Access individual metrics
    print(f"\nKey metrics:")
    print(f"  KS Statistic: {results['ks_statistic']:.6f}")
    print(f"  P-value: {results['p_value']:.6f}")
    print(f"  Is Significant: {results['is_significant']}")

    # Generate plot
    plot_risk_comparison(df)
