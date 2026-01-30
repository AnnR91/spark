import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
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


def compare_risk_distributions(df, risk_column='risk_category', time_column='time_', count_column='count', stability_threshold=0.1):
    """
    Compare risk distributions between latest and previous quarter using Jensen-Shannon divergence.

    Parameters:
    - stability_threshold: JS divergence below this value is considered stable (default 0.1)
      JS divergence ranges from 0 (identical) to 1 (maximally different)
      Common thresholds: <0.05 very stable, <0.1 stable, <0.2 moderate change, >=0.2 significant change
    """

    counts_previous, counts_latest, previous_q, latest_q = get_aggregated_counts(
        df, time_column, risk_column, count_column
    )

    # Calculate proportions for JS divergence
    total_previous = counts_previous.sum()
    total_latest = counts_latest.sum()

    prop_previous = counts_previous / total_previous
    prop_latest = counts_latest / total_latest

    # Jensen-Shannon divergence (returns distance, not divergence - already sqrt of JS divergence)
    js_distance = jensenshannon(prop_previous, prop_latest, base=2)

    # JS divergence is the square of JS distance
    js_divergence = js_distance ** 2

    # Stability score: 1 - JS divergence (1 = identical, 0 = maximally different)
    stability_score = 1 - js_divergence

    pct_previous = (prop_previous * 100).round(2)
    pct_latest = (prop_latest * 100).round(2)

    summary = pd.DataFrame({
        f'{previous_q} Count': counts_previous,
        f'{previous_q} %': pct_previous,
        f'{latest_q} Count': counts_latest,
        f'{latest_q} %': pct_latest,
        'Diff %': (pct_latest - pct_previous).round(2)
    })

    is_stable = js_divergence < stability_threshold

    print("=" * 60)
    print(f"RISK DISTRIBUTION COMPARISON: {previous_q} vs {latest_q}")
    print("=" * 60)
    print(f"\nTotal counts: {previous_q} = {total_previous}, {latest_q} = {total_latest}")
    print("\n" + summary.to_string())
    print("\n" + "-" * 60)
    print("JENSEN-SHANNON STABILITY ANALYSIS")
    print("-" * 60)
    print(f"JS Divergence: {js_divergence:.6f}")
    print(f"JS Distance: {js_distance:.6f}")
    print(f"Stability Score: {stability_score:.4f} (1 = identical, 0 = max different)")
    print("-" * 60)

    if js_divergence < 0.05:
        interpretation = "VERY STABLE - distributions nearly identical"
    elif js_divergence < 0.1:
        interpretation = "STABLE - minor differences"
    elif js_divergence < 0.2:
        interpretation = "MODERATE CHANGE - noticeable shift"
    else:
        interpretation = "SIGNIFICANT CHANGE - major distribution shift"

    print(f"Result: {interpretation}")

    # Create export-ready DataFrame with all results
    results_df = summary.copy()
    results_df['js_divergence'] = js_divergence
    results_df['js_distance'] = js_distance
    results_df['stability_score'] = stability_score
    results_df['is_stable'] = is_stable
    results_df['interpretation'] = interpretation
    results_df['comparison'] = f"{previous_q} vs {latest_q}"
    results_df['total_previous'] = total_previous
    results_df['total_latest'] = total_latest
    results_df.index.name = 'risk_category'
    results_df = results_df.reset_index()

    return {
        'js_divergence': js_divergence,
        'js_distance': js_distance,
        'stability_score': stability_score,
        'is_stable': is_stable,
        'interpretation': interpretation,
        'summary': summary,
        'latest_quarter': latest_q,
        'previous_quarter': previous_q,
        'results_df': results_df
    }


def plot_risk_comparison(df, risk_column='risk_category', time_column='time_', count_column='count'):
    """Visualize risk distribution comparison between latest and previous quarter."""

    counts_previous, counts_latest, previous_q, latest_q = get_aggregated_counts(
        df, time_column, risk_column, count_column
    )

    categories = ['accept', 'moderate', 'block']

    pct_previous = (counts_previous / counts_previous.sum() * 100).reindex(categories, fill_value=0)
    pct_latest = (counts_latest / counts_latest.sum() * 100).reindex(categories, fill_value=0)

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

    print("\n" + "=" * 60)
    print("EXPORT-READY DATAFRAME:")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save to Excel
    output_path = 'risk_comparison_js_results.xlsx'
    results_df.to_excel(output_path, index=False, sheet_name='Risk Comparison')
    print(f"\nResults saved to: {output_path}")

    # Access individual metrics
    print(f"\nKey metrics:")
    print(f"  Stability Score: {results['stability_score']:.4f}")
    print(f"  JS Divergence: {results['js_divergence']:.6f}")
    print(f"  Is Stable: {results['is_stable']}")

    # Generate plot
    plot_risk_comparison(df)
