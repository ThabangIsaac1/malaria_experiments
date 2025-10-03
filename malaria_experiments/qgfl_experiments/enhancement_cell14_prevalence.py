"""
Cell 14: Prevalence-Stratified Analysis
Extracted from 01_run_baseline.ipynb (lines 1054-1250)

This function will be added to train_baseline.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate


def run_prevalence_stratified_analysis(
    test_results: dict,
    class_names: dict,
    task: str,
    save_dir: Path,
    use_wandb: bool = False
) -> dict:
    """
    Analyze model performance across different parasitemia (infection density) levels.

    This is THE KEY METRIC from the QGFL paper:
    "QGFL achieves remarkable improvement in detecting infected cells in the
    clinically vital 1–3% parasitaemia range"

    Args:
        test_results: Dictionary from evaluator.evaluate_model('test')
        class_names: Dictionary of class IDs to names
        task: Task type ('binary', 'species', 'staging')
        save_dir: Directory to save visualization
        use_wandb: Whether to log to W&B

    Returns:
        Dictionary with stratified metrics for W&B logging
    """

    # Only run for binary classification task
    if task != 'binary':
        print(f"\nSkipping prevalence-stratified analysis (only for binary task, current: {task})")
        return {}

    print("\n" + "="*70)
    print("PREVALENCE-STRATIFIED ANALYSIS")
    print("="*70)
    print("Analyzing performance across parasitemia levels...")
    print("(This is the key metric for malaria detection quality)")

    # Get stratified results from evaluator
    if 'stratified' not in test_results:
        print("⚠️  No stratified results found in test_results")
        print("   Make sure ComprehensiveEvaluator calculated prevalence bins")
        return {}

    stratified = test_results['stratified']

    # Standard parasitemia bins used in malaria detection
    bins = ['0-1%', '1-3%', '3-5%', '>5%']

    # Verify all bins exist
    missing_bins = [b for b in bins if b not in stratified]
    if missing_bins:
        print(f"⚠️  Missing bins: {missing_bins}")
        available_bins = [b for b in bins if b in stratified]
        if not available_bins:
            print("   No stratified data available")
            return {}
        bins = available_bins

    # =====================================
    # 1. DISPLAY RESULTS TABLE
    # =====================================
    strat_data = []
    for bin_name in bins:
        stats = stratified[bin_name]
        strat_data.append([
            bin_name,
            f"{stats['mean_recall']:.3f}",
            f"{stats['std_recall']:.3f}",
            stats['count']
        ])

    print("\n" + tabulate(
        strat_data,
        headers=['Parasitemia Level', 'Mean Recall', 'Std Dev', 'N Images'],
        tablefmt='fancy_grid',
        numalign='right'
    ))

    # Clinical significance notes
    print("\nClinical Significance:")
    for bin_name in bins:
        recall = stratified[bin_name]['mean_recall']
        std = stratified[bin_name]['std_recall']
        count = stratified[bin_name]['count']

        clinical_note = ""
        if bin_name == '0-1%':
            clinical_note = " ← Ultra-low (hardest to detect, most critical)"
        elif bin_name == '1-3%':
            clinical_note = " ← CRITICAL RANGE (early detection, key metric)"
        elif bin_name == '3-5%':
            clinical_note = " ← Moderate (routine detection)"
        else:  # >5%
            clinical_note = " ← High (easier detection)"

        print(f"  {bin_name}: {recall:.3f} ± {std:.3f} (n={count}){clinical_note}")

    # =====================================
    # 2. CREATE VISUALIZATION
    # =====================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Across Parasitemia Levels', fontsize=14, fontweight='bold')

    recalls = [stratified[b]['mean_recall'] for b in bins]
    stds = [stratified[b]['std_recall'] for b in bins]
    counts = [stratified[b]['count'] for b in bins]

    # Color gradient: red (critical) → green (less critical)
    colors = ['#B71C1C', '#FF6F00', '#FDD835', '#43A047']

    # Subplot 1: Bar chart with error bars
    bars = []
    for i, (bin_name, recall, std, count) in enumerate(zip(bins, recalls, stds, counts)):
        bar = ax1.bar(
            i, recall,
            yerr=std if count > 0 else 0,
            capsize=5,
            color=colors[i] if i < len(colors) else '#666666',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        bars.append(bar)

    ax1.set_xticks(range(len(bins)))
    ax1.set_xticklabels(bins)
    ax1.set_xlabel('Parasitemia Level (%)', fontsize=12)
    ax1.set_ylabel('Mean Infected Cell Recall', fontsize=12)
    ax1.set_title('Recall by Parasitemia Level', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add clinical threshold line
    ax1.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(len(bins)-0.5, 0.82, 'Target: 0.8', fontsize=9, color='green')

    # Add sample counts on bars
    for i, (bar_container, count) in enumerate(zip(bars, counts)):
        if bar_container:
            bar = bar_container[0]
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2, height + 0.02,
                f'n={count}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

    # Subplot 2: Line plot with confidence intervals
    x_pos = np.arange(len(bins))

    ax2.plot(
        x_pos, recalls,
        'o-',
        markersize=10,
        linewidth=2.5,
        color='#D32F2F',
        markeredgecolor='black',
        markeredgewidth=1,
        label='Mean Recall'
    )

    # Error bars
    ax2.errorbar(
        x_pos, recalls,
        yerr=stds,
        fmt='none',
        ecolor='#D32F2F',
        alpha=0.3,
        capsize=5,
        capthick=2
    )

    # Confidence interval shading
    ax2.fill_between(
        x_pos,
        [max(0, r-s) for r, s in zip(recalls, stds)],
        [min(1, r+s) for r, s in zip(recalls, stds)],
        alpha=0.15,
        color='#D32F2F'
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bins)
    ax2.set_xlabel('Parasitemia Level (%)', fontsize=12)
    ax2.set_ylabel('Mean Infected Cell Recall', fontsize=12)
    ax2.set_title('Performance Trend Across Parasitemia Levels', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')

    # Add clinical threshold line
    ax2.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    # Save figure
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'prevalence_stratified_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")

    # Close to free memory
    plt.close(fig)

    # =====================================
    # 3. PREPARE RETURN DATA FOR W&B
    # =====================================
    stratified_summary = {
        'bins': {},
        'clinical_assessment': {},
        'figure_path': str(save_path)
    }

    # Per-bin metrics
    for bin_name in bins:
        stratified_summary['bins'][bin_name] = {
            'mean_recall': float(stratified[bin_name]['mean_recall']),
            'std_recall': float(stratified[bin_name]['std_recall']),
            'count': int(stratified[bin_name]['count'])
        }

    # Clinical assessments
    if '1-3%' in stratified and stratified['1-3%']['count'] > 0:
        critical_recall = stratified['1-3%']['mean_recall']
        stratified_summary['clinical_assessment']['critical_range_recall'] = float(critical_recall)
        stratified_summary['clinical_assessment']['meets_target'] = critical_recall >= 0.8

        if critical_recall < 0.5:
            status = "POOR - Fails to detect early infections"
        elif critical_recall < 0.7:
            status = "FAIR - Misses many early infections"
        elif critical_recall < 0.8:
            status = "GOOD - Close to clinical target"
        else:
            status = "EXCELLENT - Meets clinical requirements"

        stratified_summary['clinical_assessment']['status'] = status
        print(f"\nClinical Assessment (1-3% range): {status}")

    print("\n" + "="*70)

    # Log to W&B if requested
    if use_wandb:
        try:
            import wandb

            # Log metrics
            for bin_name in bins:
                wandb.log({
                    f'stratified/{bin_name.replace("%", "pct")}/recall': stratified[bin_name]['mean_recall'],
                    f'stratified/{bin_name.replace("%", "pct")}/std': stratified[bin_name]['std_recall'],
                    f'stratified/{bin_name.replace("%", "pct")}/count': stratified[bin_name]['count'],
                })

            # Log figure
            wandb.log({'stratified/performance_plot': wandb.Image(str(save_path))})

            print("✓ Logged to W&B")
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")

    return stratified_summary


if __name__ == '__main__':
    """Test the function with mock data"""
    print("Testing prevalence-stratified analysis function...")

    # Create mock test results
    mock_results = {
        'stratified': {
            '0-1%': {'mean_recall': 0.45, 'std_recall': 0.15, 'count': 25},
            '1-3%': {'mean_recall': 0.62, 'std_recall': 0.12, 'count': 38},
            '3-5%': {'mean_recall': 0.78, 'std_recall': 0.08, 'count': 15},
            '>5%': {'mean_recall': 0.85, 'std_recall': 0.06, 'count': 42}
        }
    }

    mock_class_names = {0: 'Uninfected', 1: 'Infected'}
    mock_save_dir = Path('./test_output')

    result = run_prevalence_stratified_analysis(
        test_results=mock_results,
        class_names=mock_class_names,
        task='binary',
        save_dir=mock_save_dir,
        use_wandb=False
    )

    print("\nReturned summary:")
    import json
    print(json.dumps(result, indent=2))
    print("\n✓ Test complete!")
