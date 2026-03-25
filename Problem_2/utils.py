import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(results, out_path='Problem_2/model_comparison.png'):
    """
    Plot model comparison: novelty and diversity side-by-side.
    
    Args:
        results: dict with model names as keys and (novelty, diversity) tuples as values
        out_path: path to save the plot
    """
    models = list(results.keys())
    novelties = [results[m][0] for m in models]
    diversities = [results[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, novelties, width, label='Novelty Rate (%)', color='tab:blue')
    bars2 = ax.bar(x + width/2, diversities, width, label='Diversity (%)', color='tab:orange')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Model Comparison: Novelty vs Diversity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")





if __name__ == '__main__':
    # Example data from your runs
    results = {
        'vanilla': (94.15, 91.70),
        'blstm': (76.38, 80.46),
        'attention': (89.70, 91.80)
    }
    
    param_counts = {
        'vanilla': 31032,
        'blstm': 204856,
        'attention': 137272
    }

    # Generate all plots
    plot_model_comparison(results)
    print("\n✓ All plots generated successfully!")
