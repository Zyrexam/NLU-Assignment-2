import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(epochs, losses, model_name='Model', out_path='Problem_1/training_loss.png'):
    """Plot training loss curve for word2vec training."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', color='tab:blue', linewidth=2, markersize=8)
    plt.title(f'{model_name} Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(skipgram_losses, cbow_losses, out_path='Problem_1/word2vec_comparison.png'):
    """
    Plot Skip-gram vs CBOW training loss comparison.
    
    Args:
        skipgram_losses: list of Skip-gram epoch losses
        cbow_losses: list of CBOW epoch losses
        out_path: path to save the plot
    """
    epochs = list(range(1, len(skipgram_losses) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, skipgram_losses, marker='o', label='Skip-gram', linewidth=2.5, markersize=8, color='tab:blue')
    plt.plot(epochs, cbow_losses, marker='s', label='CBOW', linewidth=2.5, markersize=8, color='tab:orange')
    
    plt.title('Word2Vec Models: Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Skip-gram loss data
    sg_epochs = [1, 2, 3, 4, 5]
    sg_losses = [36.3448, 22.9155, 21.6787, 20.9501, 20.6115]
    
    # CBOW loss data
    cbow_epochs = [1, 2, 3, 4, 5]
    cbow_losses = [7.7699, 7.2214, 6.3830, 5.9539, 5.6693]
    
    # Generate individual plots
    plot_training_loss(sg_epochs, sg_losses, 'Skip-gram', 'Problem_1/skipgram_training_loss.png')
    plot_training_loss(cbow_epochs, cbow_losses, 'CBOW', 'Problem_1/cbow_training_loss.png')
    
    # Generate comparison plot
    plot_model_comparison(sg_losses, cbow_losses, 'Problem_1/word2vec_comparison.png')
    
    print("✓ Saved: Problem_1/skipgram_training_loss.png")
    print("✓ Saved: Problem_1/cbow_training_loss.png")
    print("✓ Saved: Problem_1/word2vec_comparison.png")