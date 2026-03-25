import matplotlib.pyplot as plt


def plot_training_loss(epochs, losses, out_path='Problem_1/training_loss.png'):
    """Plot training loss curve for word2vec training."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', color='tab:blue')
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Example usage with sample values
    example_epochs = [1, 2, 3, 4, 5]
    example_losses = [36.3448, 22.9155, 21.6787, 20.9501, 20.6115]
    out_path = 'Problem_1/SkipGram_training_loss.png'
    plot_training_loss(example_epochs, example_losses, out_path)
    print(f"Saved loss plot to {out_path}")