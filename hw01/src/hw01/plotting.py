import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from pathlib import Path

def plot_confusion_matrix(
    labels: List[str],
    conf_matrix: np.ndarray,
    accuracy: float,
    filename: str = "confusion_matrix.png",
    figsize: Tuple[int, int] = (10, 8),
):

    plt.figure(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.tick_params(
        axis='x',
        bottom=False,
        labelbottom=False,
        top=True,
        labeltop=True,
    )
    
    plt.title(f"Rocchio Classifier Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    save_path = Path.cwd() / filename
    plt.savefig(save_path)
    plt.close()