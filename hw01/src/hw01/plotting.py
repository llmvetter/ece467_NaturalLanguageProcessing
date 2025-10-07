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
        annot=True,              # Show numerical values
        fmt='d',                 # Format as integers
        cmap='Blues',            # Color scheme
        xticklabels=labels,      # Set predicted labels on X-axis
        yticklabels=labels       # Set true labels on Y-axis
    )

    plt.tick_params(
        axis='x',          # Apply to the x-axis
        bottom=False,      # Hide ticks at the bottom
        labelbottom=False, # Hide labels at the bottom
        top=True,          # Show ticks at the top
        labeltop=True      # Show labels at the top
    )
    
    # Set labels and title
    plt.title(f"Rocchio Classifier Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Ensure all labels fit within the figure area
    plt.tight_layout()
    
    # Save the plot to the current working directory
    save_path = Path.cwd() / filename
    plt.savefig(save_path)
    plt.close()