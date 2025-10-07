import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from itertools import product

class Evaluator:
    """
    Calculates classification ccuracy and confusion matrix.
    """
    def __init__(self, true_labels: List[str], predictions: List[str]):
        if len(true_labels) != len(predictions):
            raise ValueError("True labels and predictions must have the same length.")
            
        self.true_labels = np.array(true_labels)
        self.predictions = np.array(predictions)
        
        # Dynamically determine all unique labels
        all_labels = set(true_labels) | set(predictions)
        self.label_names = sorted(list(all_labels))
        self.label_to_index = {label: i for i, label in enumerate(self.label_names)}
        self.num_classes = len(self.label_names)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
    def calculate_accuracy(self) -> float:
        """
        Calculates the overall classification accuracy.
        Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
        """
        correct_predictions = np.sum(self.true_labels == self.predictions)
        total_predictions = len(self.true_labels)
        
        if total_predictions == 0:
            return 0.0
            
        accuracy = correct_predictions / total_predictions
        return accuracy

    def calculate_confusion_matrix(self) -> np.ndarray:
        """
        Calculates the confusion matrix.
        The matrix is indexed such that:
        - Rows (i): True Labels
        - Columns (j): Predicted Labels
        C[i, j] is the number of instances with true label i predicted as label j.
        """
        
        # Reset the matrix
        self.confusion_matrix.fill(0) 

        # Efficiently populate the matrix
        for true_label, pred_label in zip(self.true_labels, self.predictions):
            true_idx = self.label_to_index.get(true_label)
            pred_idx = self.label_to_index.get(pred_label)
            
            # This check is technically redundant if all labels are in self.label_to_index,
            # but is good practice if any labels slipped through the initialization.
            if true_idx is not None and pred_idx is not None:
                self.confusion_matrix[true_idx, pred_idx] += 1
                
        return self.confusion_matrix

    def get_labeled_confusion_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Returns the list of label names (in matrix order) and the computed matrix.
        """
        # Ensure the matrix is calculated before returning
        if np.sum(self.confusion_matrix) == 0:
             self.calculate_confusion_matrix()
             
        return self.label_names, self.confusion_matrix