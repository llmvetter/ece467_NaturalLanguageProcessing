import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from itertools import product

class Evaluator:

    def __init__(
        self,
        true_labels: List[str],
        predictions: List[str],
    ):
        if len(true_labels) != len(predictions):
            raise ValueError("True labels and predictions must have the same length.")
            
        self.true_labels = np.array(true_labels)
        self.predictions = np.array(predictions)
        all_labels = set(true_labels) | set(predictions)
        self.label_names = sorted(list(all_labels))
        self.label_to_index = {label: i for i, label in enumerate(self.label_names)}
        self.num_classes = len(self.label_names)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
    def calculate_accuracy(self) -> float:
        correct_predictions = np.sum(self.true_labels == self.predictions)
        total_predictions = len(self.true_labels)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def calculate_confusion_matrix(self) -> np.ndarray:

        for true_label, pred_label in zip(self.true_labels, self.predictions):
            true_idx = self.label_to_index.get(true_label)
            pred_idx = self.label_to_index.get(pred_label)

            if true_idx is not None and pred_idx is not None:
                self.confusion_matrix[true_idx, pred_idx] += 1

        return self.confusion_matrix