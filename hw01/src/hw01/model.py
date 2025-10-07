import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import numpy as np

from .data import CorpusLoader, Article


class Vectorizer:

    def __init__(self):

        self.vocab: Dict[str, int] = {}
        self.idf_vector: np.ndarray = np.array([])
        self.N: int = 0

    def _calculate_idf(self, token_list: list[str]):

        idf_list = []
        self.N = len(token_list)
        doc_frequency = defaultdict(int)
        
        for article in token_list:
            unique_per_doc = set(article)
            for term in unique_per_doc:
                doc_frequency[term] += 1
        
        for i, (term, df) in enumerate(doc_frequency.items()):

            self.vocab[term] = i
            
            # IDF: log(N / DF(t))
            # added 1 to the denominator (df) and numerator (N) for smoothing
            idf = math.log((self.N + 1) / (df + 1))
            idf_list.append(idf)

        self.idf_vector = np.array(idf_list, dtype=np.float64)

    def _calculate_tf(self, tokens: list[str]) -> Dict[int, float]:
        """
        TF = (Count of term t in document d) / (Total number of terms in document d)
        """
        tf_map = defaultdict(float)
        term_counts = Counter(tokens)
        
        for term, count in term_counts.items():
            if term in self.vocab:
                term_index = self.vocab[term]
                tf_map[term_index] = count / len(tokens)
                
        return tf_map

    def transform(self, token_list: list[str]) -> np.ndarray:
        """
        Converts a list of token lists into a dense NumPy TF-IDF matrix.
        TF-IDF = TF * IDF.
        """
            
        vocab_size = len(self.vocab)
        num_docs = len(token_list)

        # Initialize the dense TF-IDF matrix using NumPy
        tfidf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float64)

        for doc_idx, doc_tokens in enumerate(token_list):
            tf_map = self._calculate_tf(doc_tokens)
            
            # Populate the current row (document vector)
            for term_index, tf_score in tf_map.items():
                # TF-IDF = TF * IDF (using pre-calculated IDF vector)
                tfidf_score = tf_score * self.idf_vector[term_index]
                tfidf_matrix[doc_idx, term_index] = tfidf_score

        return tfidf_matrix

    def fit(self, token_list: list[str]) -> np.ndarray:
        self._calculate_idf(token_list)
        return self.transform(token_list)


class Classifier:

    def __init__(self):
        self.centroids: Dict[str, np.ndarray] = {}
        
    def fit(self, vectors: np.ndarray, labels: List[str]):

        unique_labels = sorted(list(set(labels)))
        labels_np = np.array(labels)
        vocab_size = vectors.shape[1]

        for label in unique_labels:

            mask = (labels_np == label)
            class_vectors = vectors[mask, :]
            num_docs = class_vectors.shape[0]
            centroid = np.mean(class_vectors, axis=0)
            self.centroids[label] = centroid

        self.centroid_labels = labels
        self.centroid_matrix = np.array([self.centroids[label] for label in labels])
        self.centroid_norms = np.linalg.norm(self.centroid_matrix, axis=1, keepdims=True)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        dot_products = x @ self.centroid_matrix.T
        denominators = x_norm @ self.centroid_norms.T
        similarity_matrix = dot_products / denominators
        best_centroid_indices = np.argmax(similarity_matrix, axis=1)
        predictions = [self.centroid_labels[i] for i in best_centroid_indices]
        return predictions