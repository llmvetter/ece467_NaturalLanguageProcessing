import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

@dataclass
class Article:
    """
    Data structure to hold the information and processing steps for a single article.
    """
    doc_id: str
    label: str
    raw_text: str
    tokens: List[str]


class CorpusLoader:
    def __init__(
        self,
        corpus_path: str,
        labels_path: str,
):
        """
        Initializes the loader with the root directory and NLTK tools.
        :param data_root: The root directory containing 'corpus1/train/'
        """
        self.corpus_path = Path(corpus_path)
        self.labels_path = Path(labels_path)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.articles: List[Article] = []

    def _load_labels(self) -> Dict[str, str]:
        """
        Loads the labels from the separate file into a dictionary mapping 
        article filenames to their labels (e.g., '03785.article': 'Cri').
        """
        labels = {}

        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:

                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 2:
                        full_path, label = parts
                        filename = Path(full_path).name
                        labels[filename] = label

        except FileNotFoundError:
            print(f"Error: Labels file not found at {self.labels_path}")
        return labels

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Performs the core text preprocessing steps.
        1. Clean: Remove excessive whitespaces/tabs.
        2. Tokenize: Break into words.
        3. Lowercase.
        4. Remove non-alphabetic tokens (punctuation/numbers).
        5. Remove stopwords.
        6. Stemming (simplest form of normalization).
        """
        # 1. Clean (Remove leading/trailing spaces and internal non-standard whitespace)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 2. & 3. Tokenize and Lowercase
        tokens = word_tokenize(text.lower())
        
        processed_tokens = []
        for token in tokens:
            # 4. Remove non-alphabetic tokens and 5. Stopword removal
            if token.isalpha() and token not in self.stop_words:
                # 6. Stemming
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)

        return processed_tokens

    def load_corpus(self):
        """
        Main method to load all articles, preprocess them, and store the results.
        :param articles_dir_name: Directory name (e.g., 'corpus1/train')
        :param labels_file_name: Labels file name (e.g., 'labels.txt')
        """
        
        # Ensure articles directory exists
        if not self.corpus_path.is_dir():
            print(f"Error: Article directory not found at {self.corpus_path}")
            return

        # 1. Load Labels
        labels_map = self._load_labels()
        if not labels_map:
            print("Cannot proceed without labels.")
            return

        for article_path in self.corpus_path.glob('*.article'):
            doc_id = article_path.name
            
            # Check if we have a label for this article
            if doc_id not in labels_map:
                print(f"Warning: Skipping {doc_id} - no label found.")
                continue

            # Load the raw text
            try:
                raw_text = article_path.read_text(encoding='latin-1')
            except Exception as e:
                print(f"Error reading {doc_id}: {e}")
                continue

            # Preprocess the text
            processed_tokens = self._preprocess_text(raw_text)

            # Create and store the Article object
            article_object = Article(
                doc_id=doc_id,
                label=labels_map[doc_id],
                raw_text=raw_text,
                tokens=processed_tokens
            )
            self.articles.append(article_object)

    def get_articles(self) -> List[Article]:
        """Returns the list of processed Article objects."""
        return self.articles
