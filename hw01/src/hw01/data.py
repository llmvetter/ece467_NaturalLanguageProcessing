import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        labels_path: str,
):

        self.labels_path = Path(labels_path)
        self.base_path = self.labels_path.parent
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.articles: list[Article] = []
        self.train_articles: list[Article] = []
        self.val_arcticles: list[Article] = []

    def _load_labels_and_paths(self) -> List[Tuple[str, str, str]]:
        """
        Loads relative paths and labels, returning a list of 
        (relative_path, doc_id, label) tuples.
        """
        article_info = []
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 1: 
                    rel_path = parts[0]
                    label = parts[1]
                    doc_id = Path(rel_path).name
                    article_info.append((rel_path, doc_id, label))
        return article_info

    def _preprocess(self, text: str) -> List[str]:
        """
        Performs the core text preprocessing steps.
        1. Clean: Remove excessive whitespaces/tabs.
        2. Tokenize: Break into words.
        3. Lowercase.
        4. Remove non-alphabetic tokens (punctuation/numbers).
        5. Remove stopwords.
        6. Lemmatization (for normalization).
        """
        text = re.sub(r'\s+', ' ', text.strip())
        tokens = word_tokenize(text.lower())
        
        processed_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized_token)
        return processed_tokens

    def load(self):

        article_info = self._load_labels_and_paths()

        for rel_path, doc_id, label in article_info:
            
            article_abs_path = self.base_path / rel_path
            raw_text = article_abs_path.read_text(encoding='latin-1')
            processed_tokens = self._preprocess(raw_text)
            article_object = Article(
                doc_id=doc_id,
                label=label,
                raw_text=raw_text,
                tokens=processed_tokens
            )
            self.articles.append(article_object)

    def split(
        self,
        val_ratio: float = 0.2,
        seed: int = None,
    ) -> Tuple[List[Article], List[Article]]:

        articles_shuffled = self.articles.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(articles_shuffled)
        val_size = int(len(articles_shuffled) * val_ratio)
        self.val_articles = articles_shuffled[:val_size]
        self.train_articles = articles_shuffled[val_size:]
        
        return self.train_articles, self.val_articles