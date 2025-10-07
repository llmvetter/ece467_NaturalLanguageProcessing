import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import CorpusLoader, Article
from .logging import configure_logging
from .model import Vectorizer, Classifier
from .eval import Evaluator
from .plotting import plot_confusion_matrix


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # prompt loading destination
    log.info("Loading Text Corpus")
    train_corpus: CorpusLoader = CorpusLoader(
        corpus_path=settings.data.train_corpus_path,
        labels_path=settings.data.train_labels_path,
    )
    train_corpus.load_corpus()
    train_data: list[Article] = train_corpus.articles
    train_tokens: list[str] = [article.tokens for article in train_data]
    train_labels: list[str] = [article.label for article in train_data]
    log.info("Type train_tokens", type = type(train_tokens))
    log.info("Tyoe train_labels", type = type(train_labels))

    # Initialize the model
    vectorizer = Vectorizer()
    classifier = Classifier()

    # Fit the vectorizer
    tfidf_matrix = vectorizer.fit(train_tokens)
    log.info("Vectorizer fitted.", matrix_shape = type(tfidf_matrix))
    log.info("Vectorizer Vocab Size", vocab = len(vectorizer.vocab))

    # Fit the classifier
    classifier.fit(
        vectors = tfidf_matrix,
        labels = train_labels,
    )
    log.info("Classifier fitted.")

    # Load the val data
    val_corpus = CorpusLoader(
        corpus_path=settings.data.train_corpus_path,
        labels_path=settings.data.train_labels_path,
    )
    val_corpus.load_corpus()
    val_data: list[Article] = val_corpus.articles
    val_tokens: list[str] = [article.tokens for article in val_data]
    val_labels: list[str] = [article.label for article in val_data]
    log.info("Validation Data loaded", n_articles=len(val_data))

    # Predict the val data
    val_vectors = vectorizer.transform(val_tokens)
    predictions = classifier(val_vectors)
    evaluator = Evaluator(
        true_labels = val_labels,
        predictions = predictions,
    )

    accuracy = evaluator.calculate_accuracy()
    labels, conf_matrix = evaluator.get_labeled_confusion_matrix()
    conf_matrix_list = conf_matrix.tolist()
    log.info("Model Accuracy", metric="accuracy", value=accuracy)

    plot_confusion_matrix(
        labels=labels,
        conf_matrix=conf_matrix,
        accuracy=accuracy,
    )

