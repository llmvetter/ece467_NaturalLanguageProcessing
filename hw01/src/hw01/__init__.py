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

    # Load dataset
    log.info("Loading Trainnig Corpus")
    train_corpus: CorpusLoader = CorpusLoader(
        labels_path=settings.data.train_labels_path,
    )
    train_corpus.load()

    # Split dataset
    train_data, val_data = train_corpus.split(
        val_ratio = settings.data.split,
    )
    if settings.training.mode == "test":
        train_data = train_corpus.articles

    train_tokens: list[str] = [article.tokens for article in train_data]
    train_labels: list[str] = [article.label for article in train_data]
    val_tokens: list[str] = [article.tokens for article in val_data]
    val_labels: list[str] = [article.label for article in val_data]

    log.info("Training data loaded", size = len(train_data))
    log.info("Validation data loaded", size = len(val_data))

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

    # Predict the val data
    val_vectors = vectorizer.transform(val_tokens)
    predictions = classifier(val_vectors)
    evaluator = Evaluator(
        true_labels = val_labels,
        predictions = predictions,
    )

    accuracy = evaluator.calculate_accuracy()
    log.info("Eval Accuracy", metric="accuracy", value=accuracy)

    conf_matrix = evaluator.calculate_confusion_matrix()
    plot_confusion_matrix(
        labels=evaluator.label_names,
        conf_matrix=conf_matrix,
        accuracy=accuracy,
    )

    if settings.training.mode == "test":

         # Load dataset
        log.info("Loading Test Corpus")
        test_corpus: CorpusLoader = CorpusLoader(
            labels_path=settings.data.test_labels_path,
        )
        test_corpus.load()
        test_data = test_corpus.articles
        test_tokens: list[str] = [article.tokens for article in test_data]
        test_labels: list[str] = [article.label for article in test_data]

        log.info("Test data loaded", size = len(train_data))

        test_vectors = vectorizer.transform(test_tokens)
        predictions = classifier(test_vectors)
        evaluator = Evaluator(
            true_labels = test_labels,
            predictions = predictions,
        )
        accuracy = evaluator.calculate_accuracy()
        log.info("Test Accuracy", metric="accuracy", value=accuracy)

        # write to file