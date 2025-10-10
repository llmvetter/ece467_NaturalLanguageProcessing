import structlog

from .config import load_settings
from .data import CorpusLoader
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
    log.info("Loading Training Corpus")
    full_corpus: CorpusLoader = CorpusLoader(
        labels_path=settings.data.train_labels_path,
    )
    full_corpus.load()

    if settings.training.mode == "eval":
        log.info("Running in eval mode")
        train_data, eval_data = full_corpus.split(
            val_ratio=settings.data.split,
        )
        log.info("Training data loaded", size=len(train_data))
        log.info("Validation data loaded", size=len(eval_data))

    elif settings.training.mode == "test":
        log.info("Running in test mode")

        train_data = full_corpus.articles
        log.info("Loading separate Test Corpus")
        test_corpus: CorpusLoader = CorpusLoader(
            labels_path=settings.data.test_data_path,
        )
        test_corpus.load()
        eval_data = test_corpus.articles
        log.info("Training data loaded", size=len(train_data))
        log.info("Test data loaded", size=len(eval_data))

    train_tokens: list[str] = [article.tokens for article in train_data]
    train_labels: list[str] = [article.label for article in train_data]
    val_tokens: list[str] = [article.tokens for article in eval_data]
    val_labels: list[str] = [article.label for article in eval_data]

    # Initialize the model
    vectorizer = Vectorizer()
    classifier = Classifier()

    # Fit the vectorizer
    tfidf_matrix = vectorizer.fit(train_tokens)
    log.info("Vectorizer fitted, Vocab Size", vocab=len(vectorizer.vocab))

    # Fit the classifier
    classifier.fit(
        vectors=tfidf_matrix,
        labels=train_labels,
    )
    log.info("Classifier fitted.")

    # Predict the val data
    val_vectors = vectorizer.transform(val_tokens)
    predictions = classifier(val_vectors)

    if settings.training.mode == "eval":
        evaluator = Evaluator(
            true_labels=val_labels,
            predictions=predictions,
        )
        accuracy = evaluator.calculate_accuracy()
        log.info("Eval Accuracy", metric="accuracy", value=accuracy)

        conf_matrix = evaluator.calculate_confusion_matrix()
        plot_confusion_matrix(
            labels=evaluator.label_names,
            conf_matrix=conf_matrix,
            accuracy=accuracy,
        )

    elif settings.training.mode == "test":
        # Load test dataset
        log.info(
            "Writing final predictions to file",
            output_path=settings.data.output_path,
        )

        output_lines = []
        for article, predicted_label in zip(eval_data, predictions):
            output_line = f"{article.rel_path} {predicted_label}\n"
            output_lines.append(output_line)

        try:
            with open(settings.data.output_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(output_lines)
            log.info("Prediction file successfully written.")

        except Exception as e:
            log.error("Failed to write output file", error=str(e))
