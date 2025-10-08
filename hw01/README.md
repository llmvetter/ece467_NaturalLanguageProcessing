# Conventional Text Categorization Project (Fall 2024) 📚

## Overview
This repository contains the implementation of a **Rocchio Tf-IDF Classifier**.

The implementation was done in python. It was developed using the WSL Integration (Ubuntu) on Windows 11. The required packages are defined in the pyproject.toml

## Setup

This repository makes use of NLTK and will need specific library components to be downloaded locally.
This includes: wordnet, stopwords, punkt, punkt_tab, averaged_perceptron_tagger_en
These need to be installed seperately into the virtual environment the repository is run in.

## Running the code
First, set the parameters in the `config.py` file. Specifically, provide:


1.  **Training File:** Path to file containing the relative paths to the training documents alongside their respective labels (e.g., `corpusX_train.labels`).
2.  **Testing File:** Path to file containing the relatice paths to the testing documents. No labels. (e.g., `corpusX_test.list`).
3.  **Output File:** Name for the .txt file to store the predicted labels.

After setting the parameters, run the code by:

```bash
just run hw01
```

## Impementation Details

This repo implements the classic Rocchia TF-IDF text classification algorithm. The system can run in either `eval` mode or `test` mode.

When run in `eval` mode, the training data will be split into training and validation data (defaulting to a 80/20 split). This was used to tune the algorithm with respect to the validation set for each provided training dataset.

The system uses the basic NLTK tokenizer which tokonizes the training and test data in the same way. The following text preprocessing techniques are applied in order:

1. **Cleaning** Removes excessive whitespaces and tabs
2. **Tokenize** Simple tokenization, breaking the text down into words (unigrams)
3. **Lowercase** Normalization step, lowercases all the tokens.
4. Remove non-alphabetic tokens (punctuation/numbers)
5. Remove stopwords using NLTK stopwords for english
6. **Lemmatization** Further normalization step. Uses NLTK WordNetLemmatizer and POS specific lemmatization.

The system uses the TF-IDF weighting scheme to create document vectors.
During the tuning phase, the system was primarily evaluated on the validation sets for each corpus (which are dynamically created by splitting the training corpus). Experiments were run using the PorterStemmer, Using Bigrams and using Bigrams in combination with Mimimum word-counts per document as well as Maximum Vocabulary size (as the Bigram bloat up the vocab size significantly which leads to excessively large vectors).
