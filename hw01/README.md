# Conventional Text Categorization Project (Fall 2024) 📚

## Overview
This repository contains the implementation of a **conventional text categorization system** developed for **Natural Language Processing (NLP) - Fall 2024**.

The system utilizes one of the classic, conventional machine learning approaches for text classification (e.g., Naïve Bayes, k-NN, etc.). The **core algorithm's logic is implemented from scratch**, adhering strictly to the constraint of *not* using pre-existing routines for word statistics, text categorization, or machine learning from any library (including NLTK, outside of basic utility functions like tokenization, stemming, or lemmatization).

## Features
* **Custom Implementation:** Core machine learning method is self-implemented.
* **Data Handling:** Trains on a user-specified labeled document list and predicts categories for documents in a user-specified unlabeled test list.
* **Output:** Generates a predictions file listing test documents with their assigned categories.
* **Corpora:** Tested on three provided corpora:
    1.  **News Documents (5 categories):** Str, Pol, Dis, Cri, Oth.
    2.  **Image Captions (2 categories):** O (Outdoor), I (Indoor).
    3.  **News Documents (6 categories):** Wor, USN, Sci, Fin, Spo, Ent.
* **Evaluation:** Incorporates k-fold cross-validation or a tuning set approach for corpora lacking provided test labels.

## Usage
The program prompts the user for the following inputs:
1.  **Training File:** Path to the file containing relative paths and true labels (e.g., `corpusX_train.labels`).
2.  **Testing File:** Path to the file containing relative paths of documents to be classified (e.g., `corpusX_test.list`).
3.  **Output File:** Name for the file to store the predicted labels.