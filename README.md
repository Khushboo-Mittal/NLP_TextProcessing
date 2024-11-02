# NLP-TextProcessing

This project demonstrates text processing techniques applied to Twitter data for sentiment analysis. The pipeline involves preprocessing raw text data and applying various natural language processing (NLP) techniques such as stemming, lemmatization, tokenization, vectorization, and word embedding. The final goal is to build a logistic regression model to predict sentiment categories (Positive, Negative, Neutral, or Irrelevant).

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
  - [Tokenization](#tokenization)
  - [Removing Stopwords](#removing-stopwords)
  - [Stemming and Lemmatization](#stemming-and-lemmatization)
- [Feature Extraction](#feature-extraction)
  - [Count Vectorization](#count-vectorization)
  - [TF-IDF Vectorization](#tf-idf-vectorization)
- [Word Embeddings](#word-embeddings)
  - [Word2Vec](#word2vec)
  - [GloVe](#glove)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Dependencies](#dependencies)

## Project Overview

Using a dataset of Twitter posts, this project preprocesses and transforms text data into a format suitable for machine learning. Techniques like tokenization, stopword removal, stemming, lemmatization, and word embedding are used to extract meaningful features from text. Logistic Regression is then employed to classify tweet sentiment.

## Data Preprocessing

### Tokenization
Tokenization splits the text into individual words (tokens), which are essential for subsequent analysis.

### Removing Stopwords
Stopwords, such as 'the', 'is', and 'in', are removed as they typically do not contribute to the meaning of the sentence.

### Stemming and Lemmatization
- **Stemming** reduces words to their base forms by removing suffixes. This helps normalize words like "running" to "run".
- **Lemmatization** reduces words to their dictionary forms, considering the word's context. This ensures that words like "better" and "good" are related.

## Feature Extraction

### Count Vectorization
Count Vectorizer transforms text data into a matrix of token counts, indicating word frequency in each document.

### TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) assigns weights to words based on their importance. Words appearing frequently across documents are weighted lower than words that are unique to specific documents, improving classification.

## Word Embeddings

### Word2Vec
Word2Vec captures semantic relationships between words, creating dense vector representations of words that contain contextual information. Here, we use the `Gensim` library to train Word2Vec on the preprocessed tweets, generating embeddings for each word and sentence.

### GloVe
GloVe (Global Vectors for Word Representation) embeddings are pre-trained vectors capturing co-occurrence information between words. The GloVe vectors are loaded from a pre-trained model and used to represent sentences by averaging word vectors.

## Model Training and Evaluation

A Logistic Regression model is trained on the TF-IDF-transformed data. The model is evaluated based on accuracy, precision, recall, and F1-score metrics, demonstrating its ability to classify tweet sentiment effectively.

## Dependencies

- pandas
- numpy
- nltk
- sklearn
- gensim
- transformers (for advanced tokenization)

Make sure to download necessary NLTK data files before running the script:
```python
nltk.download('stopwords')
nltk.download('wordnet')
```

## Getting Started

1. Load your Twitter dataset (`twitter_data.csv`), containing columns like `tweet_content`.
2. Run `preprocess_text` to tokenize, remove stopwords, and apply stemming and lemmatization.
3. Use `CountVectorizer` or `TfidfVectorizer` for vectorization.
4. Generate embeddings with `Word2Vec` or load pre-trained GloVe vectors.
5. Train and evaluate a Logistic Regression model.

## Results

The model achieves an accuracy of 78% on the test set, with individual scores for each sentiment category. These scores indicate the effectiveness of feature extraction and preprocessing techniques in building a sentiment analysis classifier.


## Contributors

| Contributor      | Role/Responsibilities                                       |
|------------------|-------------------------------------------------------------|
| **Khushboo Mittal**       | Data Loading and Cleaning, Tokenization, Stopwords Removal, Stemming and Lemmatization|
| **Tanisha Priya**                | Count Vectorization and TF-IDF Vectorization|
| **Prachi Tavse**                |Word2Vec Embeddings, Word2Vec Sentence Vetorization|
| **Harshita Jangde**                | Load GloVe Embeddings, Sentence Vectorization, Embedding Application, Model Training and Evaluation|

