# NLP - Text Preprocessing

### 1. **Business Problem Statement**
In today’s social media-driven world, analyzing sentiment on platforms like Twitter is crucial for businesses aiming to understand public opinion and trends. This project seeks to develop an NLP pipeline capable of processing Twitter sentiment data to gain insights into how entities (such as brands, products, or individuals) are perceived. The primary objective is to create a DIY solution that includes all foundational NLP processing techniques (stemming, lemmatization, count vectorization, TF-IDF) and advanced embeddings (Word2Vec, GloVe). Additionally, by implementing a custom Encoder-Decoder model in PyTorch, the project will explore sentiment prediction using an unsupervised deep learning approach.

### 1. **Download Necessary NLTK Resources**
First, we download the required NLTK resources for stopwords and WordNet data, which are essential for text preprocessing (removing common words and lemmatization).

### 2. **Text Preprocessing**
We define a preprocessing function that:
- Converts the text to lowercase.
- Removes stopwords.
- Applies stemming and lemmatization to normalize the text.

### 3. **Text Vectorization**

#### 3.1 **Count Vectorizer**
We use the `CountVectorizer` to convert the text data into a matrix of token counts.

#### 3.2 **TF-IDF Vectorizer**
Next, we use `TfidfVectorizer` to convert the text data into Term Frequency-Inverse Document Frequency (TF-IDF) features.

#### 3.3 **Word2Vec and GloVe**
We also apply Word2Vec and GloVe embeddings to convert the text data into dense vector representations. (For brevity, you can add the embeddings here if needed.)

### 4. **Model Training: Logistic Regression**
Finally, we train a logistic regression model using the TF-IDF vectorizer's output as the features.

### 5. **Conclusion**
- The text preprocessing pipeline includes steps such as stopwords removal, stemming, and lemmatization.
- The vectorization process converts the text data into numerical representations using Count Vectorizer, TF-IDF, and embeddings like Word2Vec and GloVe.
- Finally, a logistic regression model is trained on the TF-IDF features and evaluated for performance.
