# NLP - Sentiment Analysis

## Problem Statement

**Question:** What is the problem we are trying to solve?

**Answer:** Given a short paragraph, we want to determine whether it contains positive or negative sentiments.

---

## Training Data

**Question:** What does the training data look like?

**Answer:** The training data is stored in CSV files under two columns: 'sentiment' and 'paragraph'. The 'sentiment' column can contain 'positive', 'negative' or 'neutral', while the 'paragraph' column contains the extracted text.

![Steam Reviews Data Frame](images/steam_reviews_preliminary.png)

![News Articles](images/news.png)

---

## Data Handling

**Question:** Where to get the data?

**Answer:** One of the most reliable sources for collecting sentimental paragraphs is customer reviews of a product. For this project, we are using game reviews from Steam as our training data for positive and negative sentiments. News articles are typically written in a neutral tone and seldom contain emotions because their purpose is to present factual information to readers. Therefore, we will use them as neutral training data.

**Question:** How to get the data?

**Answer:** There are various ways to obtain our training data. For dynamic websites like Steam, which present content to the client using JavaScript, we need tools like Selenium and Beautiful Soup to simulate a browser and parse the data. For news articles, we will download a dataset containing URLs to online newspapers from Kaggle and use Requests and Beautiful Soup to obtain the articles.

**Question:** How to store the data?

**Answer:** We store the training data in CSV files with two headers: 'sentiment' and 'paragraph'. To conserve disk space, all files are compressed and bundled into a ZIP file.

[Scraped Steam Reviews](https://drive.google.com/file/d/175SDxxaWLFDwc8rsS7HXR701Y98Sq_qL/view?usp=share_link)

[Scraped News Articles](https://drive.google.com/file/d/1-43d6oVSrXvQtsxqK5HgUn4gmDAA1Rtt/view?usp=share_link)

**Question:** What preprocessing do we need to do?

**Answer:** Unlike professionally-written news articles, Steam reviews can be informal and short, often containing special symbols and emojis. Therefore, we must filter out both symbols and extremely short paragraphs to remove low-quality training data. After filtering, we tokenize the remaining paragraphs using the following sequence: 'tokenization', 'part-of-speech tagging', and 'lemmatization' or 'stemming'. Depending on the model, we may also need to use 'word embedding'.

![Lemmatization](images/lemmatization.webp)

![Word Embedding](https://miro.medium.com/v2/resize:fit:1400/0*H-w8rjUvuN6kw5kk.png)

---

## Data Analysis (Model Training)

**Question:** What are the common approaches to solve this problem?

**Answer:** A naive approach for 'word embedding' is 'one-hot encoding', which encodes a single bit on a sparse matrix the size of the entire dictionary. 'Word2Vec' is a machine learning approach that vectorizes each token in a vector space based on the neighboring words such that similar words are closer. For binary classification tasks, such as spam mail detection, one could use CNN (Convolutional Neural Network) or RNN (Recurrent Neural Network). A more modern technique would be to use BERT (Bidirectional Encoder Representations from Transformers), which is typically used for tasks such as 'word embedding', 'machine translation', 'question answering', and more.

![Word2Vec](https://devopedia.org/images/article/221/9279.1570465016.png)
