# Sentiment Analysis for User Feedback

## Project Overview

This project is a Sentiment Analysis tool that classifies user feedback into three categories:

* Positive
* Neutral
* Negative

The model is trained using Twitter sentiment data and can predict the sentiment of any user-provided text through a Streamlit web application.

---

## Dataset Used

Dataset: Twitter Sentiment Dataset

Columns:

* clean_text : Preprocessed tweet text
* category : Sentiment label

Label Mapping:

* -1 = Negative
* 0 = Neutral
* 1 = Positive

Dataset Size: Approximately 163,000 records.

---

## Preprocessing Steps

1. Removed missing values.
2. Converted sentiment labels to text labels.
3. Applied TF-IDF Vectorization.
4. Split data into training and testing sets.

---

## Model Used

Algorithm:

* Logistic Regression

Feature Extraction:

* TF-IDF (Term Frequency-Inverse Document Frequency)

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Visualizations

1. Sentiment Distribution Plot
2. Confusion Matrix

---

## Project Structure

Sentimental-analysis/

├── data/
│   └── Twitter_Data.csv

├── model/
│   ├── sentiment_model.pkl
│   └── tfidf.pkl

├── train.py

├── app.py

├── confusion_matrix.png

├── sentiment_distribution.png

├── README.md

---

## How to Run

### Train the Model

```bash
python train.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Streamlit Features

* Text input area
* Sentiment prediction
* Confidence score display
* Positive, Neutral, and Negative classification

---

## Learning Outcomes

* Data preprocessing using Pandas
* Feature extraction using TF-IDF
* Text classification using Logistic Regression
* Model evaluation and visualization
* Building an interactive Streamlit application

---

## Author

Vaishnavi

AI Internship Project
