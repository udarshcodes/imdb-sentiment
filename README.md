# IMDb Sentiment Classification

This project trains a text classification model on the IMDb Movie Review dataset to predict whether a review is positive or negative.

## Live Demo

There is currently no live demo available for this project as it is a local training script.

## About

This is a machine learning text classification project built to perform sentiment analysis on movie reviews. It serves as a practical implementation of natural language processing techniques using Python. The project demonstrates how to preprocess textual data and train a logistic regression model. It is a baseline implementation and does not use advanced deep learning techniques.

## Features

- Train a logistic regression model on a dataset of IMDb movie reviews
- Preprocess text data by converting reviews into a bag-of-words representation and removing English stop words
- Evaluate model accuracy on a 20% holdout test set

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python |
| Data Manipulation | pandas |
| Machine Learning | scikit-learn |

## How It Works

The script first loads the IMDb dataset using pandas. It extracts the review text and sentiment labels, then processes the text using scikit-learn's `CountVectorizer` to create a sparse numeric matrix of word counts. Finally, the data is split into training and testing sets, and a `LogisticRegression` model is trained and evaluated on the test set to determine its accuracy.

## Running Locally

1. Navigate to the project directory:
```bash
cd imdb-sentiment
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python imdb_sentiment.py
```

## Project Structure

```text
.
├── imdb.csv             # The IMDb movie review dataset
├── imdb_sentiment.py    # Main script for training and evaluating the model
├── requirements.txt     # Python dependencies
├── LICENSE              # Project license
└── README.md            # Project documentation
```

## What I Learned & Key Decisions

- Chose Logistic Regression as the baseline model for its simplicity, interpretability, and fast training time on sparse text data.
- Used a Bag-of-Words representation (`CountVectorizer`) rather than more complex word embeddings to keep the preprocessing straightforward and computationally lightweight.
- Removed English stop words during vectorization to reduce noise and improve model focus on meaningful vocabulary.
- Kept the dataset loading and model training within a single script for ease of execution and readability.
