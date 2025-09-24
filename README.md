IMDb Sentiment Classification

This project trains a text classification model on the IMDb Movie Review dataset to predict whether a review is positive or negative.

##The dataset contains two columns:
1. review → text of the movie review
2. entiment → label (positive or negative)

##Preprocessing Steps
Text Cleaning & Stopword Removal
Used CountVectorizer(stop_words='english') to remove common stopwords (e.g., "the", "is", "and").

##Text Vectorization
Converted raw text into a bag-of-words representation using CountVectorizer.
Each review is represented as a sparse numeric vector based on word counts.

##Train/Test Split
Split dataset into 80% training and 20% testing using train_test_split.

##Models Used
Logistic Regression (LogisticRegression(max_iter=1000))
Trained on the vectorized reviews.
Achieved 88.3% accuracy on the IMDb dataset.

##Results
Logistic Regression → 88.3% accuracy

##How to Run
Install dependencies: pip install pandas scikit-learn
Run the script: python imdb_sentiment.py

Output:
Accuracy of Logistic Regression model: 0.883
