import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv("imdb.csv")  # file must be in the same folder

# 2. Preprocessing
X = df['review']
y = df['sentiment']

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", accuracy)
