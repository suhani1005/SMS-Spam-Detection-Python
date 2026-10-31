import pandas as pd
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Remove duplicate messages
data = data.drop_duplicates(subset="message")

# Remove missing or empty records
data = data.dropna(subset=["label", "message"])
data = data[data["message"].str.strip() != ""]

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nMessage categories:")
print(data["label"].value_counts())


# Prepare input and output
X = data["message"]
y = data["label"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create the ML pipeline
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# Parameters to test
parameter_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "classifier__alpha": [0.1, 0.5, 1.0]
}


# Focus on F1 score for spam messages
spam_f1_scorer = make_scorer(
    f1_score,
    pos_label="spam"
)


# Search for the best combination
search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring=spam_f1_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)


print("\nSearching for the best model...")
search.fit(X_train, y_train)

# Select the best model
model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print(
    "\nBest cross-validation spam F1 score:",
    round(search.best_score_ * 100, 2),
    "%"
)


# Evaluate on the separate test dataset
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nFinal Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )
)


# Save the best model
model_path = MODEL_DIR / "spam_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nBest model saved successfully!")
print("Location:", model_path)