# Preliminary Task 2
# Makala Mesina-Forester

# Import libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pyreadr
import re


## Preprocessed data already saved as data/claims-preprocessed.csv

# # Load raw claims data
# raw_claims_data = pyreadr.read_r("data/claims-raw.RData")
# data = raw_claims_data["claims_raw"].copy()
# print("Raw data shape:", data.shape)

# # Create binary target variable (Relevant claim)
# data["bclass_bin"] = (data["bclass"] == "Relevant claim content").astype(int)
# print(data["bclass_bin"].value_counts())

# # Define text extraction function
# def extract_text(html_text):
#     if not isinstance(html_text, str) or not html_text.strip():
#         return ""
#     soup = BeautifulSoup(html_text, "lxml")
   
#     # Extract paragraph text
#     paragraphs = soup.find_all("p")
#     text = " ".join(
#         p.get_text(separator=" ", strip=True)
#         for p in paragraphs
#     )
#     return text

# # Define cleaning function
# def clean_text(text):
#     text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # Remove URLs
#     text = re.sub(r"\S+@\S+", " ", text)                    # Remove emails
#     text = re.sub(r"[^a-zA-Z\s]", " ", text)                # Remove punctuation and numbers
#     text = text.lower()                                     # Lowercase
#     text = re.sub(r"\s+", " ", text).strip()                # Remove extra whitespace

#     return text

# # Preprocessings
# data = data[
#     data["text_tmp"].str.contains("<!", na=False, regex=False)
# ].copy()

# data["text_clean"] = (
#     data["text_tmp"]
#     .apply(extract_text)
#     .apply(clean_text)
# )

# # Remove rows with empty cleaned text
# data = data[data["text_clean"] != ""].copy()

# # Check results
# print("Preprocessed data shape:", data.shape)
# print("Empty texts:", (data["text_clean"] == "").sum())
# print(data[["text_clean", "bclass_bin"]].head())

# # Save data
# data.to_csv("data/claims-preprocessed.csv", index=False)

# Load csv
data = pd.read_csv("data/claims-preprocessed.csv")
print(data.shape)

# Split data
X = data["text_clean"].fillna("")
y = data["bclass_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=69,
    stratify=y)

print("Training Set Size:", len(X_train))
print("Testing Set Size:", len(X_test))

# Word Tokenization
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    stop_words="english",
    min_df=2)

X_word_train = word_vectorizer.fit_transform(X_train)
X_word_test = word_vectorizer.transform(X_test)

print("Word Matrix Shape:", X_word_train.shape)

# Bigram Tokenization
bigram_vectorizer = TfidfVectorizer(
    ngram_range=(2, 2),
    stop_words="english",
    min_df=2)

X_bigram_train = bigram_vectorizer.fit_transform(X_train)
X_bigram_test = bigram_vectorizer.transform(X_test)

print("Bigram Matrix Shape:", X_bigram_train.shape)


# PCA - Word
word_svd = TruncatedSVD(n_components=20, random_state=69)   # 20 components

X_word_train_pc = word_svd.fit_transform(X_word_train)
X_word_test_pc = word_svd.transform(X_word_test)

print("Word Component Shape:", X_word_train_pc.shape)

# PCA - Bigram
bigram_svd = TruncatedSVD(n_components=20, random_state=69)

X_bigram_train_pc = bigram_svd.fit_transform(X_bigram_train)
X_bigram_test_pc = bigram_svd.transform(X_bigram_test)

print("Bigram Component Shape:", X_bigram_train_pc.shape)


## Model 1: Logistic regression 
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_word_train_pc, y_train)

# Predict probabilities using test set
prob1 = model1.predict_proba(X_word_test_pc)[:, 1]
print("Model 1 Probabilities:", prob1[:10])

# Predicted log-oddss
log_odds_test = model1.decision_function(X_word_test_pc)
print("Predicted Log-Odds:", log_odds_test[:10])

# Model 1 Pipeline
word_pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 1),
        stop_words="english",
        min_df=2
    ),
    TruncatedSVD(
        n_components=20,
        random_state=69
    ),
    LogisticRegression(max_iter=1000))

# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=69)

log_odds_train = cross_val_predict(
    word_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="decision_function")

print("Training Log-Odds Shape:", log_odds_train.shape)

# Combine training features
X_model2_train = np.column_stack([
    log_odds_train,
    X_bigram_train_pc])

# Combine testing features
X_model2_test = np.column_stack([
    log_odds_test,
    X_bigram_test_pc])

print("Model 2 Training Shape:", X_model2_train.shape)
print("Model 2 Testing Shape:", X_model2_test.shape)


## Model 2: Logistic regression using log-odds and bigrams
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_model2_train, y_train)

# Predict probabilities and classes
prob2 = model2.predict_proba(X_model2_test)[:, 1]
pred2 = model2.predict(X_model2_test)

print("Model 2 Probabilities:", prob2[:10])

## Comparing Models
# Model 1
pred1 = model1.predict(X_word_test_pc)
print("Model 1 Accuracy:", accuracy_score(y_test, pred1))
print("Model 1 AUROC:", roc_auc_score(y_test, prob1))

# Model 2
print("Model 2 Accuracy:", accuracy_score(y_test, pred2))
print("Model 2 AUROC:", roc_auc_score(y_test, prob2))

## Results:
# Model 1 Accuracy: 0.7589
# Model 1 AUROC: 0.8174
# Model 2 Accuracy: 0.7589
# Model 2 AUROC: 0.8250