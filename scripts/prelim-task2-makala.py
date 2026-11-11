# Preliminary Task 2
# Makala Mesina-Forester

# Import libraries
import pandas as pd
import pyreadr
import re
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Preprocessed data already saved
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

# # Save preprocessed data
# data.to_csv("data/claims-preprocessed.csv", index=False)

# Load preprocessed data
data = pd.read_csv("data/claims-preprocessed.csv")
print(data.shape)

# Split data
X = data["text_clean"].fillna("")
y = data["bclass_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=15,
    stratify=y
)

print("Training Set Size:", len(X_train))
print("Testing Set Size:", len(X_test))

# Word Tokenization
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    stop_words="english",
    min_df=2
)

X_word_train = word_vectorizer.fit_transform(X_train)
X_word_test = word_vectorizer.transform(X_test)

print("Word matrix shape:", X_word_train.shape)

# Bigram Tokenization
bigram_vectorizer = TfidfVectorizer(
    ngram_range=(2, 2),
    stop_words="english",
    min_df=2
)

X_bigram_train = bigram_vectorizer.fit_transform(X_train)
X_bigram_test = bigram_vectorizer.transform(X_test)

print("Bigram matrix shape:", X_bigram_train.shape)


# PCA - Word
word_svd = TruncatedSVD(n_components=20, random_state=42)   # 20 components

X_word_train_pc = word_svd.fit_transform(X_word_train)
X_word_test_pc = word_svd.transform(X_word_test)

print("Word component shape:", X_word_train_pc.shape)

# PCA - Bigram
bigram_svd = TruncatedSVD(n_components=20, random_state=42)

X_bigram_train_pc = bigram_svd.fit_transform(X_bigram_train)
X_bigram_test_pc = bigram_svd.transform(X_bigram_test)

print("Bigram component shape:", X_bigram_train_pc.shape)


