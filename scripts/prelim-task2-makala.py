# Preliminary Task 2
# Makala Mesina-Forester

# Import libraries
import pandas as pd
import pyreadr
import re
from bs4 import BeautifulSoup
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


