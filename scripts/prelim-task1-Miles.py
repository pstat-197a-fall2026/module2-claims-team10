from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import pyreadr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# These (to the best of my knowledge) are the 
# Python-equivalent libraries for the R packages 
# used in the original R script

#1 Loading and Checking RData files
raw_data = pyreadr.read_r("data/claims-raw.RData")
df = raw_data["claims_raw"]

#Checking the columns and number of pages loaded (it works as intended)
#print("Columns in dataset:", list(df.columns))
#print("Pages loaded:", len(df))

#2 Binary target for model: 1 is relevant claim content, 0 otherwise
#Matches Sid's code for task 3 hopefully
df["bclass_bin"] = (df["bclass"] == "Relevant claim content").astype(int)

#3 Baseline extractor: <p> tags only (Python equivalent of baseline in preprocessing.R)
def extract_just_p(html_text):
    #If whitespace or empty string, return empty string
    if not isinstance(html_text, str) or not html_text.strip():
        return ""
    
    #Parsing raw html to extract text from <p> tags 
    soup = BeautifulSoup(html_text, "html.parser")
    elements = soup.find_all("p")

    #returning one long text, seperated by spaces
    return " ".join(el.get_text(separator=" ", strip=True) for el in elements)

#4 Augmented Extractor (fulfilling Task 1 with <p> and <h1>-<h6>)
def extract_p_and_headers(html_text):
    #If whitespace or empty string, return empty string
    if not isinstance(html_text, str) or not html_text.strip():
        return ""
    
    #Parsing raw html to extract text from <p> and header tags
    soup = BeautifulSoup(html_text, "html.parser")
    elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    #returning one long text, seperated by spaces
    return " ".join(el.get_text(separator=" ", strip=True) for el in elements)

#5 Applying the extractors to the dataset
df["text_baseline"] = df["text_tmp"].apply(extract_just_p)

df["text_augmented"] = df["text_tmp"].apply(extract_p_and_headers)

# Verification check on Document 0 (it works as intended)
#print("\n SAMPLE EXTRACTED TEXT (Document 0)")
#print("Baseline length (characters):", len(df["text_baseline"].iloc[0]))
#print("Augmented length (characters):", len(df["text_augmented"].iloc[0]))

#6 Train / Test Split (80/20 stratified split, matching Sid's seed)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=69, stratify=df["bclass_bin"]
)

y_train = train_df["bclass_bin"]
y_test = test_df["bclass_bin"]

#7 TF-IDF Vectorizers, Transforming into a matrix of TF-IDF features (max 10,000 features, English stop words removed)
tfidf_base = TfidfVectorizer(max_features=10000, stop_words="english")

tfidf_aug = TfidfVectorizer(max_features=10000, stop_words="english")

#8 PCA via TruncatedSVD, Reducing the 10k columns to 50 components for both baseline and augmented extractors
pca_base = TruncatedSVD(n_components=50, random_state=69)
pca_aug = TruncatedSVD(n_components=50, random_state=69)

#9 Logistic Regression Classifiers, Using the 50 PCA components as predictors
clf_base = LogisticRegression(max_iter=1000, random_state=69)
clf_aug = LogisticRegression(max_iter=1000, random_state=69)

#Fitting Baseline Model (<p> only)
print("\nFitting Baseline Model (<p> only)...")

#Learning the 10k and 50 component weights on training set
X_train_b = pca_base.fit_transform(tfidf_base.fit_transform(train_df["text_baseline"]))

#Applying to testing set
X_test_b = pca_base.transform(tfidf_base.transform(test_df["text_baseline"]))

#Fitting the Logistic Regression model on training set
clf_base.fit(X_train_b, y_train)

#Seeing results on testing set
acc_base = accuracy_score(y_test, clf_base.predict(X_test_b))

#Fit Augmented Model (<p> + headers), Same coding process as above but with augmented text
print("Fitting Augmented Model (<p> + headers)...")

X_train_a = pca_aug.fit_transform(tfidf_aug.fit_transform(train_df["text_augmented"]))

X_test_a = pca_aug.transform(tfidf_aug.transform(test_df["text_augmented"]))

clf_aug.fit(X_train_a, y_train)

acc_aug = accuracy_score(y_test, clf_aug.predict(X_test_a))

#10 Print Results in Table for Deliverable 1
print("\n" + "=" * 45)
print("PRELIMINARY TASK 1 RESULTS")
print("=" * 45)
print(f"Baseline Accuracy (<p> only):       {acc_base:.4%}")
print(f"Augmented Accuracy (<p> + headers): {acc_aug:.4%}")
print(f"Net Difference:                     {(acc_aug - acc_base) * 100:+.2f}%")
print("=" * 45)