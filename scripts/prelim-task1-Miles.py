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