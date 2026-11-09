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

print("Columns in dataset:", list(df.columns))
print("Pages loaded:", len(df))