# Preliminary Task 2
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split

# Load data
claims = pyreadr.read_r("data/claims-clean-example.RData")
data = claims["claims_clean"]
print(data.shape)
print(data.columns)
print(data.head())

print(data["bclass"].value_counts(dropna=False))
print(data["text_clean"].isna().sum())