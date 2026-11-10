# Preliminary Task 2
import pandas as pd
import pyreadr

# Load data
claims = pyreadr.read_r("data/claims-clean-example.RData")

print(claims.keys())

for name, df in claims.items():
    print("Object:", name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    
    

