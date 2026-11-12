import os
import pickle
import pandas as pd
import pyreadr
from cleaning_Iris import clean_html

# load and clean raw data
claims_test = next(iter(pyreadr.read_r("data/claims-test.RData").values()))
text = claims_test["text_tmp"].apply(clean_html)

# create dataframe of predictions on test data
pred_df = pd.DataFrame({".id": claims_test[".id"].values})

# binary classification
with open("results/best-binary-model.pkl", "rb") as f:
    binary_model = pickle.load(f)
binary_labels = {1: "Relevant claim content", 0: "N/A: No relevant content."}
pred_df["bclass.pred"] = pd.Series(binary_model.predict(text)).map(binary_labels).values

# multiclass
multi_df = pd.read_csv("results/pred_df_multi.csv")
pred_df = pred_df.merge(multi_df, on=".id", how="left")

# save df
pred_df.to_csv("results/pred_df.csv", index=False)
print(pred_df.shape)
print(pred_df.head())
print("\nbclass vs mclass predictions:")
print(pd.crosstab(pred_df["bclass.pred"], pred_df["mclass.pred"]))