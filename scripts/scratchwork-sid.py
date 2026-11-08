from pathlib import Path
import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf


np.random.seed(69)
tf.random.set_seed(69)

# import pyreadr
# pyreadr.read_r("../data/claims-clean-example.RData")["claims_clean"].to_csv("../data/claims-clean-example.csv", index=False)
# pyreadr.read_r("../data/claims-raw.RData")["claims_raw"].to_csv("../data/claims-raw.csv", index=False)

claims_clean = pd.read_csv("../data/claims-clean-example.csv")
claims_clean["text_clean"] = claims_clean["text_clean"].fillna("")

# Making the bclass variable binary so that it is 1 if there is relevant claim content and 0 otherwise.
claims_clean["bclass_bin"] = (claims_clean["bclass"] == "Relevant claim content").astype(int)

# Splitting the dataset 80/20, stratifying by the binary variable
train_df, test_df = train_test_split(claims_clean, test_size=0.2, random_state=69,stratify=claims_clean["bclass_bin"],)

train_text = tf.constant(train_df["text_clean"].tolist(), dtype=tf.string)
train_labels = train_df["bclass_bin"].to_numpy(dtype=np.float32)

test_text = tf.constant(test_df["text_clean"].tolist(), dtype=tf.string)
test_labels = test_df["bclass_bin"].to_numpy(dtype=np.float32)


preprocess_layer = layers.TextVectorization(
    standardize=None,
    split="whitespace",
    ngrams=None,  
    max_tokens=10000,  
    output_mode="tf_idf",
)

# Scans the train_text to learn the 10000 vocab words and each words TF-IDF
preprocess_layer.adapt(train_text)

# Vectorize into numeric TF-IDF matrices for fast Keras training
X_train = preprocess_layer(train_text)
X_test = preprocess_layer(test_text)

