from pathlib import Path
import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf


np.random.seed(69)
tf.random.set_seed(69)

# import pyreadr
# pyreadr.read_r("../data/claims-clean-example.RData")["claims_clean"].to_csv("../data/claims-clean-example.csv", index=False)
# pyreadr.read_r("../data/claims-raw.RData")["claims_raw"].to_csv("../data/claims-raw.csv", index=False)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "claims-clean-example.csv"
claims_clean = pd.read_csv(DATA_PATH)
claims_clean["text_clean"] = claims_clean["text_clean"].fillna("")

# Making the bclass variable binary so that it is 1 if there is relevant claim content and 0 otherwise.
claims_clean["bclass_bin"] = (claims_clean["bclass"] == "Relevant claim content").astype(int)

# Splitting the dataset 80/20, stratifying by the binary variable
train_df, test_df = train_test_split(
    claims_clean, test_size=0.2, random_state=69, stratify=claims_clean["bclass_bin"]
)

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


# Initializing the neural network model
baseline_nn = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(25, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="Baseline_NN",
)

baseline_nn.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["binary_accuracy"],
)

baseline_nn.fit(
    X_train,
    train_labels,
    validation_split=0.3,
    epochs=5,
    batch_size=32,
    verbose=1,
)

# Tuning the model with 32 layers, early stopping and adding stronger regularization
tuned_NN = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[1],)),
        layers.Dropout(0.4),
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-3),
        ),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="Tuned_NN",
)

tuned_NN.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["binary_accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

tuned_NN.fit(
    X_train,
    train_labels,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# Evaluation of both neural networks
base_probs = baseline_nn.predict(X_test, verbose=0).ravel()
base_preds = (base_probs >= 0.5).astype(int)

tuned_probs = tuned_NN.predict(X_test, verbose=0).ravel()
tuned_preds = (tuned_probs >= 0.5).astype(int)

nn_stats = pd.DataFrame(
    [
        {
            "Model": "Baseline NN",
            "Accuracy": accuracy_score(test_labels, base_preds),
            "Precision": precision_score(test_labels, base_preds),
            "Recall": recall_score(test_labels, base_preds),
            "F1-Score": f1_score(test_labels, base_preds),
            "ROC-AUC": roc_auc_score(test_labels, base_probs),
        },
        {
            "Model": "Tuned NN",
            "Accuracy": accuracy_score(test_labels, tuned_preds),
            "Precision": precision_score(test_labels, tuned_preds),
            "Recall": recall_score(test_labels, tuned_preds),
            "F1-Score": f1_score(test_labels, tuned_preds),
            "ROC-AUC": roc_auc_score(test_labels, tuned_probs),
        },
    ]
)

# Table showcasing KPIS for each NN model trained.
print(nn_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


print("Baseline NN Confusion Matrix:")
print(confusion_matrix(test_labels, base_preds))
print("Baseline NN Classification Report:")
print(classification_report(test_labels, base_preds, target_names=["N/A", "Relevant Claim"]))

print("Tuned NN Confusion Matrix:")
print(confusion_matrix(test_labels, tuned_preds))
print("Tuned NN Classification Report:")
print(classification_report(test_labels, tuned_preds, target_names=["N/A", "Relevant Claim"]))




