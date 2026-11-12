# 4. Develop the best predictive model you can for (a) binary classification and also for (b) the multi-class setting. 
# You can use any preprocessing and any modeling technique you like, including ones not discussed in class (consider exploring 
# RNN or SVM). Export a deployable copy of each model and a set of predictions from each model on the test data `claims-test.RData`. 
# Predictions should be formatted as a single data frame named `pred_df` with columns
#    -   `.id` containing the URL ID
#    -   `bclass.pred` containing the predicted label for the binary class setting
#    -   `mclass.pred` containing the predicted label for the multiclass setting

import os

import pyreadr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import pickle
import keras
from keras import layers
import tensorflow as tf

df = pyreadr.read_r("data/claims-clean-example.RData")['claims_clean']

# Preprocess
cleaned_df = df.copy()
cleaned_df['text_clean'] = cleaned_df['text_clean'].fillna('')

multiclass_encoder = LabelEncoder() # encode multiclass labels 
multiclass_encoder.fit(cleaned_df['mclass'])
cleaned_df['mclass_encoded'] = multiclass_encoder.transform(cleaned_df['mclass'])
cleaned_df = cleaned_df[['.id', 'text_clean', 'mclass_encoded']]

# Split dataset 
train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=69, stratify=cleaned_df['mclass_encoded'])

# Vectorize text data using TF-IDF into bigrams 
# Task 2 found that bigrams performed better than unigrams, so we will use bigrams here.
tfidf = TfidfVectorizer(
    stop_words="english", 
    ngram_range=(2, 2),
    min_df=3, 
    max_features=10000, 
    sublinear_tf=True
)

tfidf_matrix_train = tfidf.fit_transform(train_df['text_clean'])
tfidf_matrix_test = tfidf.transform(test_df['text_clean'])

tfidf_df_train = pd.DataFrame(
    tfidf_matrix_train.toarray(), 
    columns=tfidf.get_feature_names_out(),
    index=train_df.index
)

tfidf_df_test = pd.DataFrame(
    tfidf_matrix_test.toarray(), 
    columns=tfidf.get_feature_names_out(),
    index=test_df.index
)

final_train_df = pd.concat([train_df, tfidf_df_train], axis=1)
final_test_df = pd.concat([test_df, tfidf_df_test], axis=1)

# Split features and labels for training and testing
X_train = final_train_df.drop(columns=['.id', 'text_clean', 'mclass_encoded'])
y_train = final_train_df['mclass_encoded']

X_test = final_test_df.drop(columns=['.id', 'text_clean', 'mclass_encoded'])
y_test = final_test_df['mclass_encoded']

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Feature reduction using PCA
pca = PCA(n_components=0.7, random_state=69) # retain 70% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Multi-class logistic regression model (with penalty)
multi_logistic = LogisticRegression(solver='saga', max_iter=10000, random_state=69)

## Hyperparameter tuning
logistic_gs = GridSearchCV(
    multi_logistic,
    param_grid={
        "l1_ratio" : np.linspace(0, 1, 10, True), 
        "C" : 1/np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])
    },
    scoring='roc_auc_ovo',
    return_train_score=True,
    cv=5,
)

if not os.path.exists('results/best_logistic_results'):
    logistic_gs.fit(X_train_pca, y_train) # fit only when results do not exist
    with open('results/best_logistic_results', 'wb') as f: # saving results 
        pickle.dump(logistic_gs.best_estimator_, f)

## Evaluating best fit model on test set
multi_logistic = pickle.load(open('results/best_logistic_results', 'rb'))
print("Best parameters for multi-class logistic regression:", "L1 Ratio:", multi_logistic.get_params()['l1_ratio'], "C:", multi_logistic.get_params()['C'])

multi_logistic.fit(X_train_pca, y_train)
y_pred_logistic = multi_logistic.predict(X_test_pca)
y_proba_logistic = multi_logistic.predict_proba(X_test_pca)

## Calculate confusion matrix and metrics for multi-class logistic regression
confusion_logistic = confusion_matrix(y_test, y_pred_logistic, labels=multi_logistic.classes_)
tp = np.diag(confusion_logistic)
fp = confusion_logistic.sum(axis=0) - tp
fn = confusion_logistic.sum(axis=1) - tp
tn = confusion_logistic.sum() - (tp + fp + fn)

print("Multi-class Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Multi-class Logistic Regression Sensitivity:", np.mean(tp/(tp + fn)))
print("Multi-class Logistic Regression Specificity:", np.mean(tn/(tn + fp)))
print("Multi-class Logistic Regression AUC Score:", roc_auc_score(y_test, y_proba_logistic, multi_class='ovo', average='macro'))

## Confusion matrix for multi-class logistic regression
labels = [
    "N/A: No relevant\ncontent",
    "Other claim\ncontent",
    "Physical Activity",
    "Possible Fatality",
    "Potentially unlawful\nactivity"
]

plt.figure(figsize=(11, 7))
sns.heatmap(confusion_logistic, 
            xticklabels = labels,
            yticklabels = labels,
            annot = True,
            fmt = 'g')

plt.title("Confusion Matrix - Multi-class Logistic Regression")
plt.xlabel("Predicted Class", fontsize=12, labelpad=12)
plt.ylabel("True Class", fontsize=12, labelpad=12)
plt.xticks(rotation=0, ha="center", fontsize=10)
plt.yticks(rotation=0, fontsize=10)   
plt.savefig("results/graphs/multi_logistic_confusion_matrix.png", dpi=300, bbox_inches='tight')

# Multi-class neural network model
baseline_nn = keras.Sequential(
    [
        keras.Input(shape=(X_train_pca.shape[1],)),
        layers.Dense(25, activation="relu"),
        layers.Dense(len(np.unique(y_train)), activation="softmax"),
    ],
    name="Baseline_NN",
)

baseline_nn.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

baseline_nn.fit(
    X_train_pca,
    y_train,
    validation_split=0.3,
    epochs=5,
    batch_size=32,
    verbose=1,
)