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
from bs4 import BeautifulSoup

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
# Task 3 found that the tuned neural network performed better than the baseline neural network,
# in terms of accuracy and ROC-AUC score, so we will use the tuned neural network architecture here.
multi_nn = keras.Sequential(
    [
        keras.Input(shape=(X_train_pca.shape[1],)),
        layers.Dropout(0.4),
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-3),
        ),
        layers.Dropout(0.3),
        layers.Dense(len(np.unique(y_train)), activation="softmax"),
    ],
    name="Multi_NN",
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

multi_nn.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

multi_nn.fit(
    X_train_pca,
    y_train,
    validation_split=0.3,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop]
)

with open('results/best_neural_network_results', 'wb') as f: # saving results 
        pickle.dump(multi_nn, f)

y_proba_nn = multi_nn.predict(X_test_pca)
y_pred_nn = np.argmax(y_proba_nn, axis=1)

# Calculate confusion matrix and metrics for multi-class neural network
confusion_nn = confusion_matrix(y_test, y_pred_nn)
tp = np.diag(confusion_nn)
fp = confusion_nn.sum(axis=0) - tp
fn = confusion_nn.sum(axis=1) - tp
tn = confusion_nn.sum() - (tp + fp + fn)

print("Multi-class Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Multi-class Neural Network Sensitivity:", np.mean(tp/(tp + fn)))
print("Multi-class Neural Network Specificity:", np.mean(tn/(tn + fp)))
print("Multi-class Neural Network AUC Score:", roc_auc_score(y_test, y_proba_nn, multi_class='ovo', average='macro'))

## Confusion matrix for multi-class neural network
labels = [
    "N/A: No relevant\ncontent",
    "Other claim\ncontent",
    "Physical Activity",
    "Possible Fatality",
    "Potentially unlawful\nactivity"
]

plt.figure(figsize=(11, 7))
sns.heatmap(confusion_nn, 
            xticklabels = labels,
            yticklabels = labels,
            annot = True,
            fmt = 'g')

plt.title("Confusion Matrix - Multi-class Neural Network")
plt.xlabel("Predicted Class", fontsize=12, labelpad=12)
plt.ylabel("True Class", fontsize=12, labelpad=12)
plt.xticks(rotation=0, ha="center", fontsize=10)
plt.yticks(rotation=0, fontsize=10)   
plt.savefig("results/graphs/multi_nn_confusion_matrix.png", dpi=300, bbox_inches='tight')

# Logistic Regression Results
# Accuracy: 0.7757
# Sensitivity: 0.658
# Specificity: 0.9243
# AUC Score: 0.9294

# Neural Network Results
# Accuracy: 0.7747
# Sensitivity: 0.6497
# Specificity: 0.9249
# AUC Score: 0.9258

# Seems like the multi-class logistic regression model performs slightly better than the multi-class neural 
# network model in terms of accuracy, AUC score, and sensitivity. However, the neural network has a slightly 
# higher specificity. Overall, both models perform comparably well on this multi-class classification task, 
# but since the logistic regression model is simpler and more interpretable, it may be preferred for deployment 
# in this case.

# Testing the best models on the test set and creating the final predictions dataframe
test_df = pyreadr.read_r("data/claims-test.RData")['claims_test']

## Functions imported from Task 1 for extracting text from HTML
def extract_p_and_headers(html_text):
    if not isinstance(html_text, str) or not html_text.strip():
        return ""
    
    soup = BeautifulSoup(html_text, "xml")
    elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    
    return " ".join(el.get_text(separator=" ", strip=True) for el in elements)

test_df["cleaned_text"] = test_df["text_tmp"].apply(extract_p_and_headers)

print("Actual testing set shape:", test_df.shape)

## Preprocessing the test set
test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
tfidf_matrix_test = tfidf.transform(test_df['cleaned_text'])
tfidf_df_test = pd.DataFrame(
    tfidf_matrix_test.toarray(), 
    columns=tfidf.get_feature_names_out(),
    index=test_df.index
)

final_test_df = pd.concat([test_df, tfidf_df_test], axis=1)

X_test_pca = pca.transform(final_test_df.drop(columns=['.id', 'text_tmp', 'cleaned_text']))
y_pred = multi_logistic.predict(X_test_pca)
final_matrix = pd.concat([final_test_df['.id'], pd.Series(multiclass_encoder.inverse_transform(y_pred), name='mclass.pred')], axis=1)
print(final_matrix.head())

## Combining binary and multi-class predictions into a single dataframe
