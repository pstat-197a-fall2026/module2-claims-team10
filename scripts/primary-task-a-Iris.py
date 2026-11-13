import pickle
import pandas as pd
import pyreadr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 69

# load data
df = pyreadr.read_r("data/claims-clean-example.RData")["claims_clean"]
df["text_clean"] = df["text_clean"].fillna("")
# binary label (1 = relevant)
df["y"] = (df["bclass"] == "Relevant claim content").astype(int)
 
# 80/20 stratified split
train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])

# model building
def make_model(classifier):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                            min_df=3, max_features=50000, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])
 
models = {
    "Logistic Regression": make_model(LogisticRegression(C=10, max_iter=2000)),
    "Linear SVM":          make_model(LinearSVC(C=1)),
    "Naive Bayes":         make_model(MultinomialNB()),
}

# compare with 10-fold cross-validation, and check on 20% holdout set
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
rows = []

for name, model in models.items():
    cv_scores = cross_val_score(model, train["text_clean"], train["y"], cv=cv, n_jobs=-1)
    model.fit(train["text_clean"], train["y"])
    pred = model.predict(test["text_clean"])
    tn, fp, fn, tp = confusion_matrix(test["y"], pred).ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(test["text_clean"])
    else:
        scores = model.predict_proba(test["text_clean"])[:, 1]
    rows.append({"model": name,
                 "cv_accuracy": cv_scores.mean(),
                 "cv_std": cv_scores.std(),
                 "holdout_accuracy": accuracy_score(test["y"], pred),
                 "sensitivity": tp / (tp + fn),
                 "specificity": tn / (tn + fp),
                 "auc": roc_auc_score(test["y"], scores)})

results = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
print("\nModel comparison:")
print(results.round(4).to_string(index=False))

# Model comparison:
#               model  cv_accuracy  cv_std  holdout_accuracy  sensitivity  specificity    auc
#          Linear SVM       0.8236  0.0281            0.8014       0.8133       0.7882 0.8744
# Logistic Regression       0.8224  0.0254            0.7944       0.8044       0.7833 0.8716
#         Naive Bayes       0.7663  0.0261            0.7617       0.8089       0.7094 0.8445

# find best model
best_name = results.iloc[0]["model"]
best_model = models[best_name]
print(f"\nBest model by CV: {best_name}")

# fit best model on all labeled data and save
final_model = best_model.fit(df["text_clean"], df["y"])
with open("results/best-binary-model.pkl", "wb") as f:
    pickle.dump(final_model, f)

# SVM model metrics:
# Accuracy: 0.8014
# Sensitivity: 0.8133
# Specificity: 0.7882
# AUC: 0.8744