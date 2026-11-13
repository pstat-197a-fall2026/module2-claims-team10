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
    holdout_acc = accuracy_score(test["y"], model.predict(test["text_clean"]))
    rows.append({"model": name, "cv_accuracy": cv_scores.mean(),
                 "cv_std": cv_scores.std(), "holdout_accuracy": holdout_acc})
 
results = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
print("\n10-fold CV model comparison:")
print(results.round(4).to_string(index=False))

best_name = results.iloc[0]["model"]
best_model = models[best_name]
print(f"\nBest model by CV: {best_name}\n")
print(classification_report(test["y"], best_model.predict(test["text_clean"]),
                            target_names=["No relevant content", "Relevant claim"]))

# fit best model on all labeled data and save
final_model = best_model.fit(df["text_clean"], df["y"])
with open("results/best-binary-model.pkl", "wb") as f:
    pickle.dump(final_model, f)

# hold-out metrics for the best momdel
pred = best_model.predict(test["text_clean"])
tn, fp, fn, tp = confusion_matrix(test["y"], pred).ravel()
scores = best_model.decision_function(test["text_clean"])

print("Accuracy:", round(accuracy_score(test["y"], pred), 4))
print("Sensitivity:", round(tp / (tp + fn), 4))
print("Specificity:", round(tn / (tn + fp), 4))
print("AUC:", round(roc_auc_score(test["y"], scores), 4))

# Accuracy: 0.9509
# Sensitivity: 0.9644
# Specificity: 0.936
# AUC: 0.9882