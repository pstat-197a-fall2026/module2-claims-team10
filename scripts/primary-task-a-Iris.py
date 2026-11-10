import re
import warnings
import joblib
import pandas as pd
import pyreadr
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
SEED = 69

#  preprocessing: html to text, drop urls/emails, punctuation, digits, lowercase
def clean_html(html):
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = " ".join(t.get_text(separator=" ", strip=True) for t in tags)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()

def main():
    df = pyreadr.read_r("data/claims-raw.RData")["claims_raw"]
    df["text_clean"] = df["text_tmp"].apply(clean_html)
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
    
    # fit best model
    final_model = best_model.fit(df["text_clean"], df["y"])
    joblib.dump(final_model, "results/binary-model.joblib")
    print("Saved model -> results/binary-model.joblib")

if __name__ == "__main__":
    main()