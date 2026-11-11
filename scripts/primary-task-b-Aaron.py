# 4. Develop the best predictive model you can for (a) binary classification and also for (b) the multi-class setting. 
# You can use any preprocessing and any modeling technique you like, including ones not discussed in class (consider exploring 
# RNN or SVM). Export a deployable copy of each model and a set of predictions from each model on the test data `claims-test.RData`. 
# Predictions should be formatted as a single data frame named `pred_df` with columns
#    -   `.id` containing the URL ID
#    -   `bclass.pred` containing the predicted label for the binary class setting
#    -   `mclass.pred` containing the predicted label for the multiclass setting

import pyreadr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

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

# Vectorizes text data using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                        min_df=3, max_features=10000, sublinear_tf=True)

tfidf_matrix_train = tfidf.fit_transform(train_df['text_clean'])
tfidf_matrix_test = tfidf.transform(test_df['text_clean'])

tfidf_df_train = pd.DataFrame(
    tfidf_matrix_train.toarray(), 
    columns=tfidf.get_feature_names_out()
)

tfidf_df_test = pd.DataFrame(
    tfidf_matrix_test.toarray(), 
    columns=tfidf.get_feature_names_out()
)

final_train_df = pd.concat([train_df, tfidf_df_train], axis=1)
final_test_df = pd.concat([test_df, tfidf_df_test], axis=1)

X_train = final_train_df.drop(columns=['.id', 'text_clean', 'mclass_encoded'])
y_train = final_train_df['mclass_encoded']

X_test = final_test_df.drop(columns=['.id', 'text_clean', 'mclass_encoded'])
y_test = final_test_df['mclass_encoded']

# Feature reduction using PCA
pca = PCA(n_components=0.7, random_state=69) # retain 70% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)







