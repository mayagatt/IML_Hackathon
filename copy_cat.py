import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
# import matplotlib.pyplot as plt
# import string
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import data_preprocessing as dpre

train, test = dpre.create_train_test_data()
train_text = train['string_group']
train_target = train['project_number']
test_text = test['string_group']

# Replace all elements of punct by white spaces and delete all  numbers and a few chosen punctuation

# ps = PorterStemmer()

# def str_clean(text):
#     punct = '():[]?.,|_^-&><;!"/%'
#     table = str.maketrans(punct, ' ' * len(punct), "0123456789$#'=")
#     cleaned_comment = []
#     for word in text.split():
#         cleaned_comment.extend(word.translate(table).split())
#         cleaned_comment = [ps.stem(word) for word in cleaned_comment]
#     return " ".join(cleaned_comment)
#
#
# train_text = train_text.map(lambda x: str_clean(x))
#
# test_text = test_text.map(lambda x: str_clean(x))

tfidf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                   analyzer='word', token_pattern=r'\w{1,}',
                                   ngram_range=(1, 2),
                                   max_features=10000)
train_features = tfidf_vectorizer.fit_transform(train_text)
# test_features = tfidf_vectorizer.fit_transform(test_text)

pred = pd.DataFrame([])
labels = [0, 1, 2, 3, 4, 5, 6]

# for label in labels:
#     target = train[train["project_number"] == label]
#     classifier = LogisticRegression(solver='sag')
#     #, scoring='roc_auc'
#     cv_scores = cross_val_score(classifier, train_features, train_target, cv=10)
#     mean_cv = np.mean(cv_scores)
#     print('CV score for label {} is {}'.format(label, mean_cv))
#     print('\n')
#     classifier.fit(train_features, train_target)
#     # pred[label] = classifier.predict_proba(test_features)[:, 1]
#
# # print(pred)

from sklearn.naive_bayes import GaussianNB


for label in labels:
    target = train[train["project_number"] == label]
    classifier = LinearSVC(C=1e10, random_state=0, tol=1e-5, max_iter=100000)
    print("pre-cv")
    cv_scores = cross_val_score(classifier, train_features, train_target, cv=3)
    mean_cv = np.mean(cv_scores)
    print('CV score for label {} is {}'.format(label, mean_cv))
    print('\n')
    classifier.fit(train_features, train_target)
    # pred[label] = classifier.predict_proba(test_features)[:, 1]

# print(pred)