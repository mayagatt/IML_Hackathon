"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors: Hadas Nahshon, Guy Lutzker, Maya harari, Omer Plotnik

===================================================
"""
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import data_preprocessing as dpre
from sklearn.externals import joblib
import pandas as pd
import eda_process

##train final model and save it to a pkl file

##parse data and evaluate model score
train, test = dpre.create_train_test_data()  # create train and test dataframes
model = eda_process.fit_model(LinearSVC(C=1.6, random_state=25, tol=1e-5))
score = eda_process.score_model(model, test['string_group'], test['project_number'])
print(score)

## train final model and save it in a pkl file
all_data = pd.concat([train, test], ignore_index=True)
all_text = all_data['string_group']  # extract train text from train
all_target = all_data['project_number']

final_model = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC(C=1.6, random_state=25, tol=1e-5))])

final_model.fit(all_text, all_target)

joblib_file = "trained_model.pkl"
joblib.dump(final_model, joblib_file)

print("done!")