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
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import data_preprocessing as dpre
from sklearn.externals import joblib
import pandas as pd
import eda_process

##train final model and save it to a pkl file

train, test = dpre.create_train_test_data()  # create train and test dataframes
all = pd.concat(train, test, ignore_index=True)
all_text = train['string_group']  # extract train text from train
model = eda_process.fit_model(MultinomialNB())
joblib_file = "trained_model.pkl"
joblib.dump(model, joblib_file)