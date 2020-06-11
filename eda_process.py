"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors: Hadas Nahshon, Guy Lutzker, Maya Harari, Omer Plotnik

===================================================
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import data_preprocessing

# create train and test dataframes
train, test = data_preprocessing.create_train_test_data()

# split to test, train texts and labels
train_text, train_target = train['string_group'], train['project_number']
test_text, test_target = test['string_group'], test['project_number']


def fit_model(model):
    '''

    :param model: to be fitted
    :return: fitted model
    '''
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', model)])

    text_clf.fit(train_text, train_target)
    return text_clf


def score_model(clf, test_data, test_target):
    '''

    :param clf: modle, to score
    :param test_data: to predict labels for
    :param test_target: true labels of test data
    :return: model score over test data
    '''
    score = clf.score(test_data, test_target)
    return score
