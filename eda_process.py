from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import data_preprocessing as dpre

train, test = dpre.create_train_test_data()  # create train and test dataframes
train_text = train['string_group']  # extract train text from train
train_target = train['project_number']  # extract train_text labels from train
test_text = test['string_group']  # extract test text from test
test_target = test['project_number']  # extract test_text labels from test


BUILDING_TOOL_PATH = "building_tool_all_data.txt"  # 0
JINA_PATH = "jina_all_data.txt"  # 1
HOROVOD_PATH = "horovod_all_data.txt"  # 2
ESPNET_PATH = "espnet_all_data.txt"  # 3
PYSOLFC_PATH = "PySolFC_all_data.txt"  # 4
PADDLEHUB_PATH = "PaddleHub_all_data.txt"  # 5
PYTORCH_GEOMETRIC_PATH = "pytorch_geometric_all_data.txt"  # 6
ENCODING = "utf8"

path_lst = [BUILDING_TOOL_PATH, ESPNET_PATH, HOROVOD_PATH, JINA_PATH,
            PADDLEHUB_PATH, PYSOLFC_PATH, PYTORCH_GEOMETRIC_PATH]


def fit_model(model):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', model)])

    text_clf.fit(train_text, train_target)
    return text_clf


def score_model(clf, docs_new, docs_new_target):
    score = clf.score(docs_new, docs_new_target)
    return score

