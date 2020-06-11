from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import data_preprocessing as dpre

train, test = dpre.create_train_test_data()
train_text = train['string_group']
train_target = train['project_number']
test_text = test['string_group']

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

vocab_lst = []  # list of CountVectorizer.vocabulary_
Xs = []  # list of X created by CountVectorizer.fit_transform


def files_to_strings(paths):
    """
    Function receives list of paths and returns list where each file is a single string.
    :param paths:
    :return:
    """
    docs = []
    for path in path_lst:
        with open(path, 'r', encoding=ENCODING, errors='ignore') as f:
            docs.append(f.read())
    return docs


def fit_no_pipeline(docs_new):
    cv = CountVectorizer()
    X_count = cv.fit_transform(train_text)

    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(X_count)

    clf = MultinomialNB().fit(X_train_tf, train_target)
    X_new_counts = cv.transform(docs_new)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %d' % (doc, category))


def fit_with_pipeline(docs_new, docs_new_target):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfVectorizer()),
                         ('clf', MultinomialNB())])

    text_clf.fit(train_text.tolist(), train_target.tolist())
    predicted = text_clf.predict(docs_new)
    np.mean(predicted == docs_new_target)
    return text_clf


new_strings = ["get_bottom_edges(edges, n=1)",
               "if LooseVersion(tf.__version__) < LooseVersion('1.1.0'):if LooseVersion(tf.__version__) < LooseVersion('1.1.0'):"]

fit_no_pipeline(new_strings)
