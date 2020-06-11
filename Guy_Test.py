from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.stem import PorterStemmer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors


def Model_Evaluation(model, X, y):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def split_train_test(path):
    '''
    split a file into groups of 1 to 5 rows, and than split these groups of
    lines to separate "train" and "test" lists of groups
    :param path: the files path
    :return: train - a list that contains 80% of the line-groups
             test - a list that contains 20% of the line-groups
    '''
    f = open(path, encoding="utf8")
    lines = f.readlines()
    line_groups = []
    while len(lines) > 0:
        rand = random.randint(1, 5)
        if len(lines) >= rand:
            line_groups.append(" ".join(lines[0:rand]))
            del lines[0:rand]
    train, test = train_test_split(line_groups, test_size=0.2)
    return train, test


def create_train_test_data():
    '''
    create 2 pandas data frames that contains two columns - "string_group" contains a string
    and "project_number" contains the number code of the file the string came from
    :return: train_df, test_df
    '''
    project_list = ['building_tool_all_data.txt', 'espnet_all_data.txt',
                    'horovod_all_data.txt',
                    'jina_all_data.txt', 'PaddleHub_all_data.txt',
                    'PySolFC_all_data.txt', 'pytorch_geometric_all_data.txt']
    total_train = []
    total_test = []
    for i, p in enumerate(project_list):
        train, test = split_train_test(p)
        train_df = pd.DataFrame({"string_group": train, "project_number": i})
        test_df = pd.DataFrame({"string_group": test, "project_number": i})

        total_train.append(train_df)
        total_test.append(test_df)

    total_train_df = pd.concat(total_train, ignore_index=True)
    total_test_df = pd.concat(total_test, ignore_index=True)
    return total_train_df, total_test_df


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


def fit_no_pipeline(docs_new, model):
    cv = CountVectorizer()
    X_count = cv.fit_transform(train_text)

    tf_transformer = TfidfTransformer()
    X_train_tf = tf_transformer.fit_transform(X_count)

    clf = model.fit(
        X_train_tf, train_target)
    X_new_counts = cv.transform(docs_new)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %d' % (doc, category))


def fit_with_pipeline(docs_new, docs_new_target, model):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', model)])

    text_clf.fit(train_text, train_target)
    predicted = text_clf.predict(docs_new)
    mean_pred = np.mean(predicted == docs_new_target)
    return mean_pred


sample_size = [5, 10, 100, 500, 1000, 10000, 20000, 100000]
models = [MultinomialNB(), LogisticRegression(
    solver="saga", tol=1e-4, max_iter=1e4),
          NearestNeighbors(n_neighbors=20), LinearSVC(random_state=25,
                                                      tol=1e-5, C=0.5),
          RandomForestClassifier(max_depth=5, random_state=25),
          LinearRegression()]
for model in models:
    S = []
    for m in sample_size:
        temp_test = train.sample(m)
        new_strings = temp_test['string_group']
        new_targets = temp_test['project_number']
        S.append(fit_with_pipeline(new_strings, new_targets, model))
    plt.plot(S)
    plt.title("model: ")
    plt.show()
