from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import pandas

BUILDING_TOOL_PATH = "building_tool_all_data.txt"
JINA_PATH = "jina_all_data.txt"
HOROVOD_PATH = "horovod_all_data.txt"
ESPNET_PATH = "espnet_all_data.txt"
PYSOLFC_PATH = "PySolFC_all_data.txt"
PADDLEHUB_PATH = "PaddleHub_all_data.txt"
PYTORCH_GEOMETRIC_PATH = "pytorch_geometric_all_data.txt"
ENCODING = "utf8"

path_lst = [BUILDING_TOOL_PATH, JINA_PATH, HOROVOD_PATH, ESPNET_PATH,
            PYSOLFC_PATH, PADDLEHUB_PATH, PYTORCH_GEOMETRIC_PATH]

vocab_lst = []
Xs = []
word_count_dict = []

for path in path_lst:
    corpus = open(path, encoding=ENCODING)
    vectorizer = CountVectorizer(input='file', decode_error='ignore', strip_accents='unicode')
    X = vectorizer.fit_transform([corpus])
    vocab_lst.append(vectorizer.vocabulary_)
    Xs.append(X)

for i in range(len(Xs)):
    print(path_lst[i])  # TODO: to help track loop process, remove from final code
    word_count = []
    for key, value in vocab_lst[i].items():
        word_count.append((key, Xs[i].toarray().T[value][0]))  # tuple: (word, # of appearances)
    word_count_dict.append(dict((x, y) for x, y in word_count))  # turn tuple to dict key:value


def hist_from_dict(pathname, dict_to_plot):
    """
    Create histogram from given dictionary.
    """
    df = pandas.Series(dict_to_plot).plot(kind='area')
    plt.title("Word Count: " + pathname, fontsize=8)
    plt.tick_params(axis='x', labelsize=6)
    plt.savefig("word_count_hist_" + pathname + ".png")
    plt.show()


def create_word_histograms(file_dicts):
    """
    Create histograms for multiple dictionaries.
    :param file_dicts: List of dictionaries.
    """
    for i in range(len(file_dicts)):
        hist_from_dict(path_lst[i], file_dicts[i])


create_word_histograms(word_count_dict)
