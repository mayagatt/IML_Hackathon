import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


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
    while len(lines) > 0 :
        rand = random.randint(1, 5)
        if len(lines) >= rand:
            group = " ".join(lines[0:rand])
            del lines[0:rand]
            if len(group) > 5:
                line_groups.append(" ".join(lines[0:rand]))

    train, test = train_test_split(line_groups, test_size=0.2)
    return train, test


def create_train_test_data():
    '''
    create 2 pandas data frames that contains two columns - "string_group" contains a string
    and "project_number" contains the number code of the file the string came from
    :return: train_df, test_df
    '''
    project_list = ['building_tool_all_data.txt', 'espnet_all_data.txt', 'horovod_all_data.txt',
                    'jina_all_data.txt', 'PaddleHub_all_data.txt', 'PySolFC_all_data.txt', 'pytorch_geometric_all_data.txt']
    total_train = []
    total_test = []
    random.seed(22)
    for i, p in enumerate(project_list):
        train, test = split_train_test(p)
        train_df = pd.DataFrame({"string_group" : train, "project_number" : i})
        test_df = pd.DataFrame({"string_group" : test, "project_number" : i})

        total_train.append(train_df)
        total_test.append(test_df)

    total_train_df = pd.concat(total_train, ignore_index=True)
    total_test_df = pd.concat(total_test, ignore_index=True)
    return total_train_df, total_test_df
