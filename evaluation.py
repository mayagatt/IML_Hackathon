"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors: Hadas Nahshon, Guy Lutzker, Maya Harari, Omer Plotnik

===================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from eda_process import *

m = 100000
MODELS = [MultinomialNB(),
          LogisticRegression(solver="saga", tol=1e-4, max_iter=1e4),
          KNeighborsClassifier(n_neighbors=20),
          LinearSVC(random_state=25, tol=1e-5, C=0.5),
          RandomForestClassifier(max_depth=5, random_state=25)]


def initial_model_eval():
    '''
    evaluate each model on training set of size m, m in sampel size, to generate different accuracies depending on m
    :return:
    '''
    sample_size = [1000, 5000, 10000, 20000, 50000, 100000]
    models = [MultinomialNB(),
              LogisticRegression(solver="saga", tol=1e-4, max_iter=1e4),
              KNeighborsClassifier(n_neighbors=20),
              LinearSVC(random_state=25, tol=1e-5, C=0.5),
              RandomForestClassifier(max_depth=5, random_state=25)]
    for model in models:
        S = []
        for i, m in enumerate(sample_size):
            CV = 10
            temp_test = train.sample(m)
            new_strings = temp_test['string_group']
            new_targets = temp_test['project_number']
            clf = fit_model(model)
            accuracies = cross_val_score(clf, new_strings, new_targets,
                                         scoring='accuracy', cv=CV)
            entries = []
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append(('m', fold_idx, accuracy))

            plt.subplot(2, 3, i + 1)
            cv_df = pd.DataFrame(entries,
                                 columns=[str(m), 'fold_idx', 'accuracy'])
            cv_df.groupby(str(m)).accuracy.mean()
            sns.boxplot(x=str(m), y='accuracy', data=cv_df)
            sns.stripplot(x=str(m), y='accuracy', data=cv_df,
                          size=6, jitter=True, edgecolor="gray", linewidth=1)
        plt.suptitle("Model: " + str(model.__class__.__name__), fontsize=8)
        plt.savefig(model.__class__.__name__ + "_acc.png")
        plt.show()


def run_model(model, s):
    '''

    :param model: to run
    :param s: different params, of model to run
    :return: mean accuracies of model score over CV groups
    '''
    CV = 10
    temp_test = train.sample(m)
    new_strings = temp_test['string_group']
    new_targets = temp_test['project_number']
    clf = fit_model(model)
    accuracies = cross_val_score(clf, new_strings, new_targets,
                                 scoring='accuracy', cv=CV)
    entries = []
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append(('s', fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=[str(s), 'fold_idx', 'accuracy'])
    cv_df.groupby(str(s)).accuracy.mean()
    sns.boxplot(x=str(s), y='accuracy', data=cv_df)
    sns.stripplot(x=str(s), y='accuracy', data=cv_df,
                  size=6, jitter=True, edgecolor="gray", linewidth=1)
    return np.mean(accuracies)


def log_regression():
    '''
    plot box plot of LogisticRegression
    :return:
    '''
    solves = ["liblinear", "newton-cg", "sag", "saga", "lbfgs"]
    model = None
    accuracies = []
    for idx, s in enumerate(solves):
        print(s)
        model = LogisticRegression(solver=s, tol=1e-4, max_iter=1e4)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, s)
        accuracies.append(s_acc)
    plt.suptitle(
        "Model: " + str(model.__class__.__name__) + "solver: " + solves[
            np.argmax(accuracies)] + " with accuracy: " + str(
            np.max(accuracies)), fontsize=8)

    plt.savefig(model.__class__.__name__ + "_solves.png")
    plt.show()


def knn():
    '''
    plot box plot of KNeighborsClassifier

    :return:
    '''
    ks = [5, 10, 20, 50, 100, 200]
    model = None
    accuracies = []
    for idx, k in enumerate(ks):
        print(k)
        model = KNeighborsClassifier(n_neighbors=k)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, k)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "k: " + str(
        ks[np.argmax(accuracies)]) + " with accuracy: " + str(
        np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_ks.png")
    plt.show()


def svc():
    '''
    plot box plot of linear svc
    :return:
    '''
    Cs = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 1, 2, 3, 10, 20]
    model = None
    accuracies = []
    for idx, C in enumerate(Cs):
        print(C)
        model = LinearSVC(C=C, random_state=25, tol=1e-5)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, C)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "C: " + str(
        Cs[np.argmax(accuracies)]) + " with accuracy: " + str(
        np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_Cs.png")
    plt.show()


def random_forest():
    '''
    plot box plot of RandomForestClassifier
    :return:
    '''
    depths = [1, 2, 5, 10, 20, 50]
    model = None
    accuracies = []
    for idx, d in enumerate(depths):
        print(d)
        model = RandomForestClassifier(max_depth=d, random_state=25)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, d)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "d: " + str(
        depths[np.argmax(accuracies)]) + " with accuracy: " + str(
        np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_depths.png")
    plt.show()


def ada_boost():
    '''
    plot box plot of AdaBoostClassifier
    :return:
    '''
    estimators = [SVC(probability=True, kernel='linear'), LogisticRegression(),
                  MultinomialNB()]
    model = None
    accuracies = []
    for idx, estimator in enumerate(estimators):
        print(estimator)
        model = AdaBoostClassifier(base_estimator=estimator)
        print(model)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, estimator)
        accuracies.append(s_acc)
        print('accuracie for ' + str(estimator) + ' is ' + str(s_acc))
    plt.suptitle(
        "Model: " + str(model.__class__.__name__) + "solver: " +
        str(estimator)[
            np.argmax(accuracies)] + " with accuracy: " + str(
            np.max(accuracies)), fontsize=8)

    plt.savefig(model.__class__.__name__ + "_solves.png")
    plt.show()
