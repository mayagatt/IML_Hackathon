from eda_process import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns


def initial_model_eval():
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
            accuracies = cross_val_score(clf, new_strings, new_targets, scoring='accuracy', cv=CV)
            entries = []
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append(('m', fold_idx, accuracy))

            plt.subplot(2, 3, i + 1)
            cv_df = pd.DataFrame(entries, columns=[str(m), 'fold_idx', 'accuracy'])
            cv_df.groupby(str(m)).accuracy.mean()
            sns.boxplot(x=str(m), y='accuracy', data=cv_df)
            sns.stripplot(x=str(m), y='accuracy', data=cv_df,
                          size=6, jitter=True, edgecolor="gray", linewidth=1)
        plt.suptitle("Model: " + str(model.__class__.__name__), fontsize=8)
        plt.savefig(model.__class__.__name__ + "_acc.png")
        plt.show()
        # S.append(score_model(clf, new_strings, new_targets))
        # plt.plot(S)
        # plt.ylim(0, 1)
        # plt.title("Model: " + model.__class__.__name__)
        # plt.show()


m = 100000
models = [MultinomialNB(),
          LogisticRegression(solver="saga", tol=1e-4, max_iter=1e4),
          KNeighborsClassifier(n_neighbors=20),
          LinearSVC(random_state=25, tol=1e-5, C=0.5),
          RandomForestClassifier(max_depth=5, random_state=25)]


def run_model(model, s):
    CV = 10
    temp_test = train.sample(m)
    new_strings = temp_test['string_group']
    new_targets = temp_test['project_number']
    clf = fit_model(model)
    accuracies = cross_val_score(clf, new_strings, new_targets, scoring='accuracy', cv=CV)
    entries = []
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append(('s', fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=[str(s), 'fold_idx', 'accuracy'])
    cv_df.groupby(str(s)).accuracy.mean()
    sns.boxplot(x=str(s), y='accuracy', data=cv_df)
    sns.stripplot(x=str(s), y='accuracy', data=cv_df,
                  size=6, jitter=True, edgecolor="gray", linewidth=1)
    return np.mean(accuracies)


def multinomial_nb():
    pass


def log_regression():
    solves = ["liblinear", "newton-cg", "sag", "saga", "lbfgs"]
    model = None
    accuracies = []
    for idx, s in enumerate(solves):
        print(s)
        model = LogisticRegression(solver=s, tol=1e-4, max_iter=1e4)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, s)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "solver: " + solves[np.argmax(accuracies)] + " with accuracy: " + str(np.max(accuracies)), fontsize=8)

    plt.savefig(model.__class__.__name__ + "_solves.png")
    plt.show()


def knn():
    ks = [5, 10, 20, 50, 100, 200]
    model = None
    accuracies = []
    for idx, k in enumerate(ks):
        print(k)
        model = KNeighborsClassifier(n_neighbors=k)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, k)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "k: " + str(ks[np.argmax(accuracies)]) + " with accuracy: " + str(np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_ks.png")
    plt.show()


def svc():
    Cs = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 1, 2, 3, 10, 20]
    model = None
    accuracies = []
    for idx, C in enumerate(Cs):
        print(C)
        model = LinearSVC(C=C, random_state=25, tol=1e-5)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, C)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "C: " + str(Cs[np.argmax(accuracies)]) + " with accuracy: " + str(np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_Cs.png")
    plt.show()


def random_forest():
    depths = [1, 2, 5, 10, 20, 50]
    model = None
    accuracies = []
    for idx, d in enumerate(depths):
        print(d)
        model = RandomForestClassifier(max_depth=d, random_state=25)
        plt.subplot(2, 3, idx + 1)
        s_acc = run_model(model, d)
        accuracies.append(s_acc)
    plt.suptitle("Model: " + str(model.__class__.__name__) + "d: " + str(depths[np.argmax(accuracies)]) + " with accuracy: " + str(np.max(accuracies)), fontsize=8)
    plt.savefig(model.__class__.__name__ + "_depths.png")
    plt.show()
    
    
def ada_boost():
    '''

    :return:
    '''
    estimators = [SVC(probability=True, kernel='linear'), LogisticRegression(), MultinomialNB()]
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
        "Model: " + str(model.__class__.__name__) + "solver: " + str(estimator)[
            np.argmax(accuracies)] + " with accuracy: " + str(
            np.max(accuracies)), fontsize=8)

    plt.savefig(model.__class__.__name__ + "_solves.png")
    plt.show()

