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
        accuracy = np.mean(accuracies)
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
