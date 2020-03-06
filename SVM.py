"""
Nick Smith
CS 445 - Machine Learning
HW 3 - Support Vector Machines
       And Feature Selection
Due: Tuesday, February 12 2019, 5pm
"""

# LIBRARIES
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn import model_selection as ms
from sklearn import svm
import matplotlib.pyplot as plt

from datetime import datetime


# ------------------------ #
#       EXPERIMENT 1       #
# ------------------------ #

# DATASET
data = pd.read_csv("spambase.csv", header=None, index_col=57)

# DATA SET SPLIT
X_train, X_test, y_train, y_test = ms.train_test_split(data, data.index.values, test_size=.5)

# SCALING
scaler = preproc.StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=y_train)
X_test = pd.DataFrame(scaler.transform(X_test), index=y_test)


# CLASSIFIER
classifier = svm.SVC(kernel='linear').fit(X_train, y_train)
print('SVM Accuracy: {acc:.4f}'.format(acc=classifier.score(X_test,y_test)))

# AVERAGE PRECISION
y_score = classifier.decision_function(X_test)
average_precision = average_precision_score(y_test, y_score)
print('Average Precision-Recall score: {0:0.4f}'.format(average_precision))

# TRUE AND FALSE POSITIVE RESULTS
tpr, fpr, thresh = metrics.roc_curve(y_true=y_test,
                                     y_score=classifier.decision_function(X_test),
                                     pos_label=1)

def rocCurvePlotter(tpr, fpr, auc):
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve')
    plt.plot([0,1], [0,1], color='gray', lw=2, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("ROC Curve", fontsize=20)
    plt.xlabel('False Positives', fontsize=15)
    plt.ylabel('True Positives', fontsize=15)
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend(loc='lower right', fontsize=10)
    plt.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc),
             bbox=dict(facecolor='white'),
             horizontalalignment='center',
             fontsize=18)
    plt.show()

rocCurvePlotter(fpr, tpr, metrics.roc_auc_score(y_true=y_test,
                                                y_score=classifier.decision_function(X_test)))

# ------------------------ #
#     EXPERIMENT 2 & 3     #
# ------------------------ #

# DETERMINE FEATURE WEIGHTS
weights = [0] * 57
print("Determining Feature Weights: ")
for i in range(100):
    if i % 10 == 0:
        print ('{i}%'.format(i=i))

    # USE NEW VERSION OF DATA SET
    X_t, _, y_t, _ = ms.train_test_split(data, data.index.values, test_size=0.5)
    s = preproc.StandardScaler().fit(X_t)
    X_t = pd.DataFrame(scaler.transform(X_t), index=y_t)

    # TRAIN CLASSIFIER
    c = svm.SVC(kernel='linear').fit(X_t, y_t)

    # GET WEIGHTS
    w = np.flip(np.argsort(np.abs(c.coef_.ravel())), 0)

    for j, index in enumerate(w):
        weights[index] += (len(w) - j)

print("Feature Weights determined!")

def featureAdder(array, m=2): # INCREMENTALLY ADD NEW FEATURES
    return (array[0:i] for i in range(m, len(array)))

# INDEX FEATURES
features_by_weight = np.flip(np.argsort(weights), 0)
features_random = np.random.permutation(np.arange(classifier.coef_.shape[1]))

# DETERMINE BEST FEATURES
best_accys = [svm.SVC(kernel='linear').fit(X_train.loc[:, cols], y_train)
              .score(X_test.loc[:, cols], y_test) for cols in featureAdder(features_by_weight)]

# RANDOM FEATURES
best_random_accys = [svm.SVC(kernel='linear').fit(X_train.loc[:, cols], y_train)
                     .score(X_test.loc[:, cols], y_test) for cols in featureAdder(features_random)]

names = pd.read_csv('spambase.names', header=None, sep='\n')
bestWeights = list(zip(names.loc[features_by_weight[:5]].values.ravel()))
RandWeights = list(zip(names.loc[features_random[:5]].values.ravel()))
print("Top 5 Features:")
print('\n'.join(map(str,bestWeights)))
print("\nRandom 5 Features:")
print("\n".join(map(str,RandWeights)))

def plot_accuracy_graphs(b, r):
    plt.figure(figsize=(16,9))
    plt.suptitle('Accuracy vs Features', fontsize=24)
    plt.grid(linestyle='-', linewidth=0.5, axis='both')
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Number of Features', fontsize=18)
    plt.plot(range(len(b)), b, lw=2, color='blue', label = 'Best Features')
    plt.plot(range(len(r)), r, lw=2, color='red', label = 'Random Features')
    plt.ylim([min(min(b), min(r)) - 0.01, max(max(b), max(r)) + 0.01])
    plt.legend(loc='lower right', fontsize=10)
    plt.show()

plot_accuracy_graphs(best_accys, best_random_accys)

