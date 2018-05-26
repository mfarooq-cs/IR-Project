import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from prettyprint import pp
import os, re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.grid_search import GridSearchCV
from datetime import datetime as dt
#from ipy_table import *


root_path = sys.argv[1]
#top_view folders
folders = [root_path + folder + '/' for folder in os.listdir(root_path)]

print "Data Path: ", root_path
print "------------------------------------------------------------------"

#there are only 4 classes
class_titles = os.listdir(root_path)

print "Classes found: " , class_titles
print "Loading files for each class"

#list of all the files belonging to each class
files = {}
for folder, title in zip(folders, class_titles):
    files[title] = [folder + f for f in os.listdir(folder)]
    print title, ": ", len(files[title])

train_test_ratio = 0.75

print "------------------------------------------------------------------"
print "Train Ratio: " , train_test_ratio

def train_test_split(ratio, classes, files):
    """
    this method will split the input list of files to train and test sets.
    *Note: currently this method uses the simplest way an array can be split in two parts.
    Parameters
    ----------
    ratio: float
           ratio of total documents in each class assigned to the training set
    classes: list
             list of label classes
    files: dictionary
           a dictionary with list of files for each class

    Returns
    -------
    train_dic: dictionary
                a dictionary with lists of documents in the training set for each class
    test_dict: dictionary
                a dictionary with lists of documents in the testing set for each class
    """
    train_dict = {}
    test_dict = {}
    for cl in classes:
        train_cnt = int(ratio * len(files[cl]))
        train_dict[cl] = files[cl][:train_cnt]
        test_dict[cl] = files[cl][train_cnt:]
    return train_dict, test_dict

train_path, test_path = train_test_split(train_test_ratio, class_titles, files)

pattern = re.compile(r'([a-zA-Z]+|[0-9]+(\.[0-9]+)?)')

def cleanupText(path):
    """
    this method will read in a text file and try to cleanup its text.

    Parameters
    ----------
    path: str
          path to the document file
    Returns
    -------
    text_translated: str
                     cleaned up version of the raw text in the input file
    """
    from string import punctuation, digits
    text_translated = ''
    try:
        f = open(path)
        raw = f.read().lower()
        text = pattern.sub(r' \1 ', raw.replace('\n', ' '))
        text_translated = text.translate(None, punctuation + digits)
        text_translated = ' '.join([word for word in text_translated.split(' ') if (word and len(word) > 1)])
    finally:
        f.close()
    return text_translated


train_arr = []
test_arr = []
train_lbl = []
test_lbl = []

print "Text Preprocessing....."

for cl in class_titles:
    for path in train_path[cl]:
        train_arr.append(cleanupText(path))
        train_lbl.append(cl)
    for path in test_path[cl]:
        test_arr.append(cleanupText(path))
        test_lbl.append(cl)

print len(train_arr)
print len(test_arr)

vectorizer = CountVectorizer()
vectorizer.fit(train_arr)
train_mat = vectorizer.transform(train_arr)
print train_mat.shape
test_mat = vectorizer.transform(test_arr)
print test_mat.shape

tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
print train_tfmat.shape
test_tfmat = tfidf.transform(test_mat)
print test_tfmat.shape

def testClassifier(x_train, y_train, x_test, y_test, clf):
    """
    this method will first train the classifier on the training data
    and will then test the trained classifier on test data.
    Finally it will report some metrics on the classifier performance.

    Parameters
    ----------
    x_train: np.ndarray
             train data matrix
    y_train: list
             train data label
    x_test: np.ndarray
            test data matrix
    y_test: list
            test data label
    clf: sklearn classifier object implementing fit() and predict() methods

    Returns
    -------
    metrics: list
             [training time, testing time, recall and precision for every class, macro-averaged F1 score]
    """
    print "------------------------------------------------------------------"
    print "Training classifier ", clf

    metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print 'Training time: ', (end - start)

    # add training time to metrics
    metrics.append(end - start)

    print "------------------------------------------------------------------"
    print "Testing classifier "

    start = dt.now()
    yhat = clf.predict(x_test)
    end = dt.now()
    print 'Testing time: ', (end - start)

    # add testing time to metrics
    metrics.append(end - start)

    print 'Classification Analysis: '
    #     print classification_report(y_test, yhat)
    pp(classification_report(y_test, yhat))

    print 'F1 Score :', f1_score(y_test, yhat, average='macro')

    print 'Accuracy Score: ', accuracy_score(y_test, yhat)

    precision = precision_score(y_test, yhat, average=None)
    recall = recall_score(y_test, yhat, average=None)

    # add precision and recall values to metrics
    for p, r in zip(precision, recall):
        metrics.append(p)
        metrics.append(r)

    # add macro-averaged F1 score to metrics
    metrics.append(f1_score(y_test, yhat, average='macro'))

    print 'Confusion matrix:'
    print confusion_matrix(y_test, yhat)

    # plotting the confusion matrix
    #plt.imshow(confusion_matrix(y_test, yhat), interpolation='nearest')
    #plt.show()

    return metrics

metrics_dict = []
#'name', 'metrics'

# for nn in [5, 10, 15]:
for nn in [5, 10, 15]:
    print '-----------------------------------------------------------'
    print 'knn with ', nn, ' neighbors'
    knn = KNeighborsClassifier(n_neighbors=nn)
    knn_me = testClassifier(train_tfmat, train_lbl, test_tfmat, test_lbl, knn)
    metrics_dict.append({'name':'5NN', 'metrics':knn_me})
    print ' '