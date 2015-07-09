__author__ = 'eliashamida'

import optunity as opt

from funs import *
from ml import *

def balanceClasses(X_tot, y_tot, mult):
    '''
    returns feature matrix X and labels y, balanced according to 'mult' (-/+ ratio)
    :param X_tot: 2d np.array, feature matrix
    :param y_tot: 1d np.array, classification labels
    :param mult: desired -/+ ratio
    :return:
    '''
    s = len(np.where(y_tot==1)[0]) * mult # number of negative examples to take
    wy0 = np.where(y_tot==0)[0] # indices of where y==0

    # get s random indexes for negative examples. Bound number of examples to take if s larger than '0' labels
    r = [np.random.choice(wy0, size=s) if s<len(wy0) else np.random.choice(wy0, size=len(wy0))][0]
    r = np.append(r, np.where(y_tot==1)[0])
    y= y_tot[r]
    X= X_tot[r]

    mult_check = len(np.where(y==0)[0])/len(np.where(y==1)[0])
    if mult_check != mult:
        print('Problem with ratio')
        return(0)

    return( (X, y) )

def predictCV(X_tot, y_tot, mult, kernel, CV):
    '''
    returns accuracy scores for svm prediction, with 5-fold CV on X_tot feature matrix with y_tot training labels
    :param X_tot: 2d np.array, feature matrix
    :param y_tot: 1d np.array, classification labels
    :param mult:
    :return: list, accuracy scores from 5-fold CV
    '''

    # learn!
    clf = svm.SVC(kernel=kernel, class_weight = {1: mult})
    scores = cross_validation.cross_val_score(clf, X, y, cv=CV)

    return( (mult, scores) )

def predict(X_tot, y_tot, mult):

    # pick s random negative examples
    s = len(np.where(y_tot==1)[0]) * mult # number of negative examples to take
    wy0 = np.where(y_tot==0)[0] # indices of where y==0

    # get random indexes for negative examples. Bound number of examples to take if s larger than '0' labels
    r = [np.random.choice(wy0, size=s) if s<len(wy0) else np.random.choice(wy0, size=len(wy0))][0]
    r = np.append(r, np.where(y_tot==1)[0])
    y= y_tot[r]
    X= X_tot[r]

    mult = len(np.where(y==0)[0])/len(np.where(y==1)[0])

    # split data into test and training sets
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-500]]
    y_train = y[indices[:-500]]
    X_test = X[indices[-500:]]
    y_test = y[indices[-500:]]

    # learn!
    clf = svm.SVC(kernel='linear', class_weight = {1: mult})
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_weighted = accuracy_score(y_test, y_pred, sample_weight = [mult if yi == 1 else 1 for yi in y_test])

    return( accuracy_weighted )

def balanced_classify(filename, phenotype_id, gene_pheno, mp_ont, kernel, CV):
    '''
    returns accuracy score from 5-fold CV using feature in filename on phenotype_id
    :param filename: which feature to use
    :param phenotype_id: which phenotype term to classify
    :param gene_pheno: pandas.DataFrame with entrez_id's and corresponding lists of MP terms annotating it
    :param mp_ont: igraph object of MP ontology
    :return scores: 1d np.array of accuracy scores from 5-fold CV
    '''
    fmatrix = np.load(filename) # load feature matrix
    labels = getLabels(phenotype_id, gene_pheno, mp_ont)
    fmatrix, labels = removeMissingExamples(fmatrix, labels)
    X, y = balanceClasses(fmatrix, labels, 1)
    clf = svm.SVC(kernel=kernel)
    scores = cross_validation.cross_val_score(clf, X, y, cv=CV)

    return(scores)

## Functions using the optunity package

def train(x, y, w, g):
    clf = svm.SVC(kernel='rbf', gamma=g, class_weight = {1: w})
    clf.fit(x, y)
    return(clf)

def predict(x,clf):
    y = clf.predict(x)
    return(y)

def defineOjbective(data, labels):
    '''
    :param data:
    :param labels:
    :return:
    '''
    @opt.cross_validated(x=data, y=labels, num_folds=5)
    def svm_accw(x_train, y_train, x_test, y_test, hyperpar_1, hyperpar_2):
        clf = train(x_train, y_train, hyperpar_1, hyperpar_2)
        y_pred = predict(x_test, clf)
        score = accuracy_score(y_test, y_pred, sample_weight = [hyperpar_1 if yi == 1 else 1 for yi in y_test])
        return(score)

    return( svm_accw )
