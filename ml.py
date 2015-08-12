__author__ = 'eliashamida'

'''
ml.py contains essential functions for machine learning used in other scripts.
'''

from sklearn import svm, cross_validation, naive_bayes, linear_model, neighbors
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.grid_search import GridSearchCV

from funs import *


def removeMissingExamples(fmatrix, labels):
    '''
    removes training examples for which data are missing. Defined by row of only 0's
    :param p_fmatrix: 2d np.array, feature matrix
    :param p_labels: 1d np.array, labels
    :return:
    '''
    if fmatrix.shape[1] == 8413:
        ki1 = ~np.all(fmatrix[:,:219]==0, axis=1)
        ki2 = ~np.all(fmatrix[:,220:]==0, axis=1)
        keep_inds = map(all, zip(ki1, ki2))
    else:
        keep_inds = ~np.all(fmatrix == 0, axis=1) # 73586/136839 are not 0
    fmatrix = fmatrix[np.array(keep_inds)] # remove missing data (rows where all columns == 0)
    # len(np.where(p_fmatrix == 1e6)[0]) # 1,356,704/29,967,741
    labels = labels[np.array(keep_inds)]

    return( (fmatrix, labels) )

def balanced_accuracy(y_test, y_pred):
    '''
    returns balanced accuracy given predicted y_test and observed y_test
    :param y_pred: 1d-np.array of predicted labels
    :param y_test: 1d-np.array of observed labels
    :return bal_acc: flt, balanched accuracy
    '''
    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)

    bal_acc = (sens+spec)/2

    return( bal_acc )

def sensitivity(y_test, y_pred):
    '''

    :param y_test:
    :param y_pred:
    :return:
    '''
    true_pos = sum(y_pred[y_pred == y_test])
    false_neg = sum(1-y_pred[y_pred != y_test])

    if true_pos + false_neg == 0:
        sens = (true_pos)/float(true_pos+false_neg+1)
    else:
        sens = (true_pos)/float(true_pos+false_neg)

    return sens

def specificity(y_test, y_pred):
    '''

    :param y_test:
    :param y_pred:
    :return:
    '''
    true_neg = sum(1-y_pred[y_pred == y_test])
    false_pos = sum(y_pred[y_pred != y_test])

    if true_neg + false_pos == 0:
        spec = (true_neg)/float(true_neg+false_pos+1)
    else:
        spec = (true_neg)/float(true_neg+false_pos)

    return spec

def spc_sens_ratio(y_test, y_pred):
    '''
    returns ratio of specificity to sensitivity given predicted y_test and observed y_test
    :param y_pred: 1d-np.array of predicted labels
    :param y_test: 1d-np.array of observed labels
    :return ratio: flt, ratio of spc to sens
    '''
    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)

    return( spec/sens )

def return_scorer(scoring_func, **kwargs):
    '''
    returns a sklearn 'scorer' object that computes balanced accuracy
    :return function, 'scorer' object
    '''
    scorer = make_scorer(scoring_func, kwargs)
    return(scorer)

def fix_kernel(gram):
    '''
    fixes PPI adjacency matrix to diffusion kernel
    :param adj: 2d np.array, adjacency matrix of PPI
    :return diffusion_kernel: function
    '''
    def diffusion_kernel(x,x_train):
        '''
        'computes' diffusion kernel for particular x and y by subsetting diffusion kernel gram matrix precomputed
        using R
        :param x: 1d np.array, training or test data in the form of indices of the adjacency matrix
        :param x_train: 1d np.array, training in the form of indices of the adjacency matrix
        :return:
        '''
        gram_rowsub = gram[x,:]
        gram_allsub = gram_rowsub[:,x_train]

        return gram_allsub

    return diffusion_kernel