__author__ = 'eliashamida'

#!/usr/bin/python

'''
test_diff_k.py is meant to be run as a standalone script to perform cross validation using a custom kernel for the SVM,
passed as an argument.
To be called as follows:

        script          sys.argv[1]          sys.argv[2]     sys.argv[3]
python test_diff_k.py   MP:0004031      dk_string_1e-05.npy  string

'''

from ml import *

import sys
from sklearn.cross_validation import KFold

## parse command line arguments ----------------------------------------------------------------------------------------
if len(sys.argv) != 4:
    sys.exit('Error: wrong number of arguments!')
mp_term = sys.argv[1]
input = sys.argv[2]
g = sys.argv[3]

## read data and graphs from file --------------------------------------------------------------------------------------
gene_pheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
dk = np.load('data/diffusion_kernels/'+input)

graph = None
if g == 'biogrid':
    graph = ig.Graph.Read_GraphML('data/processed_data/biogrid.graphml')
elif g == 'string':
    graph = ig.Graph.Read_GraphML('data/processed_data/string.graphml')

mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')

## get and preprocess training data ------------------------------------------------------------------------------------
X = returnNetworkIndices(graph, gene_pheno)
y = getLabels(mp_term,gene_pheno,mp_ont)

keep_inds = ~np.isnan(X)
X = X[keep_inds]
y = y[keep_inds]
X = X.astype(int)
ratio = 1/(float(sum(y))/len(y))
print(ratio)

## set up diffusion kernel and scoring function ------------------------------------------------------------------------
diff_kernel = fix_kernel(dk)
s = return_scorer(roc_auc_score, sample_weight=1)

## split training set into K Folds and do cross-validation -------------------------------------------------------------
kf = KFold(len(y), n_folds = 5)

for train_index, test_index in kf:

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    K_train = diff_kernel(X_train, X_train)
    K_test = diff_kernel(X_test, X_train)

    clf = svm.SVC(kernel='precomputed', class_weight = {1:ratio}) # set weight to proportion of 0's in label vector
    print('Fitting SVM model')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    print('specificity: %0.3f' % specificity(y_test, y_pred) )
    print('sensitivity: %0.3f' % sensitivity(y_test, y_pred) )
    print('balanced accuracy: %0.3f' % balanced_accuracy(y_test, y_pred) )

'''
## get and preprocess training data ------------------------------------------------------------------------------------
X = np.load('ml/features/p_fmatrix.npy')
y = getLabels(mp_term,gene_pheno,mp_ont)
X,y = removeMissingExamples(X,y)
ratio = 1/(float(sum(y))/len(y))
print(ratio)


## set up diffusion kernel and scoring function ------------------------------------------------------------------------
s = return_scorer(roc_auc_score, sample_weight=1)

## split training set into K Folds and do cross-validation -------------------------------------------------------------
kf = KFold(len(y), n_folds = 5)

for train_index, test_index in kf:

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = svm.SVC(kernel='rbf', gamma=0.0001, class_weight = {1:500}) # set weight to proportion of 0's in label vector
    print('Fitting SVM model')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('specificity: %0.3f' % specificity(y_test, y_pred) )
    print('sensitivity: %0.3f' % sensitivity(y_test, y_pred) )
    print('balanced accuracy: %0.3f' % balanced_accuracy(y_test, y_pred) )
'''