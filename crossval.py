__author__ = 'eliashamida'

#!/usr/bin/python

'''
crossval.py is meant to be run as a standalone script to perform cross validation and print scores.
Run this script from the command line as follows:

        script              sys.argv[1]          sys.argv[2]  sys.argv[3]      sys.argv[4]     sys.argv[5]   sys.argv[6]    sys.argv[7]     sys.argv[8]     sys.argv[9] (optional)
python crossval.py 'ml/features/p_fmatrix.npy'    'MP:0004031'  '5'              '5'           'directory/'      'ba'       'kernel'         "{1:1}"        '0.01'

'''

from ml import *

import sys
import ast
from collections import namedtuple

def cross_val(params):

    ## Read in necessary objects ---------------------------------------------------------------------------------------
    genepheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    mp_ont = MPFromFile('data/ontologies/mp_ont.csv')

    ## Get feature matrix and classification labels --------------------------------------------------------------------
    X = np.load(params.featurepath)
    y = getLabels(params.phenotype, genepheno.copy(deep=True), mp_ont)
    X, y = removeMissingExamples(X, y)

    ## Set up and do gridsearch ----------------------------------------------------------------------------------------
    print('Initiating cross validation procedure ...')

    s=None
    if params.metric == 'ba':
        s = return_scorer(balanced_accuracy, greater_is_better=True)
    if params.metric == 'roc':
        s = return_scorer(roc_auc_score, sample_weight=1)
    if params.metric == 'ratio':
        s = return_scorer(spc_sens_ratio, greater_is_better=True)
    if params.metric == 'sensitivity':
        s = return_scorer(sensitivity, greater_is_better=True)
    if params.metric == 'specificity':
        s = return_scorer(specificity, greater_is_better=True)

    clf = None
    if params.kernel == 'linear':
        clf = svm.SVC(kernel=params.kernel, class_weight = params.weight)
    elif params.kernel == 'rbf':
        clf = svm.SVC(kernel=params.kernel, gamma=params.gamma, class_weight = params.weight)

    scores = cross_validation.cross_val_score(clf, X, y, cv = params.cv, n_jobs = params.jobs, scoring = s)

    ## Save output to file
    print( 'Scores: mean %0.3f, +/- %0.3f' % ( scores.mean(), scores.std()*2 ) )


## Parse command line arguments ----------------------------------------------------------------------------------------
if len(sys.argv)>10:
    sys.exit('Error: wrong number of arguments')

parameters = namedtuple('parameters', 'featurepath phenotype cv jobs outdir metric kernel weight gamma')

featurepath = sys.argv[1]
phenotype = sys.argv[2]
cv = int( sys.argv[3] )
jobs = int( sys.argv[4] )
outdir = sys.argv[5]

metric = sys.argv[6]
kernel = sys.argv[7]
weight = ast.literal_eval( sys.argv[8] )
gamma = None
if len(sys.argv) > 8: gamma = float( sys.argv[9] )

params = parameters(featurepath, phenotype, cv, jobs, outdir, metric, kernel, weight, gamma)

## Call search_grid with parsed arguments in namedtuple ----------------------------------------------------------------
cross_val(params)

