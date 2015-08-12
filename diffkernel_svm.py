__author__ = 'eliashamida'


__author__ = 'eliashamida'

#!/usr/bin/python

'''
test_diff_k.py is meant to be run as a standalone script to perform cross validation using a custom kernel for the SVM,
passed as an argument.
To be called as follows:

        script              sys.argv[1]                    sys.argv[2]  sys.argv[3]      sys.argv[4]     sys.argv[5]   sys.argv[6]    sys.argv[7]     sys.argv[8]     sys.argv[9]     sys.argv[10] (last 3 optional)
python diffkernel_svm.py 'data/diffusion_kernels/...'    'MP:0004031'      '5'              '5'           'directory/'      'ba'        'svm'            'kernel'         "{1:1}"        '0.01'


'''

from ml import *

import sys
from collections import namedtuple
import ast

def cross_val(params):
    ## read data and graphs from file --------------------------------------------------------------------------------------
    gene_pheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    dk = np.load(params.input)

    g, graph = 'string', None
    if g == 'biogrid':
        graph = ig.Graph.Read_GraphML('data/processed_data/biogrid.graphml')
    elif g == 'string':
        graph = ig.Graph.Read_GraphML('data/processed_data/string_l200.graphml')

    mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')

    ## get and preprocess training data ------------------------------------------------------------------------------------
    X_inds = returnNetworkIndices(graph, gene_pheno)
    y = getLabels(params.phenotype,gene_pheno,mp_ont)

    keep_inds = ~np.isnan(X_inds)
    X_inds = X_inds[keep_inds]
    y = y[keep_inds]
    X_inds = X_inds.astype(int)

    ## get actual feature matrix by subsetting diffusion gram matrix
    X = dk[X_inds,:]

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
    if params.classifier == 'svm':
        if params.par1 == 'linear':
            clf = svm.SVC(kernel=params.par1, class_weight = params.par2)
        elif params.par1 == 'rbf':
            clf = svm.SVC(kernel=params.par1, class_weight = params.par2, gamma=params.par3)
    elif params.classifier == 'naivebayes':
        clf = naive_bayes.GaussianNB()
    elif params.classifier == 'logit':
        clf = linear_model.LogisticRegression(penalty = 'l2', C=params.par1, class_weight = params.par2)
    elif params.classifier == 'knn':
        clf = neighbors.KNeighborsClassifier()

    scores = cross_validation.cross_val_score(clf, X, y, cv = params.cv, n_jobs = params.jobs, scoring = s)

    ## Save output to file
    print( 'Scores: mean %0.3f, +/- %0.3f' % ( scores.mean(), scores.std()*2 ) )
        ## Save output to file
    with open(params.outdir+params.phenotype+'_'+'_'+params.classifier+'_%0.2f.txt' % float(scores.mean()), 'w' ) as outfile:

        outfile.write(
            params.phenotype + '\n' +
            'Scores: mean %0.3f, +/- %0.3f' % ( scores.mean(), scores.std()*2 )
        )

## parse command line arguments ----------------------------------------------------------------------------------------
if len(sys.argv) < 8:
    sys.exit('Error: wrong number of arguments!')

input = sys.argv[1]
phenotype = sys.argv[2]
cv = int( sys.argv[3] )
jobs = int( sys.argv[4] )
outdir = sys.argv[5]
metric = sys.argv[6]

parameters = namedtuple('parameters', 'input phenotype cv jobs outdir metric classifier par1 par2 par3')

# complete with unqiue parameters
if sys.argv[7] == 'naivebayes':
    params = parameters(input, phenotype, cv, jobs, outdir, metric, sys.argv[7], None, None, None)
if sys.argv[7] == 'knn':
    params = parameters(input, phenotype, cv, jobs, outdir, metric, sys.argv[7], None, None, None)
elif sys.argv[7] == 'logit':
    params = parameters(input, phenotype, cv, jobs, outdir, metric, sys.argv[7], float(sys.argv[8]), 'auto', None)
elif sys.argv[7] == 'svm':
    params = parameters(input, phenotype, cv, jobs, outdir, metric,
                        sys.argv[7], sys.argv[8], 'auto', float(sys.argv[10]))

## Call search_grid with parsed arguments in namedtuple ----------------------------------------------------------------
cross_val(params)