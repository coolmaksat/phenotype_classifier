__author__ = 'eliashamida'

#!/usr/bin/python

'''
crossval.py is meant to be run as a standalone script to perform cross validation and print scores.
Run this script from the command line as follows:

        script              sys.argv[1]          sys.argv[2]  sys.argv[3]      sys.argv[4]     sys.argv[5]   sys.argv[6]    sys.argv[7]     sys.argv[8]
python crossval.py 'ml/features/p_fmatrix.npy'    'MP:0004031'  '5'              '5'           'directory/'      'ba'        'svm'            'kernel'

'''

from ml import *

import sys
import ast
import os
from collections import namedtuple
from sklearn import preprocessing

def cross_val(params):

    ## Read in necessary objects ---------------------------------------------------------------------------------------
    genepheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')

    ## Get feature matrix and classification labels --------------------------------------------------------------------
    X = np.load(params.featurepath)
    y = getLabels(params.phenotype, genepheno.copy(deep=True), mp_ont)
    X, y = removeMissingExamples(X, y)

    X = preprocessing.scale(X)

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

    scores = cross_validation.cross_val_score(params.clf, X, y, cv = params.cv, n_jobs = params.jobs, scoring = s)

    ## Save output to file
    print( 'Scores: mean %0.3f, +/- %0.3f' % ( scores.mean(), scores.std()*2 ) )
        ## Save output to file
    with open(params.outdir+params.phenotype+'_'+params.featurepath[13:-4]+'_'+params.classifier+'_%0.2f.txt' % float(scores.mean()), 'w' ) as outfile:

        outfile.write(
            params.phenotype + '\n' +
            'Scores: mean %0.3f, +/- %0.3f' % ( scores.mean(), scores.std()*2 )
        )

def return_clf_with_optimal_features(mp, classweight='auto'):
    '''

    :return:
    '''
    pars = ast.literal_eval(os.popen('head -2 ml/p_ba_opt/'+mp+'* | tail -1').read())

    clf = None
    if classweight == 'auto':
        clf = svm.SVC(kernel='rbf', class_weight = 'auto', gamma = pars['gamma'])
    else:
        clf = svm.SVC(kernel='rbf', class_weight = pars['class_weight'], gamma = pars['gamma'])

    return clf

## Parse command line arguments ----------------------------------------------------------------------------------------
if len(sys.argv)<8:
    sys.exit('Error: wrong number of arguments')

# naive bayes: no extra parameters. logit: par1 == 'C'. svm: par1 == 'kernel', par2 == 'weight', par3 == 'gamma'
parameters = namedtuple('parameters', 'featurepath phenotype cv jobs outdir metric classifier clf kernel')

# parameters common to all three classifiers
parameters.featurepath = sys.argv[1]
parameters.phenotype = sys.argv[2]
parameters.cv = int( sys.argv[3] )
parameters.jobs = int( sys.argv[4] )
parameters.outdir = sys.argv[5]
parameters.ls mlsfasdofjsdfmetric = sys.argv[6]
parameters.classifier = sys.argv[7]
parameters.kernel = sys.argv[8]


if parameters.classifier == 'svm':
    if parameters.kernel == 'linear':
        parameters.clf = svm.SVC(kernel=parameters.kernel, class_weight = 'auto')
    elif parameters.kernel == 'rbf':
        print(parameters.phenotype)
        parameters.clf = return_clf_with_optimal_features(parameters.phenotype)
elif parameters.classifier == 'naivebayes':
    parametersclf = naive_bayes.GaussianNB()
elif parameters.classifier == 'logit':
    parametersclf = linear_model.LogisticRegression(penalty = 'l2', C=1, class_weight = 'auto')
elif parameters.classifier == 'knn':
    parametersclf = neighbors.KNeighborsClassifier()

## Call search_grid with parsed arguments in namedtuple ----------------------------------------------------------------
cross_val(parameters)

