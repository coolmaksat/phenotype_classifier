__author__ = 'eliashamida'

#!/usr/bin/python

'''
gridsearch.py is meant to be run as a standalone script to perform a gridsearch over parameters passed as arguments.
Run this script from the command line using:

        script              sys.argv[1]        sys.argv[2]                  sys.argv[3]                         sys.argv[4]     sys.argv[5]

python gridsearch.py 'features/p_fmatrix.npy' 'MP:0004031' "[{'C':[0.01],'gamma':[1e-2],'class_weight':[{1:1}]}]"     '5'     '5'

'''

from ml import *

import sys
import ast

def search_grid(featurepath, phenotype, param_grid, cross_val, cores, outdir, metric):

    ## Read in necessary objects
    genepheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')
    #mp_ont = MPFromFile('data/ontologies/mp_ont.csv')

    ## Get feature matrix and classification labels
    X = np.load(featurepath)
    y = getLabels(phenotype, genepheno.copy(deep=True), mp_ont)
    X, y = removeMissingExamples(X, y)

    ## Set up and do gridsearch
    print('Initiating gridsearch ...')

    s=None
    if metric == 'ba':
        s = return_scorer(balanced_accuracy, greater_is_better=True)
    if metric == 'roc':
        s = return_scorer(roc_auc_score, sample_weight=1)

    clf = GridSearchCV(svm.SVC(), param_grid, cv=cross_val, scoring=s, n_jobs = cores)
    clf.fit(X, y)

    ## Save output to file
    with open(outdir+phenotype+'_'+featurepath[13:-4]+'_'+metric+'_%0.2f.txt' % float(clf.best_score_), 'w' ) as outfile:

        outfile.write(
            "Best parameters set found on development set:\n" + str(clf.best_params_) + "\n \n" +
            "Best estimator found on development set:\n" + str(clf.best_estimator_) + "\n \n" +
            "Best score found on development set:\n" + str(clf.best_score_) + "\n \n" +
            "Grid scores on development set: \n \n"
        )
        for params, mean_score, scores in clf.grid_scores_:
            outfile.write( "%0.3f (+/-%0.03f) for %r \n" % ( mean_score, scores.std() / 2, params) )
        outfile.write( "\n \n \n" )

    ## Print output to console
    '''
    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print("Best estimator found on development set:")
    print(clf.best_params_)
    print("Best score found on development set:")
    print(clf.best_score_)

    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params) )
    print()
    '''


## Parse command line arguments
if len(sys.argv)!=8:
    sys.exit('Error: wrong number of arguments')

featurepath = sys.argv[1]
phenotype = sys.argv[2]
parameters = ast.literal_eval(sys.argv[3])
cv = int( sys.argv[4] )
n_jobs = int( sys.argv[5] )
dir = sys.argv[6]
metric = sys.argv[7]

## Call search_grid with parsed arguments
search_grid(featurepath, phenotype, parameters, cv, n_jobs, dir, metric)
