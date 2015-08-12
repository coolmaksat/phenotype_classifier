__author__ = 'eliashamida'

#!/usr/bin/python
'''
To be called as follows:

python predict.py '11891.0' ml/features/p_fmatrix.npy svm rbf auto MP:0004031 outdir

'''

from ml import *
from process_data import *

import sys
import os
import ast
from collections import namedtuple


def predict_class(params):
    ## Read in necessary objects ---------------------------------------------------------------------------------------
    genepheno = params.data['gene_pheno']

    ## Get feature vector for new observation --------------------------------------------------------------------------
    entrez_pd = pd.DataFrame([params.entrez],columns=['entrez_id'])
    X_new = returnPathwayDistanceMatrix(params.phys_graph, entrez_pd, params.data['pathways'])

    # if data is unavailable for this protein (i.e. not in PPI), return string
    if np.sum(X_new) == 0:
        return 'no data'

    ## Get feature matrix and classification labels for training -------------------------------------------------------
    X = np.load(params.featurepath)
    y = getLabels(params.phenotype, genepheno.copy(deep=True), mp_ont)

    ## Train the model
    print('Training the model ...')
    params.clf.fit(X,y)

    ## Predicting class of new observation
    print('Predicting class of new observation ...')
    y_pred = params.clf.predict(X_new)
    return y_pred

def return_clf_with_optimal_features(mp, classweight):
    '''

    :return:
    '''
    pars = ast.literal_eval(os.popen('head -2 ml/p_ba_opt/'+mp+'* | tail -1').read())

    clf = None
    if classweight == 'auto':
        clf = svm.SVC(kernel='rbf', class_weight = classweight, gamma = pars['gamma'])
    else:
        clf = svm.SVC(kernel='rbf', class_weight = pars['class_weight'], gamma = pars['gamma'])

    return clf


## Parse command line arguments ----------------------------------------------------------------------------------------
if len(sys.argv)<8:
    sys.exit('Error: wrong number of arguments')

parameters = namedtuple('parameters', 'entrez data mp_ont phys_graph featurepath classifier clf par1 par2 phenotype outdir')

data = loadData()
mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')
phys_graph = ig.Graph.Read_GraphML('data/processed_data/biogrid.graphml')
parameters.data = data
parameters.mp_ont = mp_ont
parameters.phys_graph = phys_graph
parameters.classifier = sys.argv[2]
parameters.featurepath = sys.argv[3]
parameters.phenotype = sys.argv[6]
parameters.outdir = sys.argv[7]

## read in validation data set -----------------------------------------------------------------------------------------
parameters.entrez = sys.argv[1]

if parameters.classifier == 'naivebayes':
    parameters.clf = naive_bayes.GaussianNB()

elif parameters.classifier == 'svm' and sys.argv[4] == 'linear':
    parameters.par1 = sys.argv[4] # kernel
    parameters.par2 = sys.argv[5] # class_weight
    parameters.clf = svm.SVC(kernel=parameters.par1, class_weight = parameters.par2)

elif parameters.classifier == 'svm' and sys.argv[4] == 'rbf':
    parameters.par1 = sys.argv[4] # kernel
    parameters.par2 = sys.argv[5] # class_weight
    parameters.clf = return_clf_with_optimal_features(parameters.phenotype, parameters.par2)

y_pred = predict_class(parameters)

print(y_pred)

if y_pred:
    with open(parameters.outdir+parameters.entrez+parameters.phenotype, 'w' ) as outfile:
        outfile.write(
            parameters.phenotype + '\n'
        )