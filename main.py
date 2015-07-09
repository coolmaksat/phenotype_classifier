__author__ = 'eliashamida'

## Import external libraries
import numpy as np

## Import external files
from preparedata import *
from ml import *



def main():
    ## Read in and process raw data
    raw_dat = getRawData()
    pr_dat = processData(raw_dat)
    ## Make feature matrices and save them to 'features/...'
    makeFeatureMatrices(pr_dat)

    ## Make labels
    mp_ont = MPFromFile('data/ontologies/mp_ont.csv')

    markers = [sum(getLabels(pheno_id, pr_dat['gene_pheno'].copy(deep=True), mp_ont)) for pheno_id in mp_ont.vs['name']] # mean=46.2
    MP_annotnb_dict = {mp_ont.vs['name'][i]:markers[i] for i in range(len(markers))} # dictionary that map MP terms to number of markers

    inds = np.where(np.array(markers)>1000)[0]
    inds_subset = np.random.choice(inds, size=10, replace=False)
    MPids_subset = mp_ont.vs[inds_subset.tolist()]['name']

    for i in MPids_subset:
        for j in ['itp_fmatrix.npy',
                  'gobp_fmatrix.npy',
                  'gomf_fmatrix.npy',
                  'gocc_fmatrix.npy',
                  'go_fmatrix.npy',
                  'p_fmatrix.npy']:
            scores = balanced_classify('features/'+j, i, pr_dat['gene_pheno'].copy(deep=True), mp_ont )
            file = open('out/accuracy.txt', 'a')
            file.write('%s, %s: Accuracy: %0.2f (+/- %0.2f) \n' % (i, j, scores.mean(), scores.std() * 2))
            file.close()
            print(i , j)


    ## grid search

    data = np.load('features/itp_fmatrix.npy')
    labels = getLabels('MP:0004031', data['gene_pheno'].copy(deep=True), mp_ont)


    data = np.load('features/go_fmatrix.npy')
    labels = np.load('features/labels.npy')
    data, labels = removeMissingExamples(data,labels)




    bal_acc = return_scorer(balanced_accuracy, greater_is_better=True)
    clf = svm.SVC(kernel='rbf', class_weight = {1: 10})
    scores = cross_validation.cross_val_score(clf, data, labels, cv=2, scoring=bal_acc)

    param_grid = [    {'gamma': [0.001], 'class_weight': [{1:10}]}    ]
    clf = GridSearchCV(svm.SVC(), param_grid, cv=2, scoring=bal_acc)
    clf.fit(data, labels)
    clf.best_params_
    clf.grid_scores_





    @opt.cross_validated(x=data, y=labels, num_folds=5)
    def svm_accw(x_train, y_train, x_test, y_test, w, g):
        clf = train(x_train, y_train, w, g)
        y_pred = predict(x_test, clf)
        score = accuracy_score(y_test, y_pred, sample_weight = [w if yi == 1 else 1 for yi in y_test])
        return(score)

