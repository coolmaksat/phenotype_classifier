

__author__ = 'eliashamida'


from process_data import *
from funs import *
from testing import *

data = getRawData()
data = processData(data)
mp_ont = MPFromFile('data/mp_ont.csv')

X_go = np.load('features/go_fmatrix.npy')
X_p = np.load('features/p_fmatrix.npy')
X_itp = np.load('features/itp_fmatrix.npy')
X_all = {'X_p':X_p, 'X_itp':X_itp}
bal_acc = return_scorer(balanced_accuracy, greater_is_better=True)
MP_terms = ['MP:0010771','MP:0002873','MP:0005378','MP:0002442','MP:0004085','MP:0004031']
gamma = [1e-6, 1e-2]

for term in MP_terms:
    for X in X_all.keys():
        for g in gamma:
            y = getLabels(term, data['gene_pheno'].copy(deep=True), mp_ont)
            x, y = removeMissingExamples(X_all[X].copy(), y.copy())
            clf = svm.SVC(kernel='rbf', gamma=g, class_weight = 'auto')
            scores = cross_validation.cross_val_score(clf, x, y, cv=5, scoring=bal_acc, n_jobs=5)
            print('%s, %s, %s: Accuracy: %0.2f (+/- %0.2f) \n' % (term, X, g, scores.mean(), scores.std() * 2))