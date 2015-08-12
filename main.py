__author__ = 'eliashamida'

## Import external libraries
import numpy as np

## Import external files
from preparedata import *
from ml import *

from analysis import *


## get number of annotations per marker

def return_num_labels(mp_ont, mplist):

    pos = []
    for mp in mplist:
        pos.append( mp_ont.neighborhood(mp, order=len(mp_ont.vs), mode='in') ) # get list of class + subclass indices
    pos = list(chain.from_iterable(pos))
    pos = list(set(pos)) # get unique list values

    return pos

gp = pd.read_pickle('data/processed_data/gene_pheno.pkl')
mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')
num_labels = np.array( gp.phenotype.map(lambda x: len(return_num_labels(mp_ont, x))) )
num_labels = np.reshape(num_labels, (-1,1))
np.save('data/plot/mpterms.npy', num_labels)


## pathways, balanced accuracy
df_pba = parseMetricsFile('results/scores/p_ba.txt','ba', '_', [1,2,3,5])
df_pba['bool'] = df_pba.apply(enrichment_cutoff, axis=1, thresh=0.70, col='ba')
df_pba = addPositivesToDf(df_pba, 'ml/features/p_fmatrix.npy')
df_pba.to_csv('analysis/pba_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)

## interpro signatures, auc-roc
df_iroc = parseMetricsFile('results/scores/itp_roc.txt', 'roc')
df_iroc['bool'] = df_iroc.apply(enrichment_cutoff, axis=1, thresh=0.70, col='roc')
df_iroc = addPositivesToDf(df_iroc, 'ml/features/itp_fmatrix.npy')
df_iroc.to_csv('analysis/iroc_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)

## interpro signatures, balanced accuracy
df_iba = parseMetricsFile('results/scores/itp_ba.txt', 'ba', '_', [1,2,3,5])
df_iba['bool'] = df_iba.apply(enrichment_cutoff, axis=1, thresh=0.70, col='ba')
df_iba = addPositivesToDf(df_iba, 'ml/features/itp_fmatrix.npy')
df_iba.to_csv('analysis/iba_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)

## gomf signatures, balanced accuracy
df_mfba = parseMetricsFile('results/scores/gomf_ba.txt', 'ba', '_', [1,2,3,5])
df_mfba['bool'] = df_mfba.apply(enrichment_cutoff, axis=1, thresh=0.70, col='ba')
df_mfba = addPositivesToDf(df_mfba, 'ml/features/gomf_fmatrix.npy')
df_mfba.to_csv('analysis/mfba_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)

## gocc signatures, balanced accuracy
df_ccba = parseMetricsFile('results/scores/gocc_ba.txt', 'ba', '_', [1,2,3,5])
df_ccba['bool'] = df_ccba.apply(enrichment_cutoff, axis=1, thresh=0.70, col='ba')
df_ccba = addPositivesToDf(df_ccba, 'ml/features/gocc_fmatrix.npy')
df_ccba.to_csv('analysis/ccba_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)

## gocc signatures, balanced accuracy, linear SVM
df_ccba_lin = parseMetricsFile2('results/compare_classifiers/svmlin_gocc_ba_summ.txt', 'ba', ' ', [1,2,4,5,6])
df_ccba_lin['bool'] = df_ccba_lin.apply(enrichment_cutoff, axis=1, thresh=0.70, col='ba')
df_ccba_lin = addPositivesToDf(df_ccba_lin, 'ml/features/gocc_fmatrix.npy')
df_ccba_lin.to_csv('analysis/ccba_lin_ontof.txt', columns=['phenotype','bool'], sep='\t', header=False)




for mp in df_pitp[df_pitp.ba>0.7].phenotype.tolist():
    y = getLabels(mp,genepheno,mp_ont)
    print np.sum(y)

### Feature selection

from sklearn import preprocessing
from sklearn.feature_selection import RFECV

genepheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
mp_ont = ig.Graph.Read_GraphML('data/ontologies/mp_ont.graphml')
# data import and scaling
X = np.load('ml/features/p_itp_fmatrix.npy')
y = getLabels('MP:0001844',genepheno,mp_ont)
X, y = removeMissingExamples(X, y)
X = preprocessing.scale(X)

s = return_scorer(balanced_accuracy, greater_is_better=1)
clf = svm.SVC(kernel='linear', class_weight='auto')
rfecv = RFECV(estimator=clf, step=0.1, cv=5, scoring = s)
rfecv.fit(X,y)

















X_new=X

# feature selection
X_new = LinearSVC(C=1,penalty="l1",dual=False).fit_transform(X,y)
s = return_scorer(balanced_accuracy, greater_is_better=1)
clf = svm.SVC(kernel='linear', class_weight = 'auto')
scores = cross_validation.cross_val_score(clf, X_new, y, cv = 5, n_jobs = 1, scoring = s)
scores.mean()

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
X_new = clf.fit(X, y).transform(X)
clf.feature_importances_
X_new.shape

# cross-validation
s = return_scorer(balanced_accuracy, greater_is_better=True)
clf = svm.SVC(kernel='linear', class_weight = 'auto')
scores = cross_validation.cross_val_score(clf, X_new, y, cv = 5, n_jobs = 1, scoring = s)
scores.mean()

from sklearn import tree
clf = tree.DecisionTreeClassifier()
scores = cross_validation.cross_val_score(clf, X_new, y, cv = 5, n_jobs = 1, scoring = s)
scores.mean()





## preparing plots for R
df_p1 = pd.read_csv('results/processed_scores/split_p_aa', header=None)
df_p2 = pd.read_csv('results/processed_scores/split_p_ab', header=None)
df_p = pd.concat([df_p1,df_p2], axis=1)
df_p.columns = ['phenotype','ba']
df_p.phenotype = df_p.phenotype.str.split('_').str[0]
df_p = addPositivesToDf(df_p, 'ml/features/p_fmatrix.npy')
df_p = addDepthToDf(df_p)
df_p = addTopTermToDf(df_p)
df_p.to_csv('results/processed_scores/svmrbf_p_ba.csv',index=False)

df_itp1 = pd.read_csv('results/processed_scores/split_itp_aa', header=None)
df_itp2 = pd.read_csv('results/processed_scores/split_itp_ab', header=None)
df_itp = pd.concat([df_itp1,df_itp2], axis=1)
df_itp.columns = ['phenotype','ba']
df_itp.phenotype = df_itp.phenotype.str.split('_').str[0]
df_itp = addPositivesToDf(df_itp, 'ml/features/itp_fmatrix.npy')
df_itp = addDepthToDf(df_itp)
df_itp = addTopTermToDf(df_itp)
df_itp.to_csv('results/processed_scores/svmrbf_itp_ba.csv',index=False)

df_gocc1 = pd.read_csv('results/processed_scores/split_gocc_aa', header=None)
df_gocc2 = pd.read_csv('results/processed_scores/split_gocc_ab', header=None)
df_gocc = pd.concat([df_gocc1,df_gocc2], axis=1)
df_gocc.columns = ['phenotype','ba']
df_gocc.phenotype = df_gocc.phenotype.str.split('_').str[0]
df_gocc = addPositivesToDf(df_gocc, 'ml/features/gocc_fmatrix.npy')
df_gocc = addDepthToDf(df_gocc)
df_gocc = addTopTermToDf(df_gocc)
df_gocc.to_csv('results/processed_scores/svmrbf_gocc_ba.csv',index=False)

df_gomf1 = pd.read_csv('results/processed_scores/split_gomf_aa', header=None)
df_gomf2 = pd.read_csv('results/processed_scores/split_gomf_ab', header=None)
df_gomf = pd.concat([df_gomf1,df_gomf2], axis=1)
df_gomf.columns = ['phenotype','ba']
df_gomf.phenotype = df_gomf.phenotype.str.split('_').str[0]
df_gomf = addPositivesToDf(df_gomf, 'ml/features/gomf_fmatrix.npy')
df_gomf = addDepthToDf(df_gomf)
df_gomf = addTopTermToDf(df_gomf)
df_gomf.to_csv('results/processed_scores/svmrbf_gomf_ba.csv',index=False)

df_psens = parseMetricsFile('results/svmrbf_p_sens_immune_summ.txt', 'ba', '_', [1,2,3,5])
df_psens = addPositivesToDf(df_psens, 'ml/features/p_fmatrix.npy')
df_psens = addDepthToDf(df_psens)
df_psens = addTopTermToDf(df_psens)
df_psens.to_csv('results/processed_scores/psens_ba.csv')

df_pspec = parseMetricsFile('results/svmrbf_p_spec_immune_summ.txt', 'ba', '_', [1,2,3,5])
df_pspec = addPositivesToDf(df_pspec, 'ml/features/p_fmatrix.npy')
df_pspec = addDepthToDf(df_pspec)
df_pspec = addTopTermToDf(df_pspec)
df_pspec.to_csv('results/processed_scores/pspec_ba.csv')

df_p = parseMetricsFile2('results/compare_classifiers/svmlin_p_ba_summ.txt','ba',' ',[1,2,4,5,6])
df_p = addPositivesToDf(df_p, 'ml/features/p_fmatrix.npy')
df_p = addDepthToDf(df_p)
df_p = addTopTermToDf(df_p)
df_p.to_csv('results/processed_scores/svmlin_p_ba.csv',index=False)

df_itp = parseMetricsFile2('results/compare_classifiers/svmlin_itp_ba_summ.txt','ba',' ',[1,2,4,5,6])
df_itp = addPositivesToDf(df_itp, 'ml/features/itp_fmatrix.npy')
df_itp = addDepthToDf(df_itp)
df_itp = addTopTermToDf(df_itp)
df_itp.to_csv('results/processed_scores/svmlin_itp_ba.csv',index=False)


def write_libsvm_format(X, Y, outpath):
    ## takes sklearn style X and Y (feature matrix, label vector) and writes to outpath libsvm sparse format
    with open(outpath,'w') as f:
        for j in range(X.shape[0]):
            f.write(" ".join(
                      [str(int(Y[j]))] + ["{}:{}".format(i,X[j][i])
                      for i in range(X.shape[1]) if X[j][i] != 0]))


##
df_pba = parseMetricsFile('results/scores/p_ba.txt','ba', '_', [1,2,3,5])
df_pba = addTopTermToDf(df_pba)
df_pba = df_pba[(df_pba.ba>0.7) & (df_pba.top_mp_term=='MP:0005387')]
mp_map = pd.read_csv('data/ontologies/mp_mappings.csv')
mp_dict = mp_map.set_index('mp').T.to_dict('list')
df_pba['mp_name'] = df_pba.apply(lambda r: mp_dict[r.phenotype], axis=1)
df_pba = addDepthToDf(df_pba)
df_pba.phenotype.to_csv('data/validation/mp_ontology/top_IS_phenotypes.csv', index=False)

##
df_p = parseMetricsFile2('results/compare_classifiers/svmlin_p_ba_summ.txt','ba',' ',[1,2,4,5,6])
df_p = addPositivesToDf(df_p, 'ml/features/p_fmatrix.npy')
df_p = addDepthToDf(df_p)
df_p = addTopTermToDf(df_p)
df_p = df_p[(df_p.ba>0.7) & (df_p.top_mp_term=='MP:0005387')]





## sensitivity, specificity and precision
df_psens= parseMetricsFile('results/svmrbf_p_sens_immune_summ.txt', 'sens', '_', [1,2,3,5])
df_pspec = parseMetricsFile('results/svmrbf_p_spec_immune_summ.txt', 'spec', '_', [1,2,3,5])
df_pprec = parseMetricsFile('results/svmrbf_p_prec_immune_summ.txt', 'prec', '_', [1,2,3,5])

# merge 'em!
df = df_psens.merge(df_pspec,on='phenotype').merge(df_pprec,on='phenotype')





for e in mut_entrez:
    if (df_p[df_p.phenotype==e].top_mp_term == 'MP:0005387'):
        print(e)





