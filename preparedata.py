__author__ = 'eliashamida'

#!/usr/bin/python

'''

TBC

preparedata.py can be run once to compute and save all the necessary ontologies, feature matrices, labels and mp terms
for later prediction.
'''

from process_data import *
from funs import *


def saveData(data):
    '''
    saves 'data' to separate .csv files
    :param data: list of pandas.DataFrames
    :return None
    '''

    ## Does not work to save processed data in .csv's because of formatting. See .pkl below.
    #data['gene_pheno'].to_csv('data/processed_data/gene_pheno.csv',sep='\t',index=False)
    #data['interpro'].to_csv('data/processed_data/interpro.csv',sep='\t',index=False)
    #data['interactions'].to_csv('data/processed_data/interactions.csv',sep='\t',index=False)
    #data['pathways'].to_csv('data/processed_data/pathways.csv',sep='\t',index=False)
    #data['go'].to_csv('data/processed_data/go.csv',sep='\t',index=False)
    #data['entrez'].to_csv('data/processed_data/entrez.csv',sep='\t',index=False)

    ## do the following for each file. Not done here because .pkl files were created on a different system, with
    ## non-matching protocols
    #data['gene_pheno'].to_pickle('data/processed_data/gene_pheno.pkl')

def makeFeatureMatrices(data):
    '''
    makes feature matrices for distances to pathways, all GO ontologies and interpro domains and saves them as np.arrays
    :param data: list of pd.DataFrames containing processed data
    :return None
    '''

    ## pathway distances
    phys_graph = returnNetwork(data['interactions'].copy(deep=True), 'physical')
    p_fmatrix = returnPathwayDistanceMatrix(phys_graph, data['gene_pheno'].copy(deep=True), data['pathways'].copy(deep=True))

    ## GO ontologies
    go_obo = parseOboFile('data/ontologies/go-basic.obo')
    go_obo.to_csv('go')
    go = returnGoFromFile('data/ontologies/go')
    go_ont = returnIGraph(go[['term_B','term_A']], directed=True)
    gobp_ont = returnGOOntology(go, 'biological_process')
    gomf_ont = returnGOOntology(go, 'molecular_function')
    gocc_ont = returnGOOntology(go, 'cellular_component')

    go_fmatrix = returnGoMatrix(go_ont, data['gene_pheno'].copy(deep=True), data['go'].copy(deep=True))
    gobp_fmatrix = returnGoMatrix(gobp_ont, data['gene_pheno'].copy(deep=True), data['go'].copy(deep=True))
    gomf_fmatrix = returnGoMatrix(gomf_ont, data['gene_pheno'].copy(deep=True), data['go'].copy(deep=True))
    gocc_fmatrix = returnGoMatrix(gocc_ont, data['gene_pheno'].copy(deep=True), data['go'].copy(deep=True))


    ## interpro domains
    itp_ints = parseInterproFile('ParentChildTreeFile.txt')
    itp_ont = returnIGraph(itp_ints, directed=True)
    itp_fmatrix = returnInterproMatrix(itp_ont, data['gene_pheno'].copy(deep=True), data['interpro'].copy(deep=True))

    ## save feature matrices
    np.save('ml/features/p_fmatrix.npy', p_fmatrix)
    np.save('ml/features/go_fmatrix.npy', go_fmatrix)
    np.save('ml/features/gobp_fmatrix.npy', gobp_fmatrix)
    np.save('ml/features/gomf_fmatrix.npy', gomf_fmatrix)
    np.save('ml/features/gocc_fmatrix.npy', gocc_fmatrix)
    np.save('ml/features/itp_fmatrix.npy', itp_fmatrix)

## Read in and process raw data, and save pandas.DataFrame's to 'data/...'
dat = getRawData()
dat = processData(dat)
saveData(dat)

## Make feature matrices and save them to 'features/...'
makeFeatureMatrices(dat)

## Write MP terms to two files: first all terms, second filtered (>6 training exmaples for each term)
mp_ont = MPFromFile('data/ontologies/mp_ont.csv')
with open('data/mp_ids/mplist.txt', 'w') as output:
    for term in mp_ont.vs['name']:
        output.write('%s\n' % term)

with open('data/mp_ids/mplist_trunc.txt', 'w') as output:
    for term in mp_ont.vs['name']:
        if sum(getLabels(term,dat['gene_pheno'],mp_ont))>6:
            output.write('%s\n' % term)