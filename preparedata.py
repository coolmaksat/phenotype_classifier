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

def features_from_scratch(data):
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
    itp_ints = parseInterproFile('data/ontologies/ParentChildTreeFile.txt')
    itp_ont = returnIGraph(itp_ints, directed=True)
    itp_fmatrix = returnInterproMatrix(itp_ont, data['gene_pheno'].copy(deep=True), data['interpro'].copy(deep=True))

    ## save feature matrices
    np.save('ml/features/p_fmatrix.npy', p_fmatrix)
    np.save('ml/features/go_fmatrix.npy', go_fmatrix)
    np.save('ml/features/gobp_fmatrix.npy', gobp_fmatrix)
    np.save('ml/features/gomf_fmatrix.npy', gomf_fmatrix)
    np.save('ml/features/gocc_fmatrix.npy', gocc_fmatrix)
    np.save('ml/features/itp_fmatrix.npy', itp_fmatrix)

def features_from_pkl():
    '''
    makes feature matrices for distances to pathways, all GO ontologies and interpro domains and saves them as np.arrays
    :param data: list of pd.DataFrames containing processed data
    :return None
    '''

    interactions = pd.read_pickle('data/processed_data/interactions.pkl')
    gene_pheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    pathways = pd.read_pickle('data/processed_data/pathways.pkl')
    go = pd.read_pickle('data/processed_data/go.pkl')
    interpro = pd.read_pickle('data/processed_data/interpro.pkl')

    ## pathway distances
    phys_graph = returnNetwork(interactions, 'physical')
    p_fmatrix = returnPathwayDistanceMatrix(phys_graph, gene_pheno, pathways)
    np.save('ml/features/p_fmatrix.npy', p_fmatrix)
    del p_fmatrix

    ## GO ontologies
    go_ints = returnGoFromFile('data/ontologies/go')

    go_ont = returnIGraph(go_ints[['term_B','term_A']], directed=True)
    go_fmatrix = returnGoMatrix(go_ont, gene_pheno, go)
    np.save('ml/features/go_fmatrix.npy', go_fmatrix)
    del go_fmatrix

    gobp_ont = returnGOOntology(go_ints, 'biological_process')
    gobp_fmatrix = returnGoMatrix(gobp_ont, gene_pheno, go)
    np.save('ml/features/gobp_fmatrix.npy', gobp_fmatrix)
    del gobp_fmatrix

    gomf_ont = returnGOOntology(go_ints, 'molecular_function')
    gomf_fmatrix = returnGoMatrix(gomf_ont, gene_pheno, go)
    np.save('ml/features/gomf_fmatrix.npy', gomf_fmatrix)
    del gomf_fmatrix

    gocc_ont = returnGOOntology(go_ints, 'cellular_component')
    gocc_fmatrix = returnGoMatrix(gocc_ont, gene_pheno, go)
    np.save('ml/features/gocc_fmatrix.npy', gocc_fmatrix)
    del gocc_fmatrix

    ## interpro domains
    itp_ints = parseInterproFile('data/ontologies/ParentChildTreeFile.txt')
    itp_ont = returnIGraph(itp_ints, directed=True)
    itp_fmatrix = returnInterproMatrix(itp_ont, gene_pheno, interpro)
    np.save('ml/features/itp_fmatrix.npy', itp_fmatrix)
    del itp_fmatrix




'''
## Read in and process raw data, and save pandas.DataFrame's to 'data/...'
dat = getRawData()
dat = processData(dat)
saveData(dat)

## Make feature matrices and save them to 'features/...'
features_from_scratch(dat)

## Write MP terms to two files: first all terms, second filtered (>6 training exmaples for each term)
mp_ont = MPFromFile('data/ontologies/mp_ont.csv')
with open('data/mp_ids/mplist.txt', 'w') as output:
    for term in mp_ont.vs['name']:
        output.write('%s\n' % term)

with open('data/mp_ids/mplist_trunc.txt', 'w') as output:
    for term in mp_ont.vs['name']:
        if sum(getLabels(term,dat['gene_pheno'],mp_ont))>6:
            output.write('%s\n' % term)
'''