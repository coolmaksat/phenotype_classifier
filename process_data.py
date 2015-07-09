__author__ = 'eliashamida'

'''
process_data.py contains functions for processing raw data into useable pandas.DataFrames for later prediction.
'''

import pandas as pd
import numpy as np

def readFiles(*filenames):
    ## in: tuple, filenames (variable length)
    ## out: dictionary, intuitive names as keys and raw data tables as values

    if len(filenames)==0:
        print('No arguments passed')
        return(0)

    names_dict = {
        'MGI_GenePheno.rpt' : 'gene_pheno',
        'MGI_EntrezGene.rpt' : 'entrez',
        'MGI_InterProDomains.rpt' : 'interpro',
        'gene_association.mgi' : 'go',
        'BIOGRID-ORGANISM-Mus_musculus-3.3.124.tab2.txt' : 'interactions',
        'pathways_summary.txt' : 'pathways',
    }

    filenames = list(filenames)
    df_dict = {}

    if 'pathways_summary.txt' in filenames:
        filenames.remove('pathways_summary.txt')
        df_dict[names_dict['pathways_summary.txt']] = pd.read_csv('data/raw_data/pathways_summary.txt', sep='\t', skiprows=1)

    for filename in filenames:
        df_dict[names_dict[filename]] = pd.read_csv('data/raw_data/'+filename, sep='\t', header=None)
    return(df_dict)

def preProcessData(raw_data):
    ## in: dictionary, raw data from files
    ## out: dictionary, select columns kept and column names given

    if 'gene_pheno' in raw_data.keys():
        raw_data['gene_pheno'] = raw_data['gene_pheno'].drop([0, 1, 2, 3, 5, 7, 8], axis=1)
        raw_data['gene_pheno'].columns = ['phenotype', 'marker_id']
    if 'entrez' in raw_data.keys():
        raw_data['entrez'] = preProcessEntrez(raw_data['entrez'])
    if 'interpro' in raw_data.keys():
        raw_data['interpro'] = raw_data['interpro'].drop([1, 3], axis=1)
        raw_data['interpro'].columns = ['interpro_id', 'marker_id']
    if 'go' in raw_data.keys():
        raw_data['go'] = raw_data['go'].drop([0, 2, 3, 5, 7, 9, 10, 12, 13, 14, 15, 16], axis=1)
        raw_data['go'].columns = ['marker_id', 'GO_id', 'GO_code', 'GO_ontology', 'marker_type']
        raw_data['go'] = raw_data['go'][raw_data['go'].GO_code != 'IMP'] # remove 'IMP' annotations
        raw_data['go'] = raw_data['go'].reset_index(drop=True)
    if 'interactions' in raw_data.keys():
        raw_data['interactions'] = raw_data['interactions'].drop([0] + list(range(3, 11)) + list(range(13, 24)), axis=1)
        raw_data['interactions'] = raw_data['interactions'].ix[1:]  # remove first row
        raw_data['interactions'].columns = ['gene_A', 'gene_B', 'exp_system', 'int_type']
        raw_data['interactions']['gene_A'] = raw_data['interactions']['gene_A'] + '.0'
        raw_data['interactions']['gene_B'] = raw_data['interactions']['gene_B'] + '.0'
    if 'pathways' in raw_data.keys():
        raw_data['pathways'] = raw_data['pathways'].drop(['GeneSymbol','PW-NAME'], axis=1)
        raw_data['pathways'].columns = ['marker_id', 'pathway_id']

def preProcessEntrez(entrez):
    ## in: pandas dataframe, from Entrez file, with 'marker_id' 'type' 'entrez_id' columns
    ## out: pandas dataframe, changes: removed NaNs, filtered, added columns names etc.

    entrez = entrez.drop(list(range(1, 6)) + list(range(9, 15)), axis=1) # drop unnecessary columns
    entrez.columns = ['marker_id', 'type', 'secondary_id','entrez_id'] # add column names
    #entrez = entrez[entrez['type'] == 'Gene']  # keep only 'Gene' entries under 'type' column
    entrez = entrez[np.isfinite(entrez['entrez_id'])]  # drop rows with 'NaN' under 'entrez_id'
    entrez = entrez.reset_index(drop=True)  # reset indices

    # split entrez into three components: rows with 'marker_id' columns, rows with 'secondary_id' columns
    # lacking '|' and containing '|'
    entrez1 = entrez[['marker_id','type','entrez_id']] # separate 'entrez' into two dataframes
    entrez2 = entrez[['secondary_id','type','entrez_id']]
    entrez2.columns = ['marker_id','type','entrez_id'] # rename 'secondary_id' col to 'marker_id'

    entrez2 = entrez2[pd.notnull(entrez2['marker_id'])] # drop NaN rows
    entrez2_toparse = entrez2[entrez2['marker_id'].str.contains('\|')==True] # store rows with '|'
    entrez2 = entrez2[entrez2['marker_id'].str.contains('\|')==False] # store rows without '|'
    entrez2_parsed = splitCols(entrez2_toparse, 'marker_id', '\|') # give each 'marker_id' sep by '|' its own row

    # put entrez dataframe back together with its three components
    entrez = entrez1.append(entrez2, ignore_index=True)
    entrez = entrez.append(entrez2_parsed, ignore_index=True)
    entrez = entrez.drop_duplicates(subset='marker_id')
    entrez = entrez.reset_index(drop=True)

    entrez['entrez_id'] = entrez['entrez_id'].astype(str)

    return(entrez)

def createMappings(pr_data):
    '''
    creates dictionaries that map marker_id's to entrez_id's and vice versa
    :param pr_data, pandas.DataFrame containing data
    :return tuple of dictionaries
    '''
    entrez_marker_dict = { pr_data['entrez']['entrez_id'][k]: pr_data['entrez']['marker_id'][k]
                          for k in range(0, len(pr_data['entrez'])) }
    marker_entrez_dict = { pr_data['entrez']['marker_id'][k]: pr_data['entrez']['entrez_id'][k]
                          for k in range(0, len(pr_data['entrez'])) }

    return( (entrez_marker_dict, marker_entrez_dict) )

def addEntrezCol(data, marker_entrez_dict):
    '''
    adds a column with equivalent entrez_id to marker_id to 'pathways', 'gene_pheno' and 'interpro' DataFrames
    :param pandas.DataFrame, data without entrez_id column
    :return None (argument passed by reference)

    '''
    # translate marker_id's to entrez_id's for pathways: 24/2549 marker_id's do not have equivalent entrez_id's
    data['pathways']['entrez_id'] = [marker_entrez_dict[x] if x in marker_entrez_dict else 'NaN'
                                        for x in data['pathways']['marker_id']]
    # same as above for gene_pheno: 6306/183096 marker_id's do not have equivalent entrez_id's
    data['gene_pheno']['entrez_id'] = [marker_entrez_dict[x] if x in marker_entrez_dict else 'NaN'
                                        for x in data['gene_pheno']['marker_id']]
    data['interpro']['entrez_id'] = [marker_entrez_dict[x] if x in marker_entrez_dict else 'NaN'
                                        for x in data['interpro']['marker_id']]
    data['go']['entrez_id'] = [marker_entrez_dict[x] if x in marker_entrez_dict else 'NaN'
                                        for x in data['go']['marker_id']]

def dropEntrezDuplicates(data):
    '''
    drops marker_id column and subsequently drops duplicate rows
    :param data: list of DataFrames with marker_id column
    :return: None (argument passed by reference)
    '''
    data['gene_pheno'] = data['gene_pheno'].drop('marker_id',axis=1)
    data['pathways'] = data['pathways'].drop('marker_id',axis=1)
    data['interpro'] = data['interpro'].drop('marker_id',axis=1)
    data['go'] = data['go'].drop('marker_id',axis=1)

    data['gene_pheno'] = data['gene_pheno'].drop_duplicates()
    data['gene_pheno'] = data['gene_pheno'].reset_index(drop=True)

    data['pathways'] = data['pathways'].drop_duplicates()
    data['pathways'] = data['pathways'].reset_index(drop=True)

    data['interpro'] = data['interpro'].drop_duplicates()
    data['interpro'] = data['interpro'].reset_index(drop=True)

    data['go'] = data['go'].drop_duplicates()
    data['go'] = data['go'].drop_duplicates()

def collapseToGeneLvl(data, ME_dict):
    '''
    collapses several DataFrames to entrez_id level, returns gene_pheno DataFrame with phenotypes in list per entrez_id
    :param gene_pheno:
    :return None (argument passed by reference)
    '''

    # add entrez_id column to all relevant DataFrames
    addEntrezCol(data, ME_dict)
    dropEntrezDuplicates(data)

    # store phenotype as list for each entrez_id
    entrez_ids = data['gene_pheno'].entrez_id.drop_duplicates()
    entrez_ids = entrez_ids.reset_index(drop=True)
    l = [None]*len(entrez_ids)

    i=0
    for id in entrez_ids:
        # get all phenotype id's for each entrez_id
        l[i] = data['gene_pheno'].phenotype[data['gene_pheno'].entrez_id==id].tolist()
        i+=1

    data['gene_pheno'] = pd.DataFrame({'entrez_id': entrez_ids, 'phenotype': l })

def splitCols(df, colname, split_criterion):
    ## in: (dataframe, string, string), dataframe with split criterion and column which needs splitting
    ## out: dataframe, with new rows based on split criterion for a specific column

    s = df[colname].str.split(split_criterion).apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1) # to line up with df's index
    s.name = colname # needs a name to join
    del df[colname]
    df = df.join(s)

    return(df)

def getRawData():
    '''
    reads raw data in from files and store it in a list of pd.DataFrames
    :return: list of pd.DataFrames containing raw data in files
    '''
    # MGI_GenePheno.rpt contains MGI Marker Accession IDs and Mammalian Phenotypes, filtered for LOF mutations
    # MGI_EntrezGene.rpt: mapping between Marker IDs and Entrez IDs
    # MGI_InterProDomains.rpt: interpro domains for all genes
    # gene_association.mgi: GO annotations for all genes
    # Biogrid file: physical and genetic interactions
    data = readFiles('MGI_GenePheno.rpt',
                         'MGI_EntrezGene.rpt',
                         'MGI_InterProDomains.rpt',
                         'gene_association.mgi',
                         'BIOGRID-ORGANISM-Mus_musculus-3.3.124.tab2.txt',
                         'pathways_summary.txt')

    return(data)

def processData(data):
    '''
    cleans data and collapses to gene (entrez_id) level and saves to .pkl file
    :param data:
    :return:
    '''
    # Remove unnecessary columns and name the columns left
    preProcessData(data)

    # Create mapping between MGI Marker ID, Entrez ID
    EM_dict, ME_dict = createMappings(data)

    # Collapse all relevant DataFrames to entrez_id level, and reduce gene_pheno DataFrame according to entrez_id
    collapseToGeneLvl(data, ME_dict)

    return(data)

def loadData():
    '''
    loads data from .pkl files and stores in listl
    :return: data, list of pd.DataFrames
    '''
    gene_pheno = pd.read_pickle('data/processed_data/gene_pheno.pkl')
    entrez = pd.read_pickle('data/processed_data/entrez.pkl')
    interpro = pd.read_pickle('data/processed_data/interpro.pkl')
    go = pd.read_pickle('data/processed_data/go.pkl')
    interactions = pd.read_pickle('data/processed_data/interactions.pkl')
    pathways = pd.read_pickle('data/processed_data/pathways.pkl')

    data = {
        'gene_pheno' : gene_pheno,
        'entrez' : entrez,
        'interpro' : interpro,
        'go' : go,
        'interactions' : interactions,
        'pathways' : pathways
    }

    return(data)