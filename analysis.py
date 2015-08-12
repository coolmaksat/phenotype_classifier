__author__ = 'eliashamida'

import pandas as pd
import numpy as np

from funs import *

'''
analysis.py contains functions for analyzing results of the gridsearch others.
'''

def parseMetricsFile2(filename, metric, d, cols):
    '''

    :param filename:
    :param metric:
    :return:
    '''
    df = pd.read_csv(filename, squeeze=True, header=None, sep=d)
    df = df.drop(cols, axis=1)
    df.columns = ['phenotype',metric]
    df.ba = df.ba.str.replace(',', '')
    df[metric] = df[metric].astype('float')

    return(df)

def parseMetricsFile(filename, metric, d, cols):
    '''

    :param filename:
    :param metric:
    :return:
    '''
    df = pd.read_csv(filename, squeeze=True)
    df = df.str.split(d+'|'+metric+d+'|.txt').apply(pd.Series)
    df = df.drop(cols, axis=1)
    df.columns = ['phenotype',metric]
    df[metric] = df[metric].astype('float')

    return(df)

def fix_inds(inds_fixed):
    '''
    fixes inds 'inds_fixed' for function 'returnNbPositives'
    :param inds_fixed: inds for a particular feature
    :return: returnNbPositives (with fixed inds)
    '''
    def returnNbPositives(mp_term):
        '''
        returns number of positives for a certain MP term given indices of missing examples in feature matrix
        (these are not included)
        :param mp_term: string
        :return: int, number of positives for given 'mp_term'
        '''
        y = np.load('ml/labels/'+mp_term+'_labels.npy')
        y = y[inds_fixed]
        return(sum(y))

    return( returnNbPositives )

def addPositivesToDf(df, feature):
    '''
    adds column with # of positives to df
    :param df: 2-column pd.DataFrame, MP ID's (1st col), score (2nd col)
    :return: df: 3-column pd.Dataframe, # of positives in 3rd col
    '''
    ## get row indices of examples with no data
    X = np.load(feature)
    inds = ~np.all(X == 0, axis=1)

    ## get # of positives per term for each row
    returnNbPositives_fixed = fix_inds(inds)
    df['positives'] = df.phenotype.apply(returnNbPositives_fixed)

    return(df)

def fix_neighbrhd_pars(g, ord, dir):
    '''
    fixes inds 'ord' and 'dir' for function graph.neighborhood
    :return: neighbrhd (with fixed parameters)
    '''
    def neighbrhd(mp_term):
        '''
        fixes inds 'ord' and 'dir' for function graph.neighborhood
        :return:
        '''
        return len( g.get_shortest_paths(mp_term, 'MP:0000001', mode=dir)[0] )

    return(neighbrhd)

def addDepthToDf(df):
    '''
    adds columns with depth in mp ontology to df
    :param df: 2-column pd.DataFrame, MP ID's (1st col), score (2nd col)
    :return: df: 3-column pd.Dataframe, depth in mp ontology in 3rd col
    '''
    mp_ont = MPFromFile('data/ontologies/mp_ont.csv')
    neighbrhd_fixed = fix_neighbrhd_pars(mp_ont, mp_ont.vcount(), 'out')
    df['depth'] = df.phenotype.apply(neighbrhd_fixed)

    return(df)

def fix_mp_ont(graph):
    def returnTopMPTerm(mp_term):
        '''

        :param mp_term:
        :param graph:
        :return:
        '''
        nghbrd = graph.neighborhood(mp_term, order=graph.vcount(), mode='out')
        correct_node = [node for node in nghbrd if graph.get_eid(node,10928, directed=False, error=False)!=-1]
        node_name = graph.vs[correct_node[0]]['name']
        return node_name
    return returnTopMPTerm

def addTopTermToDf(df):
    '''

    :param df:
    :return:
    '''
    mp_ont = MPFromFile('data/ontologies/mp_ont.csv')
    fixed = fix_mp_ont(mp_ont)
    df['top_mp_term'] = df.phenotype.apply(fixed)

    return df


def returnNeighborhood(mp_term, graph):
    '''

    :param mp_term:
    :param graph:
    :return:
    '''
    nghbrd = graph.neighborhood(mp_term, order=graph.vcount(), mode='out')
    nghbrd_names = graph.vs[nghbrd]['name']

    return(nghbrd_names)




def meanMetricPerParentTerm(df, metric):
    '''

    :param df:
    :param metric:
    :return:
    '''

    dict_terms = {
        'adipose tissue phenotype':'MP:0005375',
        'behavior/neurological phenotype':'MP:0005386',
        'cardiovascular system phenotype':'MP:0005385',
        'cellular phenotype':'MP:0005384',
        'craniofacial phenotype':'MP:0005382',
        'digestive/alimentary phenotype':'MP:0005381',
        'embryogenesis phenotype':'MP:0005380',
        'endocrine/exocrine gland phenotype':'MP:0005379',
        'growth/size/body phenotype':'MP:0005378',
        'hearing/vestibular/ear phenotype':'MP:0005377',
        'hematopoietic system phenotype':'MP:0005397',
        'homeostasis/metabolism phenotype': 'MP:0005376',
        'immune system phenotype': 'MP:0005387',
        'integument phenotype': 'MP:0010771',
        'limbs/digits/tail phenotype': 'MP:0005371',
        'liver/biliary system phenotype':'MP:0005370',
        'mortality/aging': 'MP:0010768',
        'muscle phenotype': 'MP:0005369',
        'nervous system phenotype': 'MP:0003631',
        'normal phenotype': 'MP:0002873',
        'other phenotype': 'MP:0005395',
        'pigmentation phenotype': 'MP:0001186',
        'renal/urinary system phenotype': 'MP:0005367',
        'reproductive system phenotype': 'MP:0005389',
        'respiratory system phenotype': 'MP:0005388',
        'skeleton phenotype': 'MP:0005390',
        'taste/olfaction phenotype': 'MP:0005394',
        'tumorigenesis': 'MP:0002006',
        'vision/eye phenotype': 'MP:0005391'
    }

    mp_ont = MPFromFile('data/ontologies/mp_ont.csv')

    dict_hist = { key : np.median(df[metric][df.phenotype.isin(returnNeighborhood(dict_terms[key], mp_ont))])
                  for key in dict_terms.keys() }

    return(dict_hist)

def enrichment_cutoff(row, thresh, col):
    '''

    :param row:
    :return:
    '''
    if row[col] > thresh:
        return 1
    else:
        return 0

