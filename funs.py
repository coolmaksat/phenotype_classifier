__author__ = 'eliashamida'

'''
funs.py contains essential functions used in other scripts. Functions here are mainly used to create the feature
matrices and labels.
'''

from parse_files import *

import igraph as ig
from itertools import chain
import numpy as np
import pandas as pd
from collections import Counter

def histFromDF(df, col_name):
    ## plot histogram of frequency of occurrence of entries in a given column (col_name)
    ## in a DataFrame (df)
    counts = Counter(df[col_name])
    df = pd.DataFrame.from_dict(counts, orient='index')
    df.plot(kind='bar')

def histOfSlice(pfm, i):
    plt.hist(pfm[i,][pfm[i,]!=1e6], bins=200)

def returnNetwork(interactions, filter):
    ## in: pandas dataframe, interaction partners in cols 'gene_A' and 'gene_B', and 'int_type' col
    ## out: tuple of igraph objects, genetic and physical interaction networks

    # separate physical and genetic interaction networks
    int = interactions.loc[interactions['int_type']==filter,['gene_A','gene_B']] #  23658 interactions
    int = int.reset_index(drop=True)

    # Make genetic interaction network
    g = returnIGraph(int, False)

    return(g)

def returnIGraph(ints, directed):
    ## in: pandas dataframe, interaction partners in two separate columns 'gene_A' and 'gene_B'
    ## out: igraph object, interaction network based on input argument

    # store vertex names and edge pairs
    colA, colB = ints.columns[0:2]
    names = ints[colA].append(ints[colB])
    names = names.drop_duplicates().tolist() # store all non-duplicate gene names in list
    idxs = dict(zip(names, range(len(names)))) # dictionary mapping gene names to gene indices
    edges = [(ints.iloc[i,0],ints.iloc[i,1]) for i in ints.index] # list of interactions with entrez IDs
    iedges = [(idxs[e[0]], idxs[e[1]]) for e in edges] # list of interactions with index IDs

    # initialize igraph object
    g = ig.Graph(len(names), directed=directed)
    g.add_edges(iedges)
    g.vs['name'] = names # add entrez_id's ad properties to vertices
    if 'weight' in ints.columns:
        g.es['weight'] = ints.weight

    return(g)

def shortestPathToPathway(graph, id, p_comps, avPathLength):
    ## in: (igraph object, str, list), graph over which to search, name of protein, list of members of a pathway
    ## out: int, value of shortest path length of 'id' to any members of 'p_comps'

    l = list(map(lambda x: len(graph.get_shortest_paths(id, x)[0]), p_comps)) # 1 if same, 0 if no path
    l = [e if e!=0 else avPathLength for e in l]
    return(min(l))

def returnDistToPathways(graph, id, p_proteins, avPathLength):
    ## in: (igraph object, int, list), interaction network used to determine distance between 'id' and proteins
    ## in 'pathways'
    ## out: 1-d np.array, distance of 'id' to each pathway in 'pathways'

    # calculate distances to all pathways
    dist_pathways = list(map(lambda x: shortestPathToPathway(graph, id, x, avPathLength), p_proteins))

    return(np.matrix(dist_pathways))

def returnPathwayDistanceMatrix(graph, gene_pheno, pathways):
    ## in: (igraph object, list, dataframe), interaction network to define distance of proteins in 'ids' to 'pathways'
    ## proteins
    ## out: np.ndarray, examples in rows and pathways (features) in columns, array contains distances

    ## pre-process pathways data frame and create list of proteins to map over for returnDistToPathways
    # remove from pathways dataframe rows of proteins not in graph. 767/2549 present in phys_graph
    #s = [True if x in graph.vs['name'] else False for x in pathways.entrez_id] # boolean list of pathway proteins in graph
    s = pathways.entrez_id.isin(graph.vs['name'])
    pathways = pathways[s]
    pathways = pathways.reset_index(drop=True)

    # create list of lists for proteins all pathways: 219/293 pathways have been kept due to step above
    p_names = list(pathways.pathway_id.drop_duplicates()) # store list of pathway names
    p_proteins = [pathways.entrez_id[pathways.pathway_id==p] for p in p_names] # store list of lists of proteins in each pathway

    ## create 'id' list to loop over entrez_id's and initialize feature matrix
    #t = gene_pheno.entrez_id.isin(graph.vs['name'])
    #gene_pheno = gene_pheno[t]
    ids = gene_pheno.entrez_id.tolist() # list of unique protein entrez_id's
    m = np.zeros([len(gene_pheno), len(p_names)]) # initialize feature matrix

    avPathLength = graph.average_path_length(directed=True) # get average path length of graph

    for id,i in zip( ids,range(len(ids)) ):
        if id in graph.vs['name']:
            m[i,] = returnDistToPathways(graph, id, p_proteins, avPathLength)
        else:
            m[i,] = 0
        print( '%0.1f' % (float(i)*100/len(ids)) + '%' )

    return(m)

def returnInterproIndicator(itp_ont, interpro_ids):
    '''
    :param itp_ont: igraph object, ontology of interpro IDs
    :param interpro_ids: list, interpro IDs defining a marker to get children for
    :return: 1d np.array, interpro_ids and their children set to 1 in array, indexing according to itp_ont.vs
    '''
    indicator = np.zeros([1, len(itp_ont.vs)]) # initialize 1d np.array

    itp_classes = itp_ont.neighborhood(interpro_ids, order=len(itp_ont.vs), mode='in')
    itp_classes = list( set( list(chain.from_iterable(itp_classes)) ) ) # get unique elements in list form

    indicator[0,itp_classes] = 1

    return(indicator)

def returnInterproMatrix(itp_ont, gene_pheno, interpro):
    '''
    returns feature matrix for interpro class membership
    :param itp_ont: itp ontology as igraph object
    :param gene_pheno: DataFrame of entrez_id's
    :param itp: DataFrame which maps entrez_id's to interpro_id's
    :return m: np.array indicating Interpro class membership (cols) for each entrez_id (rows)

    '''
    interpro = interpro[pd.notnull(interpro.entrez_id)] # remove NaNs
    itp = interpro.reset_index(drop=True)

    # remove rows from gene_pheno dataframe entries not in interpro dataframe.entrez_id
    #s = gene_pheno.entrez_id.isin(interpro.entrez_id)
    #gene_pheno = gene_pheno[s]
    #gene_pheno = gene_pheno.reset_index(drop=True)

    # remove rows from interpro dataframe entries not in itp_ont ontology
    t = interpro.interpro_id.isin(itp_ont.vs['name'])
    interpro = interpro[t]
    interpro = interpro.reset_index(drop=True)

    # initialize feature matrix
    m = np.zeros([len(gene_pheno), len(itp_ont.vs)])

    i=0
    for id in gene_pheno.entrez_id.tolist():
        interpro_ids = interpro.interpro_id[interpro.entrez_id==id].tolist() # get all interpro_ids for this entrez_id
        if len(interpro_ids) > 0:
            m[i] = returnInterproIndicator(itp_ont, interpro_ids)
        else:
            m[i] = 0
        i+=1
        print( '%0.1f' % (float(i)*100/len(gene_pheno)) + '%' )

    return(m)

def returnGoIndicator(go_ont, go_ids):
    '''
    :param go_ont: igraph object, ontology of GO IDs
    :param go_ids: list, interpro IDs defining a marker to get children for
    :return: indicator: 1d np.array of GO_id's and their children set to 1 in array, indexing according to go_ont.vs
    '''
    indicator = np.zeros([1, go_ont.vcount()]) # initialize 1d np.array

    go_classes = go_ont.neighborhood(go_ids, order=go_ont.vcount(), mode='in')
    go_classes = list( set( list(chain.from_iterable(go_classes)) ) ) # get unique elements in list form

    indicator[0,go_classes] = 1

    return(indicator)

def returnGoMatrix(go_ont, gene_pheno, go):
    '''
    returns feature matrix for GO class membership
    :param go_ont: go ontology as igraph object
    :param gene_pheno: DataFrame of entrez_id's
    :param go: DataFrame which maps entrez_id's to GO_id's
    :return m: np.array indicating GO class membership (cols) for each entrez_id (rows)

    '''

    # remove rows from gene_pheno dataframe entries not in go dataframe.marker_id
    #s = gene_pheno.entrez_id.isin(go.entrez_id) # 8306/8670 positive
    #gene_pheno = gene_pheno[s]
    #gene_pheno = gene_pheno.reset_index(drop=True)

    # remove rows from go dataframe entries not in go_ont ontology
    t = go.GO_id.isin(go_ont.vs['name']) # 87,319/29,2019 positive
    go = go[t]
    go = go.reset_index(drop=True)

    # initialize feature matrix
    m = np.zeros([len(gene_pheno), go_ont.vcount()])

    i=0
    for id in gene_pheno.entrez_id.tolist():
        go_ids = go.GO_id[go.entrez_id==id].tolist() # get all go_id's for this entrez_id
        if len(go_ids) > 0:
            m[i] = returnGoIndicator(go_ont, go_ids)
        else:
            m[i] = 0
        i+=1
        print( '%0.1f' % (float(i)*100/len(gene_pheno)) + '%' )

    return(m)


def returnNetworkIndices(graph, gene_pheno):
    '''
   # returns 1d np.array of len(gene_pheno) with indices in PPI (as represented by igraph object)
   # :param graph: igraph object, PPI
   # :param gene_pheno: pd.DataFrame, marker ID's
   # :return index_array: 1d np.array of indices of markers in PPI graph
    '''
    # initialize index_array
    index_array = np.zeros([len(gene_pheno)])

    for i,id in zip(range(len(gene_pheno)), gene_pheno.entrez_id.tolist()):
        if id in graph.vs['name']:
            index_array[i] = graph.vs['name'].index(id)
        else:
            index_array[i] = None

    return index_array


def returnGoFromFile(filename):
    '''
    returns GO dataframe from GO interactions filefile (parsed version of OBO file)
    :param filename: .csv file
    :return pd.DataFrame: GO interactions
    '''
    go = pd.read_csv(filename, sep='\t') # 80344 rows with data
    go = go[:80344]
    go = go.drop('Unnamed: 0', axis=1)
    cols = go.columns.tolist()
    newcols = cols[1:2]+cols[:1]+cols[2:4]
    go = go[newcols]

    return(go)

def returnGOOntology(go, namespace):
    '''
    returns chosen GO ontology (defined by 'namespace') as an igraph object
    :param go: pd.DataFrame of interactions and namespaces
    :param namespace: 'biological_process', 'molecular_function' or 'cellular_component'
    :return: go_ont: igraph object for chosen ontology
    '''
    # subset 'go' DataFrame with chosen namespace
    sub_go = go[['term_B','term_A']][go.namespace==namespace]
    sub_go = sub_go.reset_index(drop=True)

    go_ont = returnIGraph(sub_go, directed=True) # create directed igraph object

    return(go_ont)

def returnMPOnt(filename):
    '''
    :param filename: destination path of mammalian phenotype ontology OBO file
    :return mp_onet: igraph object storing MP ontology
    '''
    mp = parseMPFile(filename) # ALTERNATIVE IDs NOT TAKEN INTO ACCOUNT (this is o.k.)
    mp_ont = returnIGraph(mp, True)
    return(mp_ont)

def MPFromFile(filename):
    '''
    returns MP ontology as igraph object
    :param filename: destination path of mammalian phenotype ontology interactions .csv file (parsed version of
    OBO file)
    :return mp_onet: igraph object storing MP ontology
    '''
    mp = pd.read_csv(filename, sep='\t', header=False)
    mp_ont = returnIGraph(mp, True)
    return(mp_ont)

def getLabels(pheno_id, gene_pheno, mp_ont):
    '''
    returns list of length len(gene_pheno) with class labels (1 or 0) based on pheno_id
    :param str, DataFrame, igraph object
    :return list
    '''
    pos = mp_ont.neighborhood(pheno_id, order=len(mp_ont.vs), mode='in') # get list of class + subclass indices
    positives = mp_ont.vs[pos]['name'] # translate to class names (mp_id's)
    labels = [None]*len(gene_pheno) # initialize list for labels

    for i,pheno_series in zip(range(len(gene_pheno)),gene_pheno.phenotype):
        if any([x in positives for x in pheno_series]):
            labels[i] = 1
        else:
            labels[i] = 0

    return( np.array(labels) ) # return labels as 1d np.array