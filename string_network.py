__author__ = 'eliashamida'

'''
This script is designed to run once: to create the String PPI network and save it as a .graphml file to be loaded
quickly when needed.
'''

from process_data import *
from funs import *

## read files ----------------------------------------------------------------------------------------------------------
ensembl = pd.read_csv('data/raw_data/MRK_ENSEMBL.rpt', sep='\t', header=None)
string = pd.read_csv('data/raw_data/10090.protein.actions.v10.txt', sep='\t', header=0)

## pre-process data structures -----------------------------------------------------------------------------------------
# ensembl
ensembl = ensembl.drop([1,2,3,4,6,8,9,10,11,12],axis=1)
ensembl.columns = ['marker_id','ens_gene','ens_protein']
ensembl = ensembl[ensembl.ens_protein.notnull()]
ensembl = ensembl.reset_index(drop=True)
ensembl = splitCols(ensembl, 'ens_protein', ' ')
ensembl = ensembl.reset_index(drop=True)

# all other data
data = getRawData()
preProcessData(data)

# string
string.item_id_a = string.item_id_a.map(lambda x: x.lstrip('10090.'))
string.item_id_b = string.item_id_b.map(lambda x: x.lstrip('10090.'))
string.rename(columns={'score':'weight'}, inplace=True)

## create mapping between entrez_id and ensembl protein id -------------------------------------------------------------
# marker to entrez dictionary
EM_dict, ME_dict = createMappings(data)

# add entrez_id column to ensembl
ensembl['entrez_id'] = [ME_dict[x] if x in ME_dict else 'NaN'
                        for x in ensembl['marker_id']]
ensp_entrez_dict = ensembl.set_index('ens_protein')['entrez_id'].to_dict()

# replace all possible ensembl IDs by entrez IDs in string DataFrame
string['gene_A'] = string.item_id_a.map(lambda x: ensp_entrez_dict[x] if x in ensp_entrez_dict.keys() else x)
string['gene_B'] = string.item_id_b.map(lambda x: ensp_entrez_dict[x] if x in ensp_entrez_dict.keys() else x)
string = string.drop(['item_id_a','item_id_b'], axis=1)
string_cols = string.columns.tolist()
string_cols = string_cols[-2:] + string_cols[-3:-2] + string_cols[:-3]
string = string[string_cols]

string_graph = returnIGraph(string, False)

string_graph.write_graphml('data/processed_data/string.graphml')