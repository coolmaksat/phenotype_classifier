__author__ = 'eliashamida'

from process_data import *

import pandas as pd

data = loadData()
EM_dict, ME_dict = createMappings(data)

## pre-process IMPC mutant csv file
impc = pd.read_csv('data/validation/IMPC/IMPC_genotype_phenotype.csv', usecols=['marker_accession_id','top_level_mp_term_id','top_level_mp_term_name','mp_term_id','mp_term_name'])
impc = impc[impc.top_level_mp_term_name.isin(['hematopoietic system phenotype','hematopoietic system phenotype,immune system phenotype'])]
impc = impc.drop_duplicates()
impc = impc.reset_index(drop=True)
impc['entrez_id'] = impc.apply(lambda r: ME_dict[r.marker_accession_id], axis=1)

impc_toplevel = impc[['marker_accession_id','entrez_id','top_level_mp_term_id','top_level_mp_term_name']]
impc_mpterm = impc[['marker_accession_id','entrez_id','mp_term_id','mp_term_name']]
impc_toplevel.columns = [['marker_accession_id','entrez_id','mp','top_level_mp_term_name']]
impc_mpterm.columns = [['marker_accession_id','entrez_id','mp','mp_term_name']]

impc_toplevel.to_csv('data/validation/IMPC/IMPC_toplevel.csv', index=False)
impc_mpterm.to_csv('data/validation/IMPC/IMPC_mpterm.csv', index=False)

l = []
for term in impc_mpterm.entrez_id.tolist():
    if term not in data['gene_pheno'].entrez_id.tolist():
        l.append(term)

impc_mpterm[impc_mpterm.entrez_id.isin(l)].to_csv('data/validation/IMPC/IMPC_mpterm_notintrainingset.csv', index=False)


## pre-process mutagenetix csv file
mutagenetix = pd.read_csv('data/validation/mutagenetix/mutagenetix_phenotypic_mutations_20150730.txt', sep='\t', usecols=['mgi_accession_id'])
mutagenetix['entrez_id'] = mutagenetix.apply(lambda r: ME_dict[r.mgi_accession_id], axis=1)
trainingset_entrezids = data['gene_pheno'].entrez_id.tolist()
#mutagenetix.entrez_id = mutagenetix.entrez_id.apply(lambda id: [None if id in trainingset_entrezids else id]) # remove id's present in training set
mutagenetix.to_csv('data/validation/mutagenetix/mutagenetix_entrez.csv', index=False)



## remove some columns from mutagenetix file to make it morem readable
mutagenetix = pd.read_csv('data/validation/mutagenetix/mutagenetix_phenotypic_mutations_20150730.txt', sep='\t', usecols=['mgi_accession_id','phenotypes'])
mutagenetix['entrez_id'] = mutagenetix.apply(lambda r: ME_dict[r.mgi_accession_id], axis=1)
mutagenetix.to_csv('data/validation/mutagenetix/mutagenetix_phenotypic_mutations_20150730_reduced.csv', index=False)