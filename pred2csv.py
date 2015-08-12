#!/usr/bin/python

__author__ = 'eliashamida'

from process_data import *
import sys

# parse command line arguments
filepath = sys.argv[1]
outfilepath = sys.argv[2]

# load required data strutures
data = loadData()
EM_dict, ME_dict = createMappings(data)

# process mutagenetix predictions
df = pd.read_csv(filepath, sep='_', header=None, dtype='str')
df.columns = ['entrez','mp']
df['MGI_ID'] = df.apply(lambda r: EM_dict[r.entrez], axis=1)

mp_map = pd.read_csv('data/ontologies/mp_mappings.csv')
mp_dict = mp_map.set_index('mp').T.to_dict('list')

df['mp_name'] = df.apply(lambda r: mp_dict[r.mp], axis=1)

# save to file
df.to_csv(outfilepath, index=False, header=True)