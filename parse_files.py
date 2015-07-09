__author__ = 'eliashamida'

'''
parse_files.py contains functions to parse the different ontology formats.
'''


import re
import pandas as pd
import numpy as np

def parseOboFile(filename):
    '''
    :param filename, GO .obo file to parse
    :return pd.DataFrame, with relationships and attributes in columns
    '''
    cols = ['term_A','term_B','namespace','rel']
    go_rl = pd.DataFrame(data=np.empty([1e5,4]), dtype=str,
                         columns = cols)

    with open(filename, 'r') as infile:
        currentGoTerm = None
        currentNamespace = None
        i = 0
        for line in infile:
            line = line.strip()
            if not line: continue
            if line == '[Term]' or line == '[Typedef]':
                currentGoTerm = None
                currentNamespace = None
                continue
            if line.startswith('id: ') and 'id: GO' in line:
                currentGoTerm = line.split('id: ')[1]
            if 'namespace:' in line and currentGoTerm:
                currentNamespace = line.split('namespace: ')[1]
            if 'is_a: GO' in line and currentGoTerm:
                currentRelGoTerm = re.split(': | ! ',line)[1]
                go_rl.iloc[i] = [currentRelGoTerm,
                                 currentGoTerm,
                                 currentNamespace,
                                 'is_a']
                i+=1
            if 'relationship: part_of GO' in line and currentGoTerm:
                currentRelGoTerm = re.split('part_of | ! ',line)[1]
                go_rl.iloc[i] = [currentRelGoTerm,
                                 currentGoTerm,
                                 currentNamespace,
                                 'part_of']
                i+=1


    go_rl = go_rl[:80344] # might want to change this
    cols = go_rl.columns.tolist()
    newcols = cols[1:2]+cols[:1]+cols[2:4]
    go_rl = go_rl[newcols]

    return(go_rl)

def subParseInterpro(lines, it, prefix, currentID):


    sub_intrel = pd.DataFrame(data=np.empty([0,2]), dtype=str,
                              columns = ['id_A','id_B'])

    while lines[it].split('IPR')[0] == prefix + '--':

        currentRelID = re.split(prefix+'--|::', lines[it])[1]
        temp = pd.DataFrame(data=np.empty([1,2]), dtype=str)
        temp.iloc[0,:] = [currentID, currentRelID]
        sub_intrel = sub_intrel.append(temp)
        it+=1

        if (it>8203): break

        if lines[it].split('IPR')[0] == prefix + '----':
            subsub = subParseInterpro(lines, it, prefix+'--', currentRelID)
            sub_intrel = sub_intrel.append(subsub, ignore_index=True)
            it+=len(subsub)

    return(sub_intrel)

def parseInterproFile(filename):
    '''
    :param filename, interpro relationships file to parse
    :return pd.DataFrame, with related terms in two columns
    '''
    cols = ['id_B','id_A']
    int_rel = pd.DataFrame(data=np.empty([0,2]), dtype=str,
                         columns = cols)

    with open(filename, 'r') as infile:
        currentID = None
        lines = infile.readlines()
        for it,l in enumerate(lines):
            print( '%0.2f' % (float(it)*100/len(lines)) + '%' )
            if l.startswith('IPR'):
                currentID = l.split('::')[0]
                sub_intrel = subParseInterpro(lines,it+1,'',currentID)
                int_rel = int_rel.append(sub_intrel, ignore_index=True)

    itp = int_rel[[0,1]]
    itp.columns = ['id_B','id_A']
    itp = itp.sort_index(axis=1) # sort columns alphabetically

    return(itp)

def parseMPFile(filename):
    '''
    parses mammalian phenotype ontology file to pandas.DataFrame. NOTE: Ignores alt_id's
    :param string, filename of file to parse
    :return pandas.DataFrame with related terms in two columns
    '''

    cols = ['term_B','term_A']
    mp_rl = pd.DataFrame(data=np.empty([1e5,2]), dtype=str,
                         columns = cols)

    with open(filename, 'r') as infile:
        currentMPTerm = None
        i = 0
        for line in infile:
            line = line.strip()
            if not line: continue
            if line == '[Term]':
                currentMPTerm = None
                continue
            if line.startswith('id') and 'id: MP' in line:
                currentMPTerm = line.split('id: ')[1]
            if 'is_a: MP' in line and currentMPTerm:
                currentRelMPTerm = re.split(': | ! ',line)[1]
                mp_rl.iloc[i] = [currentRelMPTerm,
                                 currentMPTerm]
                i+=1

    mp_rl = mp_rl[:13614]
    mp_rl = mp_rl.sort_index(axis=1) # sort columns alphabetically

    return(mp_rl)


