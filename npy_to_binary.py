__author__ = 'Dirk Eddelbuettel'

#!/usr/bin/python
#
# read a numpy file, and write a simple binary file containing
#   two integers 'n' and 'k' for rows and columns
#   n times k floats with the actual matrix
# which can be read by any application or language that can read binary

import struct
import numpy as np

inputfile = "data/processed_data/adj2_string.npy"
outputfile = "data/processed_data/adj2_string.bin"

# load from the file
mat = np.load(inputfile)

# create a binary file
binfile = file(outputfile, 'wb')
# and write out two integers with the row and column dimension
header = struct.pack('2I', mat.shape[0], mat.shape[1])
binfile.write(header)
# then loop over columns and write each
for i in range(mat.shape[1]):
    data = struct.pack('%id' % mat.shape[0], *mat[:,i])
    binfile.write(data)
    print( '%0.1f' % (float(i)*100/mat.shape[1]) + '%' )
binfile.close()