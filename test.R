source("http://bioconductor.org/biocLite.R")
biocLite('Biobase')
biocLite('affy')
library(pathClass)
library(RcppCNPy)

setwd('~/Dropbox/MPhil/kaust/hierarch_class/')

## Read adj.bin file (adjacency matrix in binary format)
infile <- 'data/processed_data/adj.bin'
con <- file(infile, 'rb')
dim <- readBin(con, 'integer', 2)
Mat <- matrix( readBin(con, 'numeric', prod(dim)), dim[1], dim[2] )
close(con)

print(dim(Mat))

for (i in c(1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5)) {
  dk <- calc.diffusionKernel(L=Mat,
                             is.adjacency = TRUE,
                             beta=1e-6)
  
  filename = paste0('data/processed_data/dk_',i,'.npy')
  npySave('data/processed_data/dk_0.npy', dk)
  
}

dk <- calc.diffusionKernel(L=Mat,
                           is.adjacency = TRUE,
                           beta=1e-6)

npySave('data/processed_data/dk_1e-6.npy', dk)
