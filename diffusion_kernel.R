require(igraph)

dk <- function(L, is.adjacency = FALSE, beta = 0) 
{
  if (missing(L)) 
    stop("You have to provide either a laplace or a adjacency matrix to calculate the diffusion kernel.")
  method = "thresh"
  if (is.adjacency) {
    dnames <- dimnames(L)
    L = graph.adjacency(L, mode = "undirected", diag = FALSE)
    L = graph.laplacian(L, normalized = TRUE)
    dimnames(L) = dnames
  }
  if (method == "thresh") {
    eig = eigen(L)
    ncomp = round(0.8 * ncol(L)) + 1
    V = eig$vectors[, ncol(L):ncomp]
    R = V %*% diag(exp(-beta * eig$values[ncol(L):ncomp])) %*% 
      t(V)
  }
  dimnames(R) = dimnames(L)
  KernelDiag <- sqrt(diag(R) + 1e-10)
  R.norm <- R/(KernelDiag %*% t(KernelDiag))
  R.norm
}