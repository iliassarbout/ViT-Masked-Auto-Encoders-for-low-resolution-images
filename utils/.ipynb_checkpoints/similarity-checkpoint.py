import torch

def frobenius_neg(a,b): #compute negative of frobenius norm of difference of two matrix
  x = a-b
  return(torch.sqrt(torch.trace(x @ x.T)))

def frobenius_batch_neg(a,b): #compute negative of frobenius norm of difference of a reference matrix with a batch of matrix
  x = a-b #a is 2D reference matrix, b is 3D list of matrix
  return(-torch.sqrt((x @ torch.transpose(x,1,2)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)))

def frobenius_sim(f0,f): #f are rel att matrixes outputs from model
  f = torch.mean(torch.mean(f,dim=2),dim=1)
  similarities = frobenius_batch_neg(f0,f)

def frobenius_sim_allmatrices(a,b): #component wise frobenius norm (allow to work with every depth at the same time)
  y = (a-b).reshape((b.shape[0],b.shape[1]*b.shape[2],64,64))
  y = torch.matmul(y,y.transpose(2,3)) #x2[0] @ x2.transpose(1,2)[0]
  y = y.diagonal(dim1=2, dim2=3).sum(2) #extract diagonal of matrices
  y = torch.sum(-torch.sqrt(y),dim=1)
  return(y)