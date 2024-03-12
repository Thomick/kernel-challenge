import numpy as np

class Kernel():
  def __init__(self, name='linear', sigma=0.1):
    self.name = name
    self.sigma = sigma

  def matrix(self, x, y):
    if self.name == 'rbf':
      X_norm = np.sum(x ** 2, axis = -1)
      Y_norm = np.sum(y ** 2, axis = -1)
      K = np.exp(-(X_norm[:,None]+Y_norm[None,:]-2*x@y.T)/(2*self.sigma**2))
    if self.name == 'linear':
      K = x @ y.T
    if self.name == 'laplacian':
      # implementation with low memory usage
      K = np.zeros((x.shape[0], y.shape[0]))
      for i in range(x.shape[0]):
        K[i] = np.sum(np.abs(x[i] - y), axis=-1)
      K = np.exp(-K / self.sigma**2)
      

    #print(np.linalg.eigh(K))
    return K #+ np.eye(self.n) * 1e-8