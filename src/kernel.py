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

    #print(np.linalg.eigh(K))
    return K #+ np.eye(self.n) * 1e-8