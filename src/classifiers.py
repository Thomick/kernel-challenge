import numpy as np
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import *


class LogisticRegression:
    def __init__(self, kernel, lambd=1):
        self.kernel = kernel
        self.lambd = lambd

    def fit(self, x, y):
        # y \subset {-1, 1}
        self.x = x
        self.n = len(x)
        K = self.kernel.matrix(x, x)
        K = cp.psd_wrap(K)

        alpha = cp.Variable(self.n)
        min_objective = self.lambd * cp.quad_form(alpha, K) / 2 + cp.sum(
            cp.logistic(-cp.multiply(K @ alpha, y)) / self.n)
        prob = cp.Problem(cp.Minimize(min_objective))
        prob.solve()
        self.alpha = alpha.value

    def predict(self, x):
        return self.alpha @ self.kernel.matrix(self.x, x)


class SVM:
    def __init__(self, kernel, lambd=1):
        self.kernel = kernel
        self.lambd = lambd
        self.K = None

    def fit(self, x, y):
        n = len(x)
        if self.K is None:
            self.K = cp.psd_wrap(self.kernel.matrix(x, x))

        alpha = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(alpha, self.K) - 2 * y @ alpha),
                          [np.diag(y) @ alpha <= np.ones(n) / (2 * n * self.lambd),
                           np.diag(y) @ alpha >= np.zeros(n)])
        prob.solve()
        alpha = alpha.value
        return alpha

    def compute_K(self, x):
        self.K = cp.psd_wrap(self.kernel.matrix(x, x))


class MultiClassClassifier:
    def __init__(self, num_classes, model, method='one_versus_the_rest'):
        self.num_classes = num_classes
        self.method = method

        self.model = model
        self.kernel = self.model.kernel

        #if method == 'one_versus_the_rest':
        #  self.models = [model(**model_params) for i in range(num_classes)]
        #elif method == 'pairwise':
        #  self.models = [[model(**model_params) for j in range(i+1, num_classes)] for i in range(num_classes)]

    def fit(self, x, y, n_jobs=5):
        # y \subset {0, 1, ..., num_classes - 1}
        self.x = x
        self.y = y
        self.alpha = []
        self.model.compute_K(x)

        def fit_one_model(i):
            new_y = np.where(y == i, np.ones_like(y), -np.ones_like(y))
            alpha = self.model.fit(x, new_y)
            return alpha

        if self.method == 'one_versus_the_rest':
            if n_jobs == 1:
                for i in tqdm(range(self.num_classes)):
                    self.alpha.append(fit_one_model(i))
            else:
                with tqdm_joblib(tqdm(desc="Training One-vs-rest", total=10)) as progress_bar:
                    self.alpha = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(fit_one_model)(i) for i in range(self.num_classes))

        elif self.method == 'pairwise':
            kernel_matrix = self.model.K
            for i in range(self.num_classes):
                self.alpha.append([])
                new_y = np.where(y == i, np.ones_like(y), -np.ones_like(y))
                for j in range(i + 1, self.num_classes):
                    mask = np.logical_or(y == i, y == j)
                    self.model.K = cp.psd_wrap(kernel_matrix[mask][:, mask])

                    alpha = self.model.fit(x[mask], new_y[mask])
                    self.alpha[i].append(alpha)
        else:
            print('Not implemented')

    def predict(self, x, return_scores=False):
        if self.method == 'one_versus_the_rest':
            scores = np.array(self.alpha) @ self.kernel.matrix(self.x, x)
            if return_scores:
                return scores
            return np.argmax(scores, 0)

        elif self.method == 'pairwise':
            wins_count = np.zeros((len(x), self.num_classes))
            kernel_matrix = self.kernel.matrix(self.x, x)

            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    mask = np.logical_or(self.y == i, self.y == j)
                    score = self.alpha[i][j - i - 1] @ kernel_matrix[mask]
                    wins_count[:, i] += (score > 0)
                    wins_count[:, j] += (score < 0)

            return np.argmax(wins_count, -1)
