import numpy as np


class MyGaussianMixture:
    """
    Gaussian Mixture Model with diagonal covariance matrix

    Attributes:
    - n_components: number of Gaussian components
    - means_: means of the Gaussian components (n_components x n_features)
    - covariances_: variances of the Gaussian components (n_components x n_features)
    - weights_: weights of the Gaussian components (n_components)
    """

    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.means_ = np.random.rand(n_components, n_features)
        self.covariances_ = np.random.rand(n_components, n_features) + 0.1
        self.weights_ = np.ones(n_components) / n_components

    def EM_step(self, X):
        """
        Perform one step of the EM algorithm

        Arguments:
        - X: 2D numpy array (n_samples x n_features)
        """
        N = X.shape[0]
        K = self.n_components

        # E-step
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = self.weights_[k] * self.compute_likelihood(
                X, self.means_[k], self.covariances_[k]
            )
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M-step
        Nk = gamma.sum(axis=0)
        self.weights_ = Nk / N
        # Update the means
        self.means_ = (gamma.T @ X) / Nk[:, np.newaxis]
        # Update the variances
        self.covariances_ = (gamma.T @ (X ** 2)) / Nk[:, np.newaxis] - self.means_ ** 2

    def compute_likelihood(self, X, mu, sigma):
        """
        Compute the likelihood of the data given the Gaussian distribution

        Arguments:
        - X: 2D numpy array (n_samples x n_features)
        - mu: 1D numpy array (n_features)
        - sigma: 1D numpy array (n_features)

        Returns:
        - likelihood: 1D numpy array (n_samples)
        """
        return np.exp(-0.5 * ((X - mu) ** 2 / sigma).sum(axis=1)) / np.sqrt(
            (2 * np.pi * sigma).prod()
        )

    def predict_proba(self, X):
        """
        Compute the probability of each sample in X to belong to each component

        Arguments:
        - X: 2D numpy array (n_samples x n_features)

        Returns:
        - proba: 2D numpy array (n_samples x n_components)
        """
        proba = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            proba[:, k] = self.weights_[k] * self.compute_likelihood(
                X, self.means_[k], self.covariances_[k]
            )
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def fit(self, X, n_iter=100):
        """
        Fit the Gaussian Mixture Model to the data using EM algorithm

        Arguments:
        - X: 2D numpy array (n_samples x n_features)
        """
        for _ in range(n_iter):
            self.EM_step(X)
