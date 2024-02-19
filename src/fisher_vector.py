import numpy as np

# Reference: Jorge Sanchez, Florent Perronnin, Thomas Mensink, Jakob Verbeek. Image Classification with the Fisher Vector: Theory and Practice. [https://inria.hal.science/hal-00779493v3/document]

# TODO : Check if the code is correct


class FisherVector:
    """
    Class to compute Fisher Vectors from local features

    Assumes that the local features are extracted from a GMM with diagonal covariance matrices

    Attributes:
    - n_component: number of Gaussians in the GMM
    - dimension: dimension of the local features
    - mixture_weight: proportion of the Gaussian in the GMM (n_component)
    - mus: the means of the GMM (n_component x dimension)
    - sigmas: the variances of the GMM (n_component x dimension)

    """

    def __init__(self, n_component, dimension, mixture_weight, mus, sigmas):
        self.n_component = n_component
        self.dimension = dimension
        self.mixture_weight = mixture_weight
        self.mus = mus
        self.sigmas = sigmas

    def compute_fisher_vector(self, features):
        """
        Compute the Fisher Vector from a set of local features

        Arguments:
        - features: a 2D numpy array containing the local features (n_samples x dimension)

        Returns:
        - fv: a 1D numpy array containing the Fisher Vector
        """
        # Compute the posterior probabilities
        # (n_samples x n_component)
        post_prob = self.compute_posterior_probabilities(features)

        # Compute the Fisher Vector
        fv = self.compute_fisher_vector_from_posterior_probabilities(
            features, post_prob
        )

        return fv

    def compute_posterior_probabilities(self, features):
        """
        Compute the posterior probabilities of the local features

        Arguments:
        - features: a 2D numpy array containing the local features (n_samples x dimension)

        Returns:
        - post_prob: a 2D numpy array containing the posterior probabilities (n_samples x n_component)
        """
        # Compute the likelihood
        # (n_samples x n_component)
        likelihood = self.compute_likelihood(features)

        # Compute the posterior probabilities
        # (n_samples x n_component)
        post_prob = likelihood * self.mixture_weight
        post_prob /= post_prob.sum(axis=1, keepdims=True)

        return post_prob

    def compute_likelihood(self, features):
        """
        Compute the likelihood of the local features

        Arguments:
        - features: a 2D numpy array containing the local features (n_samples x dimension)

        Returns:
        - likelihood: a 2D numpy array containing the likelihood (n_samples x n_component)
        """
        # Compute the likelihood
        # (n_samples x n_component)
        likelihood = np.exp(
            -0.5
            * ((features[:, None, :] - self.mus[None, :, :]) ** 2)
            / self.sigmas[None, :, :]
        )
        likelihood /= np.sqrt(self.sigmas * 2 * np.pi)

        return likelihood

    def compute_fisher_vector_from_posterior_probabilities(self, features, post_prob):
        """
        Compute the Fisher Vector from the posterior probabilities

        Arguments:
        - features: a 2D numpy array containing the local features (n_samples x dimension)
        - post_prob: a 2D numpy array containing the posterior probabilities (n_samples x n_component)

        Returns:
        - fv: a 1D numpy array containing the Fisher Vector
        """
        # Compute the Fisher Vector
        # (n_component x dimension)
        grad = self.compute_gradient(features, post_prob)
        grad = grad.reshape(-1)

        # L2 normalize the Fisher Vector
        fv = grad / np.sqrt(grad.dot(grad))

        return fv

    def compute_gradient(self, features, post_prob):
        """
        Compute the gradient of the Fisher Vector

        Arguments:
        - features: a 2D numpy array containing the local features (n_samples x dimension)
        - post_prob: a 2D numpy array containing the posterior probabilities (n_samples x n_component)

        Returns:
        - grad: a 2D numpy array containing the gradient of the Fisher Vector (n_component x dimension)
        """
        # Compute the gradient
        # (n_component x dimension)
        grad = np.zeros((self.n_component, self.dimension))
        for i in range(self.n_component):
            diff = features - self.mus[i]
            grad[i] = (diff * post_prob[:, i, None]).sum(axis=0)
            grad[i] /= np.sqrt(self.mixture_weight[i])

        return grad


if __name__ == "__main__":
    n_component = 2
    dimension = 2
    mixture_weight = np.array([0.5, 0.5])
    mus = np.array([[0, 0], [1, 1]])
    sigmas = np.array([[1, 1], [1, 1]])

    fv = FisherVector(n_component, dimension, mixture_weight, mus, sigmas)

    features = np.array([[1, 0]])
    print(fv.compute_fisher_vector(features))
