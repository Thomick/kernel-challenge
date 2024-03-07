import numpy as np

# Reference: Jorge Sanchez, Florent Perronnin, Thomas Mensink, Jakob Verbeek. Image Classification with the Fisher Vector: Theory and Practice. [https://inria.hal.science/hal-00779493v3/document]

# TODO : Check if the code is correct


def compute_fisher_vector(data, gmm):
    """
    Compute the Fisher Vector of a dataset using the specified Gaussian Mixture Model

    Arguments:
    - data: a 2D numpy array representing the dataset
    - gmm: a GaussianMixture object
    """
    mixture_weight = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    n_samples, n_features = data.shape
    n_components = len(mixture_weight)
    # Compute the posterior probabilities
    posterior = gmm.predict_proba(data)
    # Compute the sufficient statistics
    S = posterior.sum(axis=0, keepdims=True).T
    x = posterior.T @ data/ n_samples
    x2 = posterior.T @ (np.power(data, 2)) / n_samples
    # Compute GMM gradients
    d_pi = S.squeeze() - mixture_weight
    d_mu = x - S*means
    d_sigma = -x2 - S*np.power(means, 2) + S*covariances + 2*x*means
    # Normalize the gradients
    sqrt_pi = np.sqrt(mixture_weight)
    d_pi /= sqrt_pi
    d_mu /= sqrt_pi[:, np.newaxis] * np.sqrt(covariances)
    d_sigma /=  sqrt_pi[:, np.newaxis] * covariances * np.sqrt(2)

    # Concatenate the gradients
    fv = np.hstack((d_pi.reshape(-1), d_mu.reshape(-1), d_sigma.reshape(-1)))

    return fv 
    # TODO: Check the improved version of the Fisher Vector (power normalization, L2 normalization)


if __name__ == "__main__":
    from sklearn.mixture import GaussianMixture
    data = np.array([[1, 2, 3], [4, 5, 7], [1, 2, 4], [5, 6, 77], [2, 3, 4]])
    gmm = GaussianMixture(n_components=3, covariance_type='diag')
    gmm.fit(data)
    fv = compute_fisher_vector(data[0][None,:], gmm)
    print(fv)

    # Expected output:
    from skimage.feature import fisher_vector
    fv = fisher_vector(data[0][None,:], gmm)
    print(fv)

    print("Are the results equal?", np.allclose(fv, compute_fisher_vector(data[0][None,:], gmm), atol=1e-5))