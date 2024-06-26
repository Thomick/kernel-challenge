import numpy as np

from fisher_vector import compute_fisher_vector

from fast_hog_extractor import HOGExtractor

from sift_extractor import SIFTExtractor

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from gmm import MyGaussianMixture


class FisherExtractor:
    """
    Extract Fisher Vectors from a dataset of images using the specified local feature extractor

    Arguments:
    - n_gaussian: the number of Gaussian components
    - cell_size: the size of the HOG cell
    - nbins: the number of bins in the HOG
    """

    def __init__(self, n_gaussian=10, cell_size=8, nbins=9, pca_dim=64, use_sift=False):
        if use_sift:
            self.extractor = SIFTExtractor()
        else:
            self.extractor = HOGExtractor(cell_size=cell_size, nbins=nbins)
        self.n_gaussian = n_gaussian
        self.gmm = None
        self.pca = PCA(n_components=pca_dim)

    def fit(self, dataset):
        """
        Fit the Fisher Vector extractor to a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset
        """
        # local_features = self.extractor.extract_from_dataset(dataset)

        local_features = self.extractor.extract_from_dataset(dataset)
        # print(local_features.shape)
        local_features = self.pca.fit_transform(local_features)
        # print(local_features.shape)
        self.gmm = MyGaussianMixture(n_components=self.n_gaussian, n_features=local_features.shape[1])
        self.gmm.fit(local_features)

    def extract_from_dataset(self, dataset):
        """
        Extract Fisher Vectors from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the Fisher Vectors
        """

        # local_features = self.extractor.extract_from_dataset(dataset)
        local_features = self.extractor.extract_from_dataset(dataset)
        local_features = self.pca.transform(local_features)
        # print(local_features.shape)
        if self.gmm is None:
            raise ValueError("The Fisher Vector extractor has not been fitted yet")

        final_features = []
        for i in range(len(local_features)):
            fv = compute_fisher_vector(local_features[i][None, :], self.gmm)
            final_features.append(fv)
        return np.array(final_features)


if __name__ == "__main__":
    # Generate random dataset
    dataset = np.random.rand(200, 32, 32, 3)

    # Create the FisherExtractor
    extractor = FisherExtractor()

    # Fit the FisherExtractor
    extractor.fit(dataset)

    # Extract Fisher Vectors from the dataset
    features = extractor.extract_from_dataset(dataset)

    # Print the shape of the features
    print(features.shape)
