import numpy as np

from fisher_vector import FisherVector
from hog_extractor import HOGExtractor

# TODO: Finish the implementation of the FisherExtractor class
# TODO: Test the FisherExtractor class


class FisherExtractor:
    """
    Extract Fisher Vectors from a dataset of images using the specified local feature extractor

    Arguments:
    - n_component: the number of Gaussian components
    - cell_size: the size of the HOG cell
    - nbins: the number of bins in the HOG
    """

    def __init__(self, n_component=128, cell_size=8, nbins=9):
        self.extractor = HOGExtractor(cell_size=cell_size, nbins=nbins)
        self.fv = None
        self.n_component = n_component

    def fit(self, dataset):
        """
        Fit the Fisher Vector extractor to a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset
        """
        features = self.extractor.extract_from_dataset(dataset)
        gmm = self.fit_gmm(features)
        self.fv = FisherVector(
            n_component=gmm.n_components,
            dimension=gmm.means_.shape[1],
            mixture_weight=gmm.weights_,
            mus=gmm.means_,
            sigmas=gmm.covariances_,
        )

    def extract_from_dataset(self, dataset):
        """
        Extract Fisher Vectors from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the Fisher Vectors
        """
        features = []
        for img in dataset:
            features.append(self.extract_from_image(img))
        return np.array(features)

    def extract_from_image(self, img):
        """
        Extract Fisher Vectors from an image

        Arguments:
        - img: a 3D numpy array representing the image

        Returns:
        - features: a 1D numpy array containing the Fisher Vectors
        """
        features = self.extractor.extract_from_image(img)
        if self.fv is None:
            raise ValueError("The Fisher Vector extractor has not been fitted yet")
        features = self.fv.compute_fisher_vector(features[np.newaxis, :])
        return features
