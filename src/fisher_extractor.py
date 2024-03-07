import numpy as np

from fisher_vector import compute_fisher_vector

from fast_hog_extractor import HOGExtractor

from skimage.feature import hog

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA



class FisherExtractor:
    """
    Extract Fisher Vectors from a dataset of images using the specified local feature extractor

    Arguments:
    - n_gaussian: the number of Gaussian components
    - cell_size: the size of the HOG cell
    - nbins: the number of bins in the HOG
    """

    def __init__(self, n_gaussian=128, cell_size=8, nbins=9):
        self.extractor = HOGExtractor(cell_size=cell_size, nbins=nbins)
        self.n_gaussian = n_gaussian
        self.gmm = None

    def fit(self, dataset):
        """
        Fit the Fisher Vector extractor to a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset
        """
        #local_features = self.extractor.extract_from_dataset(dataset)
        local_features = np.array([hog(dataset[i], pixels_per_cell=(8, 8), cells_per_block=(1, 1),channel_axis=-1,orientations=9) for i in range(len(dataset))])
        local_features = PCA(n_components=32).fit_transform(local_features)
        # print(local_features.shape)
        self.gmm = GaussianMixture(n_components=self.n_gaussian, covariance_type='diag')
        self.gmm.fit(local_features)
        

    def extract_from_dataset(self, dataset):
        """
        Extract Fisher Vectors from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the Fisher Vectors
        """

        #local_features = self.extractor.extract_from_dataset(dataset)
        local_features = np.array([hog(dataset[i], pixels_per_cell=(8, 8), cells_per_block=(1, 1),channel_axis=-1,orientations=9) for i in range(len(dataset))])
        local_features = PCA(n_components=32).fit_transform(local_features)
        # print(local_features.shape)
        if self.gmm is None:
            raise ValueError("The Fisher Vector extractor has not been fitted yet")
        
        final_features = []
        for i in range(len(local_features)):
            fv = compute_fisher_vector(local_features[i][None,:], self.gmm)
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
