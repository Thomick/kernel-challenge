import numpy as np


class HOGExtractor:
    """
    Class to extract HOG features from images

    Implementation with signed gradients and no block normalization

    Attributes:
    - cell_size: size of the cell in pixels
    - nbins: number of bins in the histogram
    """

    def __init__(self, cell_size=8, nbins=9):
        self.cell_size = cell_size
        self.nbins = nbins

    def _extract_single_channel(self, img):
        """
        Extract HOG features from an image

        Arguments:
        - img: a 2D numpy array representing the image

        Returns:
        - features: a 1D numpy array containing the HOG features
        """
        # Compute the gradient
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Compute the histogram
        features = []
        for i in range(0, img.shape[0] - self.cell_size + 1, self.cell_size):
            for j in range(0, img.shape[1] - self.cell_size + 1, self.cell_size):
                cell_magnitude = magnitude[
                    i : i + self.cell_size, j : j + self.cell_size
                ]
                cell_angle = angle[i : i + self.cell_size, j : j + self.cell_size]
                hist = np.zeros(self.nbins)
                for k in range(self.cell_size):
                    for l in range(self.cell_size):
                        bin1 = int(cell_angle[k, l] / (np.pi / self.nbins))
                        bin2 = (bin1 + 1) % self.nbins
                        weight = (cell_angle[k, l] % (np.pi / self.nbins)) / (
                            np.pi / self.nbins
                        )
                        hist[bin1] += cell_magnitude[k, l] * (1 - weight)
                        hist[bin2] += cell_magnitude[k, l] * weight
                features.extend(hist)
        return np.array(features)

    def extract_from_image(self, img):
        """
        Extract HOG features from an image

        Arguments:
        - img: a 3D numpy array representing the image

        Returns:
        - features: a 1D numpy array containing the HOG features
        """
        features = []
        for i in range(img.shape[2]):
            features.extend(self._extract_single_channel(img[:, :, i]))
        return np.array(features)

    def extract_from_dataset(self, dataset):
        """
        Extract HOG features from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the HOG features
        """
        features = []
        for img in dataset:
            features.append(self.extract_from_image(img))
        return np.array(features)


if __name__ == "__main__":
    # Example usage
    img = np.random.rand(32, 32)
    hog = HOGExtractor()
    features = hog._extract_single_channel(img)
    print(features)
    print(features.shape)

    dataset = np.random.rand(100, 32, 32, 3)

    features = hog.extract_from_dataset(dataset)
    print(features.shape)
