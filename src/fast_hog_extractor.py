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

    def extract_from_dataset(self, dataset):
        """
        Extract HOG features from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the HOG features
        """
        # Compute the gradient
        h = dataset.shape[1]
        w = dataset.shape[2]
        c = dataset.shape[-1]

        gx = np.gradient(dataset, axis=2)
        gy = np.gradient(dataset, axis=1)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = (np.pi + np.arctan2(gy, gx)) % np.pi #np.arctan2(gy, gx)

        # Compute the histogram
        features = []
        for i in range(0, h - self.cell_size + 1, self.cell_size):
            for j in range(0, w - self.cell_size + 1, self.cell_size):
                cell_magnitude = magnitude[:, i : i + self.cell_size, 
                                           j : j + self.cell_size].reshape(len(dataset), -1, c)
                cell_angle = angle[:, i : i + self.cell_size,
                                   j : j + self.cell_size].reshape(len(dataset), -1, c)

                hist = np.zeros((len(dataset), c, self.nbins))

                bin1 = (cell_angle / (np.pi / self.nbins)).astype(int) % self.nbins

                bin2 = (bin1 + 1) % self.nbins
                weight = (cell_angle % (np.pi / self.nbins)) / (np.pi / self.nbins)

                for val in range(self.nbins):
                 # print(val, (bin1 == val).sum())
                  hist[:, :, val] = np.sum(cell_magnitude * (1 - weight) * (bin1 == val), 1)
                  hist[:, :, val] += np.sum(cell_magnitude * weight * (bin2 == val), 1)

                features.append(hist)
        
        features = np.concatenate(features, 2).reshape(len(dataset), -1)
        return features

