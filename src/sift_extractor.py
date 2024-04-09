import numpy as np
import cv2

class SIFTExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.kp = [cv2.KeyPoint(x, y, 8) for y in range(0, 32, 8) for x in range(0, 32, 8)]

    def extract_from_dataset(self, dataset):
        """
        Extract SIFT features from a dataset of images

        Arguments:
        - dataset: a 4D numpy array representing the dataset

        Returns:
        - features: a 2D numpy array containing the SIFT features
        """
        features = []
        for i in range(len(dataset)):
            image = dataset[i]
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, des = self.sift.compute(image, self.kp)
            des = des.flatten()
            des = des / np.linalg.norm(des)
            features.append(des)
        return np.array(features)
        

if __name__ == "__main__":
    hog = SIFTExtractor()
    dataset = np.random.rand(100, 32, 32, 3)

    features = hog.extract_from_dataset(dataset)
    print(features.shape)