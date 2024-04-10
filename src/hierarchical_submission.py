import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classifiers import SVM, MultiClassClassifier
from kernel import Kernel
from fisher_extractor import FisherExtractor
from error_correcting_classifiers import BestClassifier
from hierarchical_classifiers import train_group, train_vehicle_vs_animal_classifier
from data_processing import load_data
import pandas as pd

from sift_extractor import SIFTExtractor

train_data = load_data("../data/Xtr.csv")
train_labels = pd.read_csv('../data/Ytr.csv')['Prediction'].to_numpy()

test_data = load_data("../data/Xte.csv")

print("Extracting features")
extractor = FisherExtractor()
extractor.fit(train_data)

train_dataset = extractor.extract_from_dataset(train_data)
test_dataset = extractor.extract_from_dataset(test_data)

print("Extracting SIFT features")
extractor = SIFTExtractor()
train_dataset_sift = extractor.extract_from_dataset(train_data)
test_dataset_sift = extractor.extract_from_dataset(test_data)

is_vehicle, train_scores = train_vehicle_vs_animal_classifier(train_dataset, train_labels, test_dataset)

correct_mask = (np.isin(train_labels, [0, 1, 8, 9]) == train_scores)
train_dataset = train_dataset[correct_mask]
train_dataset_sift = train_dataset_sift[correct_mask]
train_labels = train_labels[correct_mask]
print(train_dataset.shape, test_dataset.shape)

a_scores_1, a_y_1 = train_group('animals', train_dataset_sift, train_labels, test_dataset_sift,
                                model=SVM(Kernel('rbf', 0.1), lambd=100), n=10, bs=200, seed=0)
a_scores_3, a_y_3 = train_group('animals', train_dataset_sift, train_labels, test_dataset_sift,
                                model=SVM(Kernel('cosine'), lambd=0.0001), n=10, bs=200, seed=2)
a_scores_2, a_y_2 = train_group('animals', train_dataset, train_labels, test_dataset,
                                model=SVM(Kernel('linear'), lambd=1), n=10, bs=200, seed=1)

v_scores_1, v_y_1 = train_group('v', train_dataset_sift, train_labels, test_dataset_sift,
                                model=SVM(Kernel('rbf', 0.1), lambd=100), n=6, bs=250, seed=0)
v_scores_3, v_y_3 = train_group('v', train_dataset_sift, train_labels, test_dataset_sift,
                                model=SVM(Kernel('cosine'), lambd=0.0001), n=6, bs=250, seed=2)
v_scores_2, v_y_2 = train_group('v', train_dataset, train_labels, test_dataset,
                                model=SVM(Kernel('linear'), lambd=1), n=6, bs=250, seed=1)

# a_y = np.argmax(a_scores_1 + a_scores_2 + a_scores_3, 0)
# v_y = np.argmax(v_scores_1 + v_scores_2 + v_scores_3, 0)

a_y = np.where(a_y_1 == a_y_3, a_y_1, a_y_2)
v_y = np.where(v_y_1 == v_y_3, v_y_1, v_y_2)

inv_dict = [0, 1, 8, 9]
v_y = [inv_dict[l] for l in v_y]
a_y = a_y + 2

y_test = np.where(is_vehicle, v_y, a_y)
print("Saving predictions")
pd.DataFrame({'Id': np.arange(1, len(y_test)+1), 'Prediction': y_test}).to_csv('Yte_pred.csv', index=False)
