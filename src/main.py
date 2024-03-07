import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classifiers import SVM, MultiClassClassifier
from kernel import Kernel
from fisher_extractor import FisherExtractor

from data_processing import load_data
import pandas as pd
from skimage.feature import hog, fisher_vector, learn_gmm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

data = load_data("data/Xtr.csv")
labels = pd.read_csv('data/Ytr.csv')['Prediction'].to_numpy()

test_data = load_data("data/Xte.csv")

np.random.seed(0)
val_idx = np.random.choice(np.arange(len(data)), len(data)//5, replace=False)
mask = np.ones(len(data), dtype=bool)
mask[val_idx] = 0

val_data = data[val_idx]
val_labels = labels[val_idx]

train_data = data[mask]
train_labels = labels[mask]

print("Extracting features")
extractor = FisherExtractor()
extractor.fit(train_data)

train_dataset = extractor.extract_from_dataset(train_data)
val_dataset = extractor.extract_from_dataset(val_data)
test_dataset = extractor.extract_from_dataset(test_data)

print(train_dataset.shape, val_dataset.shape, test_dataset.shape)


try_linear = False
try_rbf = True
C = 10
sigma = 'scale'

if try_linear:
    print("Linear SVM")
    simple_kernel = Kernel('linear')
    cl = OneVsRestClassifier(SVC(kernel='linear',C=C),verbose=True, n_jobs=10)
    cl.fit(train_dataset, train_labels)

    y_val = cl.predict(val_dataset)

    print("Validation accuracy:", np.mean(y_val == val_labels))

if try_rbf:
    print("RBF SVM")
    simple_kernel = Kernel('rbf', sigma=3.4)
    #svm = SVM(simple_kernel)
    #cl = MultiClassClassifier(num_classes=10, model=svm)
    print(train_dataset.shape)
    cl = OneVsRestClassifier(SVC(kernel='rbf',C=C,gamma=sigma),verbose=True, n_jobs=10)
    cl.fit(train_dataset, train_labels)

    y_val = cl.predict(val_dataset)

    print("Validation accuracy:", np.mean(y_val == val_labels))