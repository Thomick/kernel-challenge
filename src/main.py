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


try_linear = True
try_rbf = False
C = 1
sigma = np.sqrt(train_dataset.shape[1]*train_dataset.var()*2)

if try_linear:
    print("Linear SVM")
    simple_kernel = Kernel('linear')
    cl = MultiClassClassifier(num_classes=10, model=SVM(simple_kernel, lambd=C))
    cl.fit(train_dataset[:], train_labels, n_jobs=5)

    y_val = cl.predict(val_dataset)

    print("Validation accuracy:", np.mean(y_val == val_labels))

if try_rbf:
    print("RBF SVM")
    simple_kernel = Kernel('rbf', sigma=sigma)
    svm = SVM(simple_kernel, lambd=C)
    cl = MultiClassClassifier(num_classes=10, model=svm)
    #cl = OneVsRestClassifier(SVC(kernel='rbf', C=C, gamma=sigma), n_jobs=12, verbose=True)
    cl.fit(train_dataset, train_labels, n_jobs=5)

    y_val = cl.predict(val_dataset)

    print("Validation accuracy:", np.mean(y_val == val_labels))

    y_test = cl.predict(val_dataset)
    # print(val_dataset[:100])
    # print(y_test[:100])
    # print(simple_kernel.matrix(train_dataset[:10], train_dataset[:10]))