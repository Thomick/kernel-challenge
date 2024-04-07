import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classifiers import SVM, MultiClassClassifier
from error_correcting_classifiers import HammingClassifier, GolayClassifier, CustomClassifier, BestClassifier
from kernel import Kernel
from fisher_extractor import FisherExtractor

from data_processing import load_data
import pandas as pd
#from skimage.feature import hog, fisher_vector, learn_gmm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

data = load_data("../data/Xtr.csv")
labels = pd.read_csv('../data/Ytr.csv')['Prediction'].to_numpy()

test_data = load_data("../data/Xte.csv")

np.random.seed(0)
val_idx = np.random.choice(np.arange(len(labels)), len(labels)//5, replace=False)
mask = np.ones(len(labels), dtype=bool)
mask[val_idx] = 0

val_data = data[val_idx]
val_labels = labels[val_idx]

train_data = data[mask]
train_labels = labels[mask]

print("Extracting features")
extractor = FisherExtractor()
extractor.fit(train_data)


train_dataset = extractor.extract_from_dataset(train_data)
# np.save('../data/train_features.npy', train_dataset)
# train_dataset = np.load('../data/train_features.npy')
val_dataset = extractor.extract_from_dataset(val_data)
# np.save('../data/val_features.npy', val_dataset)
# val_dataset = np.load('../data/val_features.npy')
test_dataset = extractor.extract_from_dataset(test_data)
# np.save('../data/test_features.npy', test_dataset)
# test_dataset = np.load('../data/test_features.npy')

# small_train_len = len(val_dataset)
# train_dataset = train_dataset[:small_train_len]
# train_labels = train_labels[:small_train_len]

print(train_dataset.shape, val_dataset.shape, test_dataset.shape)

try_linear = True
train_one_vs_rest = True
train_pairwise = False
train_hamming = True
train_golay = True
train_custom = True
train_best = True

try_rbf = False
C = 1
n_jobs = 4
sigma = np.sqrt(train_dataset.shape[1]*train_dataset.var()*2)

if try_linear:
    print("Linear SVM")
    simple_kernel = Kernel('linear')
    model = SVM(simple_kernel, lambd=C)

    if train_one_vs_rest:
        cl = MultiClassClassifier(num_classes=10, model=model)
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("One vs the rest. Validation accuracy:", np.mean(y_val == val_labels))

    if train_pairwise:
        cl = MultiClassClassifier(num_classes=10, model=model, method='pairwise')
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("Pairwise. Validation accuracy:", np.mean(y_val == val_labels))

    if train_hamming:
        # class_order = np.random.permutation(np.arange(10))
        cl = HammingClassifier(num_classes=10, model=model)
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("Hamming. Validation accuracy:", np.mean(y_val == val_labels))

    if train_golay:
        # class_order = np.random.permutation(np.arange(10))
        class_order = np.array([0, 2, 6, 9, 5, 4, 7, 3, 1, 8]) #[5, 3, 8, 6, 9, 1, 0, 4, 2, 7]
        cl = GolayClassifier(num_classes=10, model=model, class_order=class_order)
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("Golay. Validation accuracy:", np.mean(y_val == val_labels))

    if train_custom:
        cl = CustomClassifier(num_classes=10, model=model)
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("Custom. Validation accuracy:", np.mean(y_val == val_labels))

    if train_best:
        cl = BestClassifier(num_classes=10, model=model)
        cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_val = cl.predict(val_dataset)
        print("Best. Validation accuracy:", np.mean(y_val == val_labels))

if try_rbf:
    print("RBF SVM")
    simple_kernel = Kernel('rbf', sigma=sigma)
    svm = SVM(simple_kernel, lambd=C)
    cl = MultiClassClassifier(num_classes=10, model=svm)
    # cl = OneVsRestClassifier(SVC(kernel='rbf', C=C, gamma=sigma), n_jobs=12, verbose=True)
    cl.fit(train_dataset, train_labels, n_jobs=5)

    y_val = cl.predict(val_dataset)

    print("Validation accuracy:", np.mean(y_val == val_labels))

    y_test = cl.predict(val_dataset)
    # print(val_dataset[:100])
    # print(y_test[:100])
    # print(simple_kernel.matrix(train_dataset[:10], train_dataset[:10]))
