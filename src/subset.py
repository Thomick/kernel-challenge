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

from tqdm import tqdm

data = load_data("data/Xtr.csv")
labels = pd.read_csv('data/Ytr.csv')['Prediction'].to_numpy()

test_data = load_data("data/Xte.csv")

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

#small_train_len = len(val_dataset)
#train_dataset = train_dataset[:small_train_len]
#train_labels = train_labels[:small_train_len]

print(train_dataset.shape, val_dataset.shape, test_dataset.shape)

n_cut = 100
size_cut = train_dataset.shape[0] // 4
C = 1
n_jobs = 10

y_val_votes = np.zeros((val_dataset.shape[0], 10))
simple_kernel = Kernel('rbf', sigma=3)
model = SVM(simple_kernel, lambd=C)

for i in tqdm(range(n_cut)):
    cl = BestClassifier(num_classes=10, model=model)
    train_idx = np.random.choice(train_dataset.shape[0], size_cut, replace=True)
    cl.fit(train_dataset[train_idx], train_labels[train_idx], n_jobs=n_jobs)
    y_val = cl.predict(val_dataset)
    y_val_votes += np.eye(10)[y_val]
    print("Best. Validation accuracy:", np.mean(y_val == val_labels))



print("Ensemble. Validation accuracy:", np.mean(np.argmax(y_val_votes, axis=1) == val_labels))