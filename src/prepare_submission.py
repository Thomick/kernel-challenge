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

train_data = load_data("data/Xtr.csv")
train_labels = pd.read_csv('data/Ytr.csv')['Prediction'].to_numpy()

test_data = load_data("data/Xte.csv")

np.random.seed(0)


print("Extracting features")
extractor = FisherExtractor()
extractor.fit(train_data)

train_dataset = extractor.extract_from_dataset(train_data)
test_dataset = extractor.extract_from_dataset(test_data)

C = 1
sigma = 'scale'
save_predictions = True

print("Linear SVM")
simple_kernel = Kernel('linear')
cl = MultiClassClassifier(num_classes=10, model=SVM(simple_kernel, lambd=C))
cl.fit(train_dataset, train_labels)
y_test = cl.predict(test_dataset)
print("Saving predictions")
pd.DataFrame({'Id': np.arange(1, len(y_test)+1), 'Prediction': y_test}).to_csv('submission/Yte_pred.csv', index=False)