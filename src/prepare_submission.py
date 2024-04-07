import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classifiers import SVM, MultiClassClassifier
from error_correcting_classifiers import HammingClassifier, GolayClassifier, CustomClassifier, BestClassifier
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

try_linear = True
train_one_vs_rest = True
train_pairwise = True
train_hamming = True
train_golay = True
train_custom = True
train_best = True
save = True
save_path = 'saved_models/full/'
load = False

C = 1
n_jobs = 3
sigma = np.sqrt(train_dataset.shape[1]*train_dataset.var()*2)

y_test_votes = np.zeros((test_dataset.shape[0], 10))

if try_linear:
    print("Linear SVM")
    simple_kernel = Kernel('linear')
    model = SVM(simple_kernel, lambd=C)

    if train_one_vs_rest:
        cl = MultiClassClassifier(num_classes=10, model=model)
        if load:
            cl.load('saved_models/one_versus_the_rest_linear_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        #cl.load('saved_models/one_versus_the_rest_linear_alpha.npy', train_dataset[:], train_labels)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)

    if train_pairwise:
        cl = MultiClassClassifier(num_classes=10, model=model, method='pairwise')
        if load:
            cl.load('saved_models/pairwise_linear_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)


    if train_hamming:
        class_order = np.array([1, 5, 6, 3, 7, 2, 9, 4, 0, 8])
        cl = HammingClassifier(num_classes=10, model=model, class_order=class_order)
        if load:
            cl.load('saved_models/hamming_linear_1_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        #cl.load('saved_models/hamming_linear_1_alpha.npy', train_dataset[:], train_labels)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)


    if train_golay:
        # class_order = np.random.permutation(np.arange(10))
        class_order = np.array([0, 2, 6, 9, 5, 4, 7, 3, 1, 8]) #[5, 3, 8, 6, 9, 1, 0, 4, 2, 7]
        cl = GolayClassifier(num_classes=10, model=model, class_order=class_order)
        if load:
            cl.load('saved_models/golay_linear_3_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)

    if train_custom:
        cl = CustomClassifier(num_classes=10, model=model)
        if load:
            cl.load('saved_models/custom_linear_0_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)

    if train_best:
        cl = BestClassifier(num_classes=10, model=model)
        if load:
            cl.load('saved_models/best_linear_0_alpha.npy', train_dataset[:], train_labels)
        else:
            cl.fit(train_dataset[:], train_labels, n_jobs=n_jobs)
        y_test = cl.predict(test_dataset)
        y_test_votes += np.eye(10)[y_test]
        if save:
            cl.save(save_path)

y_test = np.argmax(y_test_votes, axis=1)

print("Saving predictions")
pd.DataFrame({'Id': np.arange(1, len(y_test)+1), 'Prediction': y_test}).to_csv('submission/Yte_pred.csv', index=False)