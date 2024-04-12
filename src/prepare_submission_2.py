import numpy as np
from classifiers import SVM, MultiClassClassifier
from kernel import Kernel
from fisher_extractor import FisherExtractor
from error_correcting_classifiers import BestClassifier
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

n_jobs = 2
train_pairwise = False
train_sift = False
train_hof = False

if train_pairwise:
    print("Train pairwise classifier")
    model = SVM(Kernel('cosine'), 1.)
    cl = MultiClassClassifier(num_classes=10, model=model, method='pairwise')
    test_counts = 0
    train_counts = 0
    for i in range(5):
        cl.fit(train_dataset[i*1000:(i + 1)*1000], train_labels[i*1000:(i + 1)*1000])
        test_counts = test_counts + cl.predict(test_dataset, return_scores=True)
        train_counts = train_counts + cl.predict(train_dataset, return_scores=True)

    y_test = np.argmax(test_counts, -1)
    y_train = np.argmax(train_counts, -1)
    print("Train accuracy:", np.mean(y_train == train_labels))
    np.save('../data/pair_y', y_test)

if train_sift:
    print("Train classifier with SIFT features")
    y_scores = []
    train_scores = []
    np.random.seed(0)
    random_perm = np.random.permutation(len(train_dataset_sift))
    for i in range(25):
        idx = random_perm[np.arange(i * 200, (i + 1) * 200)]
        model = SVM(Kernel('laplacian', 1.), lambd=10000)
        cl = BestClassifier(num_classes=10, model=model)
        cl.fit(train_dataset_sift[idx], train_labels[idx], n_jobs=n_jobs)
        scores = cl.predict(test_dataset_sift, return_scores=True)
        scores = np.exp(scores) / np.sum(np.exp(scores), 0)
        y_scores.append(scores)

        scores = cl.predict(train_dataset_sift, return_scores=True)
        scores = np.exp(scores) / np.sum(np.exp(scores), 0)
        train_scores.append(scores)

    y_scores = np.mean(y_scores, 0)
    y_test = np.argmax(y_scores, 0)

    train_scores = np.mean(train_scores, 0)
    y_train = np.argmax(train_scores, 0)

    print("Train accuracy:", np.mean(y_train == train_labels))
    np.save('../data/sift_y', y_test)

if train_hof:
    print("Train classifier with HOF features")
    y_scores = []
    train_scores = []
    np.random.seed(1)
    random_perm = np.random.permutation(len(train_dataset))
    for i in range(10):
        idx = random_perm[np.arange(i * 500, (i + 1) * 500)]
        model = SVM(Kernel('linear'), lambd=0.5)
        cl = BestClassifier(num_classes=10, model=model)
        cl.fit(train_dataset[idx], train_labels[idx], n_jobs=n_jobs)
        scores = cl.predict(test_dataset, return_scores=True)
        scores = np.exp(scores) / np.sum(np.exp(scores), 0)
        y_scores.append(scores)

        scores = cl.predict(train_dataset, return_scores=True)
        scores = np.exp(scores) / np.sum(np.exp(scores), 0)
        train_scores.append(scores)

    y_scores = np.mean(y_scores, 0)
    y_test = np.argmax(y_scores, 0)

    train_scores = np.mean(train_scores, 0)
    y_train = np.argmax(train_scores, 0)

    print("Train accuracy:", np.mean(y_train == train_labels))
    np.save('../data/hof_y', y_test)

hof_y = np.load('../data/hof_y.npy')
sift_y = np.load('../data/sift_y.npy')
pair_y = np.load('../data/pair_y.npy')

y_test = np.where(sift_y == pair_y, sift_y, hof_y)
print("Saving predictions")
pd.DataFrame({'Id': np.arange(1, len(y_test)+1), 'Prediction': y_test}).to_csv('Yte.csv', index=False)
