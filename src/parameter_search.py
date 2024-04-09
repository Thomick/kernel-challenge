# Perform hyperparameter search using grid search
from classifiers import SVM, MultiClassClassifier
from error_correcting_classifiers import BestClassifier
from kernel import Kernel

from data_processing import get_train_val
from fisher_extractor import FisherExtractor

import numpy as np
import pandas as pd

train_data, train_labels, val_data, val_labels = get_train_val("data/Xtr.csv", "data/Ytr.csv")

#extractor = FisherExtractor()
#extractor.fit(train_data)
#train_dataset = extractor.extract_from_dataset(train_data)
#val_dataset = extractor.extract_from_dataset(val_data)

# perform grid search
param_grid = {'sig': [3], 'lmbd': [1,10],'pca_dim':[10,20,30,40,50], 'n_gaussian':[10,20,30,40,50]}
results = []

for s in param_grid['lmbd']:
    print(f"lambda = {s}")
    extractor = FisherExtractor()
    extractor.fit(train_data)
    train_dataset = extractor.extract_from_dataset(train_data)
    val_dataset = extractor.extract_from_dataset(val_data)
    simple_kernel = Kernel('linear', sigma=s)
    svm = SVM(simple_kernel, lambd=1)
    cl = MultiClassClassifier(num_classes=10, model=svm)
    # cl = OneVsRestClassifier(SVC(kernel='rbf', C=C, gamma=sigma), n_jobs=12, verbose=True)
    cl.fit(train_dataset, train_labels, n_jobs=5)

    y_val = cl.predict(val_dataset)

    print("   Validation accuracy:", np.mean(y_val == val_labels))

    results.append([s, np.mean(y_val == val_labels)])

results = pd.DataFrame(results, columns=['lambda', 'accuracy'])
print(results)
results.to_csv("parameters/results_rbf_error.csv", index=False)