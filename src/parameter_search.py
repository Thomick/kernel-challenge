# Perform hyperparameter search using grid search
from classifiers import SVM, MultiClassClassifier
from kernel import Kernel

from data_processing import get_train_val
from fisher_extractor import FisherExtractor

import numpy as np
import pandas as pd

train_data, train_labels, val_data, val_labels = get_train_val("data/Xtr.csv", "data/Ytr.csv")

extractor = FisherExtractor()
extractor.fit(train_data)

train_dataset = extractor.extract_from_dataset(train_data)
val_dataset = extractor.extract_from_dataset(val_data)


simple_kernel = Kernel('linear')
svm = SVM(simple_kernel)

# perform grid search
param_grid = {'lambd': [0.01, 0.1, 1, 10, 100]}
results = []

for l in param_grid['lambd']:
    print(f"Lambda: {l}")
    simple_kernel = Kernel('linear')
    cl = MultiClassClassifier(num_classes=10, model=SVM(simple_kernel, lambd=l))
    cl.fit(train_dataset[:], train_labels, n_jobs=5)

    y_val = cl.predict(val_dataset)

    print("   Validation accuracy:", np.mean(y_val == val_labels))

    results.append([l, np.mean(y_val == val_labels)])

results = pd.DataFrame(results, columns=['lambd', 'accuracy'])
print(results)
results.to_csv("parameters/results_linear.csv", index=False)