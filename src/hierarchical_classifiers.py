import numpy as np
from classifiers import SVM, MultiClassClassifier, LogisticRegression
from error_correcting_classifiers import ErrorCorrectingClassifier
from kernel import Kernel


class TwoGroupsClassifier:
    def __init__(self, num_classes, model, group1, group2):
        self.num_classes = num_classes
        self.model = model
        self.kernel = self.model.kernel
        self.group1 = group1
        self.group2 = group2

    def fit(self, x, y):
        mask = np.logical_or(np.isin(y, self.group1), np.isin(y, self.group2))
        self.x = x[mask]
        y = y[mask]
        y = np.where(np.isin(y, self.group1), np.ones_like(y), -np.ones_like(y))
        self.model.compute_K(self.x)

        self.alpha = [self.model.fit(self.x, y)]

    def predict(self, x, return_scores=False):
        scores = np.array(self.alpha) @ self.kernel.matrix(self.x, x)
        if return_scores:
            return scores
        return np.argmax(scores, 0)


def hierarchical_classification(is_vehicle, animal_group, vehicle_group, pred_19, pred_08, pred_26, pred_35, pred_47):
    mask_08 = np.logical_and(is_vehicle, vehicle_group)
    mask_19 = np.logical_and(is_vehicle, np.logical_not(vehicle_group))
    mask_26 = np.logical_and(np.logical_not(is_vehicle), animal_group == 0)
    mask_35 = np.logical_and(np.logical_not(is_vehicle), animal_group == 1)
    mask_47 = np.logical_and(np.logical_not(is_vehicle), animal_group == 2)

    res = np.zeros(len(is_vehicle))
    res[np.logical_and(mask_08, np.logical_not(pred_08))] = 8
    res[np.logical_and(mask_19, pred_19)] = 1
    res[np.logical_and(mask_19, np.logical_not(pred_19))] = 9
    res[np.logical_and(mask_26, pred_26)] = 2
    res[np.logical_and(mask_26, np.logical_not(pred_26))] = 6
    res[np.logical_and(mask_35, pred_35)] = 3
    res[np.logical_and(mask_35, np.logical_not(35))] = 5
    res[np.logical_and(mask_47, pred_47)] = 4
    res[np.logical_and(mask_47, np.logical_not(47))] = 7
    return res


def train_vehicle_vs_animal_classifier_old(train_dataset, train_labels, test_dataset, test_labels=None):
    n_jobs = 2
    model1 = LogisticRegression(Kernel('linear'), lambd=1)
    model2 = SVM(Kernel('linear'), lambd=0.5)
    model3 = SVM(Kernel('laplacian', sigma=5), lambd=10)
    cl1 = ErrorCorrectingClassifier(num_classes=2, model=model1)
    cl2 = ErrorCorrectingClassifier(num_classes=2, model=model2)
    cl3 = ErrorCorrectingClassifier(num_classes=2, model=model3)
    group = np.array([0, 1, 8, 9])

    cl_list = [cl1, cl2, cl3]
    for cl in cl_list:
        cl.groups = [group]

    y_scores = [[], [], []]
    for i in range(20):
        idx = np.arange(i * 200, (i + 1) * 200)

        cl1.fit(train_dataset[idx], train_labels[idx], n_jobs=n_jobs)
        scores = cl1.predict(test_dataset, return_scores=True)
        y_scores[0].append(scores[0])

    for i in range(8):
        # print(sigma)
        idx = np.arange(i * 500, (i + 1) * 500)

        for j, cl in enumerate(cl_list):
            if j == 0:
                continue
            cl.fit(train_dataset[idx], train_labels[idx], n_jobs=n_jobs)
            scores = cl.predict(test_dataset, return_scores=True)
            y_scores[j].append(scores[0])

    scores = [np.mean(y_scores[i], 0) >= 0 for i in range(len(cl_list))]
    scores = np.where(scores[1] == scores[2], scores[1], scores[0])
    if test_labels is not None:
        test_acc = np.mean(scores == (np.isin(test_labels, group)))
        print('Validation accuracy', test_acc)

    return scores


def train_vehicle_vs_animal_classifier(train_dataset, train_labels, test_dataset, test_labels=None):
    n_jobs = 2
    model2 = SVM(Kernel('linear'), lambd=0.5)
    cl = ErrorCorrectingClassifier(num_classes=2, model=model2)
    group = np.array([0, 1, 8, 9])

    cl.groups = [group]

    y_scores = []
    train_scores = []

    for i in range(8):
        # print(sigma)
        idx = np.arange(i * 500, (i + 1) * 500)
        cl.fit(train_dataset[idx], train_labels[idx], n_jobs=n_jobs)
        scores = cl.predict(test_dataset, return_scores=True)
        y_scores.append(scores[0])
        scores = cl.predict(train_dataset, return_scores=True)
        train_scores.append(scores[0])

    scores = np.mean(y_scores, 0) >= 0
    train_scores = np.mean(train_scores, 0) >= 0
    if test_labels is not None:
        test_acc = np.mean(scores == (np.isin(test_labels, group)))
        print('Validation accuracy', test_acc)

    return scores, train_scores


def train_pairwise_classifier(subset, group, train_dataset, train_labels, test_dataset, test_labels=None,
                              batch_size=500, n_iter=10, model=SVM(Kernel('cosine'), lambd=0.1)):
    y_scores = []
    train_mask = np.isin(train_labels, subset)
    train_dataset = train_dataset[train_mask]
    train_labels = train_labels[train_mask]
    print(len(train_dataset))
    if batch_size > len(train_dataset) * 0.7:
        batch_size = len(train_dataset)
        n_iter = 1

    if test_labels is not None:
        test_mask = np.isin(test_labels, subset)
        test_labels = test_labels[test_mask]

    cl = ErrorCorrectingClassifier(num_classes=2, model=model)
    cl.groups = [group]
    np.random.seed(0)
    for i in range(n_iter):
        idx = np.random.choice(np.arange(len(train_dataset)), batch_size, replace=False)

        cl.fit(train_dataset[idx], train_labels[idx])
        scores = cl.predict(test_dataset, return_scores=True)
        y_scores.append(scores[0])

    scores = np.mean(y_scores, 0) >= 0.
    if test_labels is not None:
        acc = np.mean(scores[test_mask] == (np.isin(test_labels, group)))
        print(acc)
    return scores


def train_animals_classifier(train_dataset, train_labels, test_dataset, test_labels=None):
    subset = [2, 3, 4, 5, 6, 7]
    label_dict = {2: 0, 3: 1, 4: 2, 5: 1, 6: 0, 7: 2}
    train_mask = np.isin(train_labels, subset)
    train_subset = train_dataset[train_mask]
    train_labels = train_labels[train_mask]
    train_labels = np.array([label_dict[l] for l in train_labels])

    if test_labels is not None:
        test_mask = np.isin(test_labels, subset)
        test_labels = test_labels[test_mask]
        test_labels = np.array([label_dict[l] for l in test_labels])

    n_jobs = 2
    model = SVM(Kernel('linear'), lambd=0.5)
    cl = MultiClassClassifier(num_classes=3, model=model)
    cl.fit(train_subset[:1000], train_labels[:1000], n_jobs=n_jobs)
    y_test = cl.predict(test_dataset)
    y_train = cl.predict(train_dataset)
    if test_labels is not None:
        print("Validation accuracy:", np.mean(y_test[test_mask] == test_labels))

    return y_test, y_train


def train_group(name, train_dataset, train_labels, test_dataset, test_labels=None,
                model=SVM(Kernel('rbf', 0.1), lambd=100), n=12, bs=200, seed=0):

    if name == 'animals':
        subset = [2, 3, 4, 5, 6, 7]
        label_dict = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        num_classes = 6
    else:
        subset = [0, 1, 8, 9]
        label_dict = {0:0, 1:1, 8:2, 9:3}
        num_classes = 4

    train_mask = np.isin(train_labels, subset)
    train_dataset = train_dataset[train_mask]
    train_labels = train_labels[train_mask]
    train_labels = np.array([label_dict[l] for l in train_labels])

    print(len(train_dataset))
    if len(train_labels) < n * bs:
        bs = len(train_labels) // n

    val_dataset = test_dataset
    if test_labels is not None:
        val_mask = np.isin(test_labels, subset)
        val_labels = test_labels[val_mask]
        val_labels = np.array([label_dict[l] for l in val_labels])

    n_jobs = 2
    y_scores = []
    np.random.seed(seed)
    all_idx = np.random.permutation(len(train_dataset))
    for i in range(n):
        idx = np.arange(i * bs, (i + 1) * bs)
        idx = all_idx[idx]
        cl = MultiClassClassifier(num_classes=num_classes, model=model)
        cl.fit(train_dataset[idx], train_labels[idx], n_jobs=n_jobs)
        scores = cl.predict(val_dataset, return_scores=True)
        scores = np.exp(scores) / np.sum(np.exp(scores), 0)
        y_scores.append(scores)

    y_scores = np.mean(y_scores, 0)
    y_val = np.argmax(y_scores, 0)
    if test_labels is not None:
        acc = np.mean(y_val[val_mask] == val_labels)
        print(acc)

    return y_scores, y_val
