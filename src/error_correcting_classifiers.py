import numpy as np
import cvxpy as cp
from scipy.spatial import distance
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import *
import os

CLASS_NAMES = np.array(["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])


class ErrorCorrectingClassifier:
    def __init__(self, num_classes, model, errors=0):
        self.num_classes = num_classes

        self.model = model
        self.kernel = self.model.kernel
        self.errors = errors
        self.method = 'generic'

    def set_groups(self, code, class_order):
        self.groups = [class_order[assignment] for assignment in code]
        # for i, group in enumerate(self.groups):
        #    print("Group {}: {}".format(i, CLASS_NAMES[group]))

        self.groups = [np.array([i]) for i in range(self.num_classes)] + self.groups
        self.code_words = np.zeros((self.num_classes, len(self.groups)))
        for i, group in enumerate(self.groups):
            self.code_words[:, i][group] = 1

    def fit(self, x, y, n_jobs=5):
        # y \subset {0, 1, ..., num_classes - 1}
        self.x = x
        self.y = y
        self.alpha = []
        self.model.compute_K(x)

        def fit_one_model(group):
            new_y = np.where(np.isin(y, group), np.ones_like(y), -np.ones_like(y))
            alpha = self.model.fit(x, new_y)
            return alpha

        if n_jobs == 1:
            for group in tqdm(self.groups):
                self.alpha.append(fit_one_model(group))
        else:
            with tqdm_joblib(tqdm(desc="Training error correcting classifier", total=len(self.groups))) as progress_bar:
                self.alpha = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(fit_one_model)(group) for group in self.groups)

    def predict(self, x):
        scores = np.array(self.alpha) @ self.kernel.matrix(self.x, x)
        old_res = np.argmax(scores[:self.num_classes], 0)

        codes = (scores.T > 0) * 1.
        distance2codeword = distance.cdist(codes, self.code_words, metric='hamming') * self.code_words.shape[1]
        new_res = np.argmin(distance2codeword, 1)

        # print(np.sum(np.min(distance2codeword, 1) <= self.errors))
        res = np.where(np.min(distance2codeword, 1) <= self.errors, new_res, old_res)
        return res
    
    def save(self, path):
        name = self.method + '_' + self.kernel.name +'_'+ str(self.errors)+ '_alpha.npy'
        path = os.path.join(path, name)
        np.save(path, np.array(self.alpha, dtype=object), allow_pickle=True)

    def load(self, path, x, y):
        self.alpha = np.load(path,allow_pickle=True)
        self.x = x
        self.y = y


class HammingClassifier(ErrorCorrectingClassifier):
    def __init__(self, num_classes, model, drop_cols=None, class_order=None):
        super().__init__(num_classes, model, errors=1)
        hamming_code = np.array([[1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                                 [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                                 [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]])

        if drop_cols is None:
            drop_cols = [1]
        code = np.delete(hamming_code == 0, drop_cols, axis=1)
        if class_order is None:
            class_order = np.array([7, 6, 2, 9, 5, 8, 4, 3, 0, 1])

        self.set_groups(code, class_order)
        self.method = 'hamming'


class GolayClassifier(ErrorCorrectingClassifier):
    def __init__(self, num_classes, model, class_order=None):
        super().__init__(num_classes, model, errors=3)
        golay_code = np.array([[1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                               [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                               [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                               [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                               [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                               [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                               [1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                               [0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
                               [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

        code = (golay_code == 0)
        if class_order is None:
            class_order = np.array([7, 8, 1, 4, 5, 3, 2, 6, 0, 9])

        self.set_groups(code, class_order)
        self.method = 'golay'


class CustomClassifier(ErrorCorrectingClassifier):
    def __init__(self, num_classes, model):
        super().__init__(num_classes, model)
        class_order = np.arange(10)
        code = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                         [1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                         [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]])
        self.set_groups(code == 0, class_order)
        self.code = (code == 0)
        self.pairs = self.groups[self.num_classes + 1:]

        self.method = 'custom'

    def fit(self, x, y, n_jobs=5):
        super().fit(x, y, n_jobs)

        kernel_matrix = self.model.K
        self.betas = []
        for pair in self.pairs:
            new_y = np.where(y == pair[0], np.ones_like(y), -np.ones_like(y))
            mask = np.logical_or(y == pair[0], y == pair[1])
            self.model.K = cp.psd_wrap(kernel_matrix[mask][:, mask])
            beta = self.model.fit(x[mask], new_y[mask])
            self.betas.append(beta)

    def predict(self, x, check_correctness=False):
        kernel_matrix = self.kernel.matrix(self.x, x)
        scores = np.array(self.alpha) @ kernel_matrix
        old_res = np.argmax(scores[:self.num_classes], 0)

        is_vehicle = (scores[self.num_classes] > 0) * 1.
        group_res = np.argmax(scores[self.num_classes + 1:], 0)
        aligns = (is_vehicle == np.isin(group_res, [0, 3]))
        if check_correctness:
            one_activation = (np.sum(scores[self.num_classes + 1:] > 0, 0) == 1)
            aligns = np.logical_and(aligns, one_activation)
        # print(np.sum(aligns))

        new_res = np.zeros_like(old_res)
        for i, pair in enumerate(self.pairs):
            mask = np.logical_or(self.y == pair[0], self.y == pair[1])
            score = self.betas[i] @ kernel_matrix[mask]
            res = np.where(score > 0, np.ones_like(score) * pair[0], np.ones_like(score) * pair[1])
            new_res[group_res == i] = res[group_res == i]

        return np.where(aligns, new_res, old_res)


class BestClassifier(ErrorCorrectingClassifier):
    def __init__(self, num_classes, model):
        super().__init__(num_classes, model)
        class_order = np.arange(10)
        code = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                         [1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                         [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]])
        self.set_groups(code == 0, class_order)
        self.method = 'best'

    def predict(self, x):
        scores = np.array(self.alpha) @ self.kernel.matrix(self.x, x)
        old_scores = scores[:self.num_classes]

        for i, classes in enumerate(self.groups[self.num_classes + 1:]):
            old_scores[classes[0]] += scores[self.num_classes + 1 + i]
            old_scores[classes[1]] += scores[self.num_classes + 1 + i]

        old_scores[[0, 1, 8, 9]] += scores[self.num_classes]
        old_scores[[2, 3, 4, 5, 6, 7]] -= scores[self.num_classes]

        return np.argmax(old_scores[:self.num_classes], 0)
