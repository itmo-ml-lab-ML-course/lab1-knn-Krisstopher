import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Kernel import Kernel


class Lowess:
    def __init__(self, knn, lowess_kernel=Kernel.GAUSSIAN):
        self.lowess_kernel = lowess_kernel
        self.knn = knn

    def correct_weights(self, x_train, y_train):
        weights = []
        classes_by_lex = np.sort(np.unique(y_train))

        if isinstance(self.knn, KNeighborsClassifier):
            all_scores = self.knn.predict_proba(x_train)

        for i in range(len(x_train)):
            self.knn.fit(np.delete(x_train, i, axis=0), np.delete(y_train, i))
            if isinstance(self.knn, KNeighborsClassifier):
                ind = np.where(classes_by_lex == y_train[i])[0][0]
                weight = self.lowess_kernel(1 - all_scores[i][ind])
            else:
                scores = self.knn.compute_scores(x_train[i])
                weight = self.lowess_kernel(1 - scores[y_train[i]])

            weights.append(weight)

        return weights