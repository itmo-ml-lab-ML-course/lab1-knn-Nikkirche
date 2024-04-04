import enum
import math
from math import floor

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

uniform_kernel = lambda k: 0.5 if abs(k) < 1 else 0
triungular_kernel = lambda k: max(0, 1 - abs(k))
gaussian_kernel = lambda k: 1 / (math.sqrt(2 * math.pi) * math.exp((-k * k) / 2))
epachnikov_kernel = lambda k: max(0, 0.75 * (1 - k ** 2))


class MyKNeighborsClassifier(BaseEstimator):
    def __init__(self, n_neighbors, metric='euclidean', kernel='triungular_kernel', is_fixed_window=True,
                 k=5 ,h  = 1.0, weights=None):
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.is_fixed_window = is_fixed_window
        self.k = k
        self.h = h
        self.weights = weights
        self.y = None

    def fit(self, x, y):
        if (self.kernel == 'uniform_kernel'):
            self.kernel = uniform_kernel
        if (self.kernel == 'triungular_kernel'):
            self.kernel = triungular_kernel
        if (self.kernel == 'gaussian_kernel'):
            self.kernel = gaussian_kernel
        else:
            self.kernel = epachnikov_kernel
        self.nn.fit(x)
        if self.weights is None:
            self.weights = [1 for _ in range(len(x))]
        self.y = y

    def predict(self, x, ignored_index =None):

        res = []
        l = max(self.y) + 1
        for el in x:
            arr = [0 for _ in range(l)]
            distances, indexes = self.nn.kneighbors([el], return_distance=True, n_neighbors=self.n_neighbors)
            for dist, index in zip(distances[0], indexes[0]):
                if(ignored_index is not  None and ignored_index == index):
                    continue
                if self.is_fixed_window:
                    div = self.h
                else:
                    div = distances[0][self.k + 1]
                arr[self.y.iloc[index]] += self.kernel(dist / div) * self.weights[index]
            res.append(np.argmax(arr))
        return res

    def get_params(self, deep=False):
        return {'metric': self.nn.metric, 'n_neighbors': self.nn.n_neighbors, 'kernel': self.kernel,
                'weights': self.weights}


def lowess(model, x, y):
    anomalies = []
    model.fit(x, y)
    for i in range(len(x)):
        res = model.predict([x[i]], i)
        if res[0] - y.iloc[i] != 0:
            anomalies.append(i)
    return anomalies
