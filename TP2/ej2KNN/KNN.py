import numpy as np


class KNN:
    def __init__(self, k):
        self.stored_classifications = None
        self.stored_attrs = None
        self.classifications = None
        self.k = k

    def fit(self, X, y):
        self.stored_attrs = X
        self.stored_classifications = y
        self.classifications = self.stored_classifications.T[0]

    def predict(self, instance, weighted=False):
        distances = np.sqrt(np.sum((self.stored_attrs - instance) ** 2, axis=1).astype(np.float64))
        # Argsort returns the indices that would sort an array
        #sorted_indices = np.argsort(distances)[:self.k]
        sorted_indices = np.argsort(distances) 

        if weighted and 0.0 in distances:
            return np.bincount(self.classifications[np.where(distances == 0.0)[0]]).argmax()

        weights = self.check_neighbours(distances, sorted_indices, weighted, self.k)

        to_sum = 1
        while (len(weights) != len(set(weights))):
            weights = self.check_neighbours(distances, sorted_indices, weighted, self.k + to_sum)
            to_sum += 1

        # return max key
        return max(weights, key=weights.get)

    def check_neighbours(self, distances, sorted_indices, weighted, k):
        weights = {}
        for idx in sorted_indices[:k]:
            if self.classifications[idx] not in weights:
                weights[self.classifications[idx]] = 0
            weights[self.classifications[idx]] += (1 / distances[idx]) if weighted else 1
        return weights
