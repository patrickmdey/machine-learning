import numpy as np


class KNN:
    def __init__(self, k):
        self.stored_classifications = None
        self.stored_attrs = None
        self.k = k

    def fit(self, X, y):
        self.stored_attrs = X
        self.stored_classifications = y
        print("X[:5]:")
        print(X[:5])
        print("y[:5]:")
        print(y[:5])

    def predict(self, instance, weighted=False):
        distances = np.sqrt(np.sum((self.stored_attrs - instance) ** 2, axis=1).astype(np.float64))
        # Argsort returns the indices that would sort an array
        sorted_indices = np.argsort(distances)[:self.k]
        classifications = self.stored_classifications.T[0]

        weights = {}
        for idx in sorted_indices:
            if classifications[idx] not in weights:
                weights[classifications[idx]] = 0
            # TODO: check distance 0 case
            weights[classifications[idx]] += (1 / distances[idx]) if weighted else 1

        # return max key
        print("weights:", weights)
        return max(weights, key=weights.get)

        # k_nearest = self.stored_classifications[np.argsort(distances)[:self.k]]
        # print("k_nearest:", k_nearest)

        # print("k_nearest.T[0]:", k_nearest.T[0])

        # return np.bincount(k_nearest.T[0]).argmax()
