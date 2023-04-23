import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        distances = np.sqrt(np.sum((self.X_train - X)**2, axis=1).astype(np.float64))
        k_nearest = self.y_train[np.argsort(distances)[:self.k]]
        return np.bincount(k_nearest.T[0]).argmax()