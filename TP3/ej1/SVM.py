import numpy as np


class SVM:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # self.weights = np.zeros(input_dim)
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=input_dim)
        self.b = 0.0

    def train(self, X, y, C, learning_rate, epochs=10000):
        learning_rate = 0.0001  # TODO: set constant
        min_error = np.inf
        best_weights = self.weights
        best_b = 0.0
        for epoch in range(1, epochs):
            i = np.random.randint(0, len(X))
            learning_rate = learning_rate * np.exp(-0.0001 * epoch)

            t = y[i] * (np.dot(X[i], self.weights) + self.b)

            if t < 1:
                self.weights = self.weights - learning_rate * \
                    (self.weights - C * y[i] * X[i])
                self.b = self.b + learning_rate * (C * y[i])
            else:
                self.weights = self.weights - learning_rate * self.weights

            error = self.compute_error(X, y)
            if error < min_error:
                min_error = error
                best_weights = self.weights
                best_b = self.b

        return min_error, best_weights, best_b

    def compute_error(self, X, y):
        tot_error = 0
        for i in range(len(X)):
            tot_error += (y[i] - self.predict(self.weights, self.b, X[i]))**2
        return tot_error

    def predict(self, weights, b, X):
        return np.sign(np.dot(X, weights) + b)

    def calculate_margin(self, weights):
        return 1 / np.linalg.norm(weights)
