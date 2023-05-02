import numpy as np


class Perceptron:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0, 1, -1)

    def train(self, X, y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            y_pred = self.predict(X)

            error = y - y_pred

            # Update the weights and bias
            self.weights += learning_rate * np.dot(X.T, error)
            self.bias += learning_rate * np.sum(error)
