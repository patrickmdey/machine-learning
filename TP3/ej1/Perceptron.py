import numpy as np


class Perceptron:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # self.weights = np.zeros(input_dim + 1)
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=input_dim+1)

    def predict(self, weights, train_example):
        return np.sign(np.dot(train_example, weights))

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        error = 100
        min_error = len(X)
        curr_epoch = 0
        min_w = self.weights
        while error > 0.000001 and curr_epoch < epochs:
            i = np.random.randint(0, len(X))
            train_example = np.append(X[i], [1])  # we append bias

            delta_w = learning_rate * \
                (y[i] - self.predict(self.weights, train_example)) * train_example
            self.weights += delta_w

            error = self.calculate_error(X, y)

            if error < min_error:
                min_error = error
                min_w = self.weights

            curr_epoch += 1

        self.weights = min_w
        return (min_error, min_w)

    def calculate_error(self, X, y):
        error = 0
        for i in range(len(X)):
            train_example = np.append(X[i], [1])
            activation_o = self.predict(self.weights, train_example)
            error += (activation_o - y[i])**2

        return error
