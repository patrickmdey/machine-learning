import numpy as np
import matplotlib.pyplot as plt
import os

class Perceptron:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=input_dim+1)

    def predict(self, weights, train_example):
        return np.sign(np.dot(train_example, weights))

    def train(self, X, y, epochs=1000, learning_rate=0.1, animate=False):
        error = 100
        min_error = len(X)
        curr_epoch = 0
        min_w = self.weights
        while error > 0.000001 and curr_epoch < epochs:
            i = np.random.randint(0, len(X))
            train_example = np.append(X[i], [1])  # we append bias

            delta_w = learning_rate * (y[i] - self.predict(self.weights, train_example)) * train_example
            self.weights += delta_w

            error = self.calculate_error(X, y)

            if error < min_error:
                min_error = error
                min_w = self.weights

            if animate and curr_epoch % 10 == 0:
                self.plot_perceptron(X, y, curr_epoch)

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

    def plot_perceptron(self, X, y, epoch):
        plt.clf()
        line_x = np.linspace(0, 5, 2)
        # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
        line_y = -(line_x * self.weights[0] + self.weights[2])/self.weights[1]

        plt.plot(line_x, line_y, color='blue')

        plt.xlim([0, 5])
        plt.ylim([0, 5])
        plt.xlabel('x')
        plt.ylabel('y')

        point_amount = len(y)

        plt.scatter(X[:, 0], X[:, 1], color=[
                    'red' if c == -1 else 'green' for c in y])

        path = "out/perceptron/animation/"
        os.mkdir(path) if not os.path.exists(path) else None
        path += str(point_amount)+"_points_animation_epoch_"+str(epoch)

        plt.tight_layout()
        plt.savefig(path+'.png')
