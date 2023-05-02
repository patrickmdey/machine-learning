import numpy as np
import matplotlib.pyplot as plt
from TP3.ej1.Perceptron import Perceptron


def random_points_within_range(x_min, x_max, y_min, y_max, n):
    x_points = np.random.uniform(x_min, x_max, size=n)
    y_points = np.random.uniform(y_min, y_max, size=n)

    x1 = np.random.uniform(x_min, x_max)
    y1 = np.random.uniform(y_min, y_max)
    x2 = np.random.uniform(x_min, x_max)
    y2 = np.random.uniform(y_min, y_max)

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    data = []
    for i in range(n):
        x = x_points[i]
        y = y_points[i]

        if y >= m * x + b:
            classification = 1
        else:
            classification = -1

        data.append([x, y, classification])

    return np.array(data), (m, b)


if __name__ == '__main__':
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    n = 30
    data, line = random_points_within_range(x_min, x_max, y_min, y_max, n)

    X = data[:, :2]
    y = data[:, -1]
    perceptron = Perceptron(2)
    perceptron.train(X, y)

    for idx, x in enumerate(X):
        prediction = perceptron.predict(x)
        if prediction == y[idx]:
            print("nice")
        else:
            print("not nice")

    line_x = np.linspace(0, 1, 2)
    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * perceptron.weights[0] + perceptron.bias)/perceptron.weights[1]

    plt.scatter(X[:, 0], X[:, 1], color=['red' if c == -1 else 'green' for c in y])
    plt.plot(line_x, line_y)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
