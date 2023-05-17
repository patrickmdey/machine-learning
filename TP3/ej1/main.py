import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Perceptron import Perceptron
from SVM import SVM
from utils import partition_dataset
import json


def plot_preceptron(X, y, weights):
    plt.clf()
    line_x = np.linspace(0, 1, 2)
    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * weights[0] + weights[2])/weights[1]

    plt.plot(line_x, line_y)
    plt.scatter(X[:, 0], X[:, 1], color=[
                'red' if c == -1 else 'green' for c in y])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Perceptron')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Decision boundary', 'Instance'])
    plt.savefig('perceptron.png')


def plot_svm(X, y, r, weights, b):
    plt.clf()
    line_x = np.linspace(0, 1, 2)

    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * weights[0] + b)/weights[1]
    line_y_up = -(line_x * weights[0] + b + r)/weights[1]
    line_y_down = -(line_x * weights[0] + b - r)/weights[1]
    plt.scatter(X[:, 0], X[:, 1], color=['red' if c == -1 else 'green' for c in y])
    plt.plot(line_x, line_y, color='black')
    plt.plot(line_x, line_y_up, linestyle='dashed', color='blue')
    plt.plot(line_x, line_y_down, linestyle='dashed', color='blue')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.legend(['Instance', 'Decision boundary', 'Margin'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SVM')
    plt.savefig('svm.png')


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


def run_perceptron(df):
    X = df.loc[:, ['x', 'y']].values
    y = df.loc[:, ['class']].values.ravel()

    perceptron = Perceptron(2)
    error, weights = perceptron.train(X, y)
    print("Perceptron:", weights)
    # for i in range(len(X)):
    #     prediction = perceptron.predict(weights, np.append(X[i], [1]))
    # print("Prediction: " + str(prediction) + ", Real: " + str(y[i]))
    plot_preceptron(X, y, weights)


def run_svm(df, max_c, c_rate, test_pctg, epochs=1000, learning_rate=0.01):
    c_precisions = []
    for c in range(1, max_c, c_rate):
        precisions = []
        partitions = partition_dataset(df, test_pctg)
        for idx, partition in enumerate(partitions):
            test = partition
            train = pd.concat([df for df in partitions if df is not partition])

            train_X = train.loc[:, ['x', 'y']].values
            train_y = train.loc[:, ['class']].values.ravel()

            svm = SVM(2)
            error, weights, b = svm.train(
                train_X, train_y, c, learning_rate, epochs)

            test_X = test.loc[:, ['x', 'y']].values
            test_y = test.loc[:, ['class']].values.ravel()

            correct = 0
            for idx, x in enumerate(test_X):
                prediction = svm.predict(weights, b, x)
                if prediction == test_y[idx]:
                    correct += 1

            precisions.append(correct / len(test_X))
        c_precisions.append(
            {"mean": np.mean(precisions), "std": np.std(precisions)})

    optimal_c = np.argmax([c["mean"] for c in c_precisions]) + 1

    X = df.loc[:, ['x', 'y']].values
    y = df.loc[:, ['class']].values.ravel()
    svm = SVM(2)
    min_error, best_weights, best_b = svm.train(
        X, y, optimal_c, learning_rate, epochs)
    print("SVM best weights and error:", best_weights, min_error)
    r = svm.calculate_margin(best_weights)

    print("SVM:", best_weights)
    print("Margin:", r)
    plot_svm(X, y, r, best_weights, best_b)

    for idx, x in enumerate(X):
        prediction = svm.predict(best_weights, best_b, x)
        y[idx] = prediction

    plot_svm(X, y, r, best_weights, best_b)


if __name__ == '__main__':

    with open("config.json") as config_file:
        config = json.load(config_file)
        generate = config["generate"] if "generate" in config else True
        file_name = config["file_name"] if "file_name" in config else "TP3-1"
        method = config["method"] if "method" in config else "SVM"
        point_amount = config["point_amount"] if "point_amount" in config else 30

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1

    if generate:
        data, line = random_points_within_range(
            x_min, x_max, y_min, y_max, point_amount)
        df = pd.DataFrame(data, columns=["x", "y", "class"])
        df.to_csv(file_name+".csv", index=False)
        line_df = pd.DataFrame([line], columns=["m", "b"])
        line_df.to_csv(file_name+"-line.csv", index=False)
    else:
        df = pd.read_csv(file_name+".csv")
        line = pd.read_csv(file_name+"-line.csv").values

    run_perceptron(df)

    test_pctg = 0.1

    max_c = 250
    c_rate = 50
    epochs = 10000
    learning_rate = 0.001

    run_svm(df, max_c, c_rate, test_pctg, epochs, learning_rate)
