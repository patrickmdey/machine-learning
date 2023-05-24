import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Perceptron import Perceptron
from SVM import SVM
from utils import *
import json
import os


def optimize_perceptron(X, y, weights, upper_amount=1, lower_amount=2):
    A = weights[0]
    B = weights[1]
    C = weights[2]

    upper_points = X[y == 1]
    lower_points = X[y == -1]

    upper_distances = np.abs(
        A * upper_points[:, 0] + B * upper_points[:, 1] + C) / np.sqrt(A**2 + B**2)
    lower_distances = np.abs(
        A * lower_points[:, 0] + B * lower_points[:, 1] + C) / np.sqrt(A**2 + B**2)

    upper_points = upper_points[np.argsort(upper_distances)[:upper_amount]]
    lower_points = lower_points[np.argsort(lower_distances)[:lower_amount]]

    lower_midpoint = (lower_points[0] + lower_points[1]) / 2

    line_x = np.linspace(0, 5, 2)

    optimal_point = (upper_points[0] + lower_midpoint) / 2

    m = (lower_points[1][1] - lower_points[0][1]) / \
        (lower_points[1][0] - lower_points[0][0])
    b = optimal_point[1] - m * optimal_point[0]

    line_y = m * line_x + b

    return line_y, upper_points, lower_points, optimal_point


def plot_preceptron(X, y, weights, epochs, l_rate, noisy=False, plot_optimal=False):
    plt.clf()
    line_x = np.linspace(0, 5, 2)
    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * weights[0] + weights[2])/weights[1]

    plt.plot(line_x, line_y, color='blue')

    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.title('Perceptron con ' + str(epochs) +
              ' épocas y learning rate ' + str(l_rate))
    plt.xlabel('x')
    plt.ylabel('y')

    point_amount = len(y)
    if plot_optimal:
        optimal_line_y, upper_points, lower_points, optimal_point = optimize_perceptron(
            X, y, weights)

        to_remove_idx = []
        for idx, point in enumerate(X):
            if point in upper_points or point in lower_points:
                to_remove_idx.append(idx)

        X = np.delete(X, to_remove_idx, axis=0)
        y = np.delete(y, to_remove_idx, axis=0)
        plt.scatter(optimal_point[0], optimal_point[1],
                    color='mediumpurple', marker="^")
        lower_middle_point = (lower_points[0] + lower_points[1]) / 2
        plt.scatter(lower_middle_point[0],
                    lower_middle_point[1], color='mediumpurple', marker="^")  # lower middle point

        plt.plot(line_x, optimal_line_y,
                 color='mediumpurple', linestyle='dashed')
        plt.scatter(upper_points[:, 0], upper_points[:,
                    1], color='lightblue')  # upper point
        plt.scatter(lower_points[:, 0], lower_points[:,
                    1], color='pink')  # lower points

        point_amount = len(y) + 3

    plt.scatter(X[:, 0], X[:, 1], color=[
                'red' if c == -1 else 'green' for c in y])

    path = "out/perceptron/" + ("with" if noisy else "no") + "_noise"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/" + str(point_amount)+"_points"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/"+str(epochs)+"_epochs"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/"+str(l_rate).replace(".", "p") + "_lrate"
    os.mkdir(path) if not os.path.exists(path) else None

    path += "/hyperplane"+("_with_optimal" if plot_optimal else "")

    plt.tight_layout()
    plt.savefig(path+'.png')


def plot_svm(X, y, r, weights, b, epochs, l_rate, noisy=False):
    plt.clf()
    line_x = np.linspace(0, 5, 2)

    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * weights[0] + b)/weights[1]
    line_y_up = -(line_x * weights[0] + b + r)/weights[1]
    line_y_down = -(line_x * weights[0] + b - r)/weights[1]
    plt.scatter(X[:, 0], X[:, 1], color=[
                'red' if c == -1 else 'green' for c in y])
    plt.plot(line_x, line_y, color='black')
    # plt.plot(line_x, line_y_up, linestyle='dashed', color='blue')
    # plt.plot(line_x, line_y_down, linestyle='dashed', color='blue')
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.legend(['Instancia', 'Clasificador'], loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SVM con ' + str(epochs) +
              ' épocas y learning_rate ' + str(l_rate))

    path = "out/svm/" + ("with" if noisy else "no") + "_noise"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/"+str(point_amount)+"_points"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/"+str(epochs)+"_epochs"
    os.mkdir(path) if not os.path.exists(path) else None
    path += "/"+str(l_rate).replace(".", "p") + "_lrate"

    # path += "/svm"
    plt.tight_layout()
    plt.savefig(path + '.png')


def run_perceptron(df, epochs, learning_rate):
    X = df.loc[:, ['x', 'y']].values
    y = df.loc[:, ['class']].values.ravel()

    perceptron = Perceptron(2)
    error, weights = perceptron.train(X, y, epochs, learning_rate)
    print("Perceptron:", weights, error)

    return X, y, weights
    # plot_preceptron(X, y, weights, epochs, learning_rate, False)
    # plot_preceptron(X, y, weights, epochs, learning_rate, True)


def run_svm(df, initial_c, max_c, c_rate, test_pctg, is_noisy, epochs=1000, learning_rate=0.01):
    c_precisions = []
    for c in np.arange(initial_c, max_c, c_rate):
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
    print("Margin:", r)

    plot_svm(X, y, r, best_weights, best_b, epochs, learning_rate, is_noisy)


if __name__ == '__main__':
    with open("config.json") as config_file:
        config = json.load(config_file)
        generate = config["generate"] if "generate" in config else True
        file_name = config["file_name"] if "file_name" in config else "TP3-1"
        method = config["method"] if "method" in config else "SVM"
        point_amount = config["point_amount"] if "point_amount" in config else 30
        epochs = config["epochs"] if "epochs" in config else 1000
        learning_rate = config["learning_rate"] if "learning_rate" in config else 0.01
        error_rate = config["error_rate"] if "error_rate" in config else 0

        is_noisy = False
        if error_rate != 0:
            is_noisy = True

        x_min, x_max, y_min, y_max = 0, 5, 0, 5

        file_name += "-"+str(point_amount)
        print(file_name)

        if generate:
            # theres a 5% chance of a point being in the wrong side of the line
            data, line = random_points_within_range(
                x_min, x_max, y_min, y_max, point_amount, 0.05)
            df = pd.DataFrame(data, columns=["x", "y", "class"])
            df.to_csv(file_name+".csv", index=False)
            line_df = pd.DataFrame([line], columns=["m", "b"])
            line_df.to_csv(file_name+"-line.csv", index=False)
        else:
            df = pd.read_csv(file_name+".csv")
            line = pd.read_csv(file_name+"-line.csv").values

        if method.lower() == "perceptron":
            X, y, weights = run_perceptron(df, epochs, learning_rate)
            plot_preceptron(X, y, weights, epochs,
                            learning_rate, is_noisy, False)
            plot_preceptron(X, y, weights, epochs,
                            learning_rate, is_noisy, True)
        elif method.lower() == "svm":
            test_pctg = 0.1
            initial_c = 0.1
            max_c = 2
            c_rate = 0.2

            run_svm(df, initial_c, max_c, c_rate,
                    test_pctg,is_noisy, epochs, learning_rate)
        else:
            print("Invalid method")
    config_file.close()
