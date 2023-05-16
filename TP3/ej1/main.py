import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Perceptron import Perceptron
from SVM import SVM
from utils import partition_dataset

def plot_preceptron(X, y, perceptron):
    plt.clf()
    line_x = np.linspace(0, 1, 2)
    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * perceptron.weights[0] + perceptron.weights[2])/perceptron.weights[1]

    plt.plot(line_x, line_y)
    plt.scatter(X[:, 0], X[:, 1], color=['red' if c == -1 else 'green' for c in y])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Perceptron')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Decision boundary', 'Instance'])
    # plt.show()
    plt.savefig('perceptron.png')

def plot_svm(X, y, r, svm):
    plt.clf()
    line_x = np.linspace(0, 1, 2)
    
    # w1*x1 + w2*x2 + b = 0 => -(w1*x1 + b)/w2 = x2
    line_y = -(line_x * svm.weights[0] + svm.bias)/svm.weights[1]
    line_y_up = -(line_x * svm.weights[0] + svm.bias + r)/svm.weights[1]
    line_y_down = -(line_x * svm.weights[0] + svm.bias - r)/svm.weights[1]
    plt.scatter(X[:, 0], X[:, 1], color=['red' if c == -1 else 'green' for c in y])
    plt.plot(line_x, line_y, color='black')
    # plt.plot(line_x, line_y_up, linestyle='dashed', color='blue')
    # plt.plot(line_x, line_y_down, linestyle='dashed', color='blue')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.legend(['Instance','Decision boundary', 'Margin'])
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
    perceptron.train(X, y)
    print("Perceptron:", perceptron.weights)
    for i in range(len(X)):
        prediction = perceptron.predict(np.append(X[i], [1]))
        # print("Prediction: " + str(prediction) + ", Real: " + str(y[i]))
    plot_preceptron(X, y, perceptron)


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
            svm.train(train_X, train_y, c, learning_rate, epochs)

            test_X = test.loc[:, ['x', 'y']].values
            test_y = test.loc[:, ['class']].values.ravel()

            correct = 0
            for idx, x in enumerate(test_X):
                prediction = svm.predict(x)
                if prediction == test_y[idx]:
                    correct += 1
            
            precisions.append(correct / len(test_X))
        c_precisions.append({"mean": np.mean(precisions), "std": np.std(precisions)})
    
    optimal_c = np.argmax([c["mean"] for c in c_precisions]) + 1

    X = df.loc[:, ['x', 'y']].values
    y = df.loc[:, ['class']].values.ravel()
    svm = SVM(2)
    svm.train(X, y, optimal_c, learning_rate, epochs)
    r = svm.calculate_margin()

    print("SVM:", svm.weights)
    print("Margin:", r)
    plot_svm(X, y, r, svm)

    for idx, x in enumerate(X):
        prediction = svm.predict(x)
        y[idx] = prediction
    
    plot_svm(X, y, r, svm)

if __name__ == '__main__':

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    n = 30
    data, line = random_points_within_range(x_min, x_max, y_min, y_max, n)

    df = pd.DataFrame(data, columns=["x", "y", "class"])
    df.to_csv("TP3-1.csv", index=False)

    run_perceptron(df)

    test_pctg = 0.1

    max_c = 250
    c_rate = 50
    epochs = 10000
    learning_rate = 0.001

    # run_svm(df, max_c, c_rate, test_pctg, epochs, learning_rate)

    
