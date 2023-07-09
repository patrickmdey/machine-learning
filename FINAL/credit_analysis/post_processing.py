import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import warnings


def save_heatmap(df, partition_amount, n_estimators, learning_rate):
    # Silence the warning
    warnings.filterwarnings(
        'ignore', message='FixedFormatter should only be used together with FixedLocator cbar.ax.set_yticklabels(tick_labels)')
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)

    ax = sns.heatmap(df, cmap=cmap,
                     annot=True, fmt=".2%", xticklabels=["No Devuelve", "Devuelve"], yticklabels=["No Devuelve", "Devuelve"])

    cbar = ax.collections[0].colorbar
    tick_labels = cbar.ax.get_yticklabels()
    tick_values = cbar.get_ticks()
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_text(f"{int(tick_values[i] * 100)}%")
    cbar.ax.set_yticklabels(tick_labels)

    title = "Matriz de confusión con " + str(partition_amount) + " particiones, " + str(n_estimators) + \
        " estimadores y " + str(learning_rate) + " learning rate"

    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()
    path = "out/"+str(partition_amount) + "/" + str(n_estimators) + \
        "_estimators/" + str(learning_rate) + "_learning_rate/"

    plt.savefig(path + "confusion_matrix.png")


def confusion_row_to_percent(row):
    total = row.sum()
    return row.apply(lambda x: (x / total).round(4))


def compute_metrics(row, stats, confusion_matrix):
    predicted = row['predicted']
    real = row['real']
    if predicted == real:
        stats[real]["tp"] += 1
    else:
        stats[real]["fp"] += 1

    confusion_matrix[real][predicted] += 1


def calculate_metrics(df, confusion_matrix):
    stats_per_rating = {}
    for stars in range(0, 2):
        stats_per_rating[stars] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    df.apply(lambda row: compute_metrics(
        row, stats_per_rating, confusion_matrix), axis=1)
    return stats_per_rating


def calculate_results(df, confusion_matrix=None):
    correct = 0
    for _, row in df.iterrows():
        predicted = row['predicted']
        real = row['real']
        if predicted == real:
            correct += 1

        if confusion_matrix is not None:
            confusion_matrix[real][predicted] += 1

    return correct


def get_metrics(partition_amount, n_estimators, learning_rate):
    classes = range(0, 2)
    confusion_matrix = {real_cat: {
        pred_cat: 0 for pred_cat in classes} for real_cat in classes}

    pre_path = "simulation_out/" + str(partition_amount) + "/" + str(
        n_estimators) + "_estimators/" + str(learning_rate) + "_learning_rate/"

    precision_per_partition = {"train": [], "test": []}

    for partition in range(partition_amount):
        post_path = "_" + str(partition) + ".csv"

        test_df = pd.read_csv(pre_path + "test/classification" + post_path)
        train_df = pd.read_csv(pre_path + "train/classification" + post_path)

        current_correct = calculate_results(test_df, confusion_matrix)
        precision_per_partition["test"].append(current_correct/len(test_df))

        # TODO: meterle un if para esto hacerlo solo en el caso de precision vs nodos
        current_correct = calculate_results(train_df)
        precision_per_partition["train"].append(current_correct/len(train_df))

    confusion_df = pd.DataFrame(confusion_matrix)
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)
    save_heatmap(confusion_df, partition_amount, n_estimators, learning_rate)

    mean_std_precision = {"train":
                          {"mean": np.mean(precision_per_partition["train"]),
                           "std": np.std(precision_per_partition["train"]),
                           "max_precision": max(precision_per_partition['train'])},
                          "test": {"mean": np.mean(precision_per_partition["test"]),
                                   "std": np.std(precision_per_partition["test"]),
                                   "max_precision": max(precision_per_partition['test'])}}

    return mean_std_precision


def results_to_csv(precision, partition_amount, n_estimators, learning_rate):
    to_append = {
        "estimators": n_estimators,
        "mean_test_precision": precision["test"]["mean"], "std_test_precision": precision["test"]["std"], "max_test_precision": precision["test"]["max_precision"],
        "mean_train_precision": precision["train"]["mean"], "std_train_precision": precision["train"]["std"], "max_train_precision": precision["train"]["max_precision"]
    }

    path = "out/" + str(partition_amount) + "/" + str(n_estimators) + \
        "_estimators/" + str(learning_rate) + "_learning_rate/" + "precision.csv"
    
    path= "out/" + str(partition_amount) + "/precision" + "_learning_rate_" + str(learning_rate) + ".csv"

    if not os.path.exists(path):
        pd.DataFrame([to_append]).to_csv(path)
    else:
        metric_df = pd.read_csv(path, usecols=["estimators", "mean_test_precision", "std_test_precision",
                                "mean_train_precision", "std_train_precision", "max_test_precision", "max_train_precision"])
        pd.concat([metric_df, pd.DataFrame([to_append])]).to_csv(path)


if __name__ == "__main__":

    partition_amount = int(sys.argv[1] if len(sys.argv) > 1 else 4)
    n_estimators = int(sys.argv[2] if len(sys.argv) > 2 else 50)
    learning_rate = float(sys.argv[3] if len(sys.argv) > 3 else 0.1)

    os.mkdir("out/") if not os.path.exists("out/") else None
    path = "out/" + str(partition_amount) + "/"
    os.makedirs(path) if not os.path.exists(path) else None
    path += str(n_estimators)+"_estimators/"
    os.makedirs(path) if not os.path.exists(path) else None
    path += str(learning_rate)+"_learning_rate/"
    os.makedirs(path) if not os.path.exists(path) else None

    # os.mkdir(path) if not os.path.exists(path) else None

    precision = get_metrics(partition_amount, n_estimators, learning_rate)

    results_to_csv(precision, partition_amount, n_estimators, learning_rate)
