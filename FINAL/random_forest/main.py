import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
import warnings
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from utils import prepare_dataset


def get_config_values():
    with open("config.json") as config_file:
        config = json.load(config_file)
        n_estimators = config["tree_amount"] if "tree_amount" in config else 50
        test_size = config["test_percentage"] if "test_percentage" in config else 0.2
        partitions = config["partitions"] if "partitions" in config else 10
    return n_estimators, test_size, partitions


def plot_heatmap(df, partition_amount, n_estimators, prec):
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="FixedFormatter should only be used together with FixedLocator")
    # Silence the warning
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)

    ax = sns.heatmap(df, cmap=cmap,
                     annot=True, fmt=".2%", xticklabels=["No tiene", "Tiene"], yticklabels=["No tiene", "Tiene"])

    cbar = ax.collections[0].colorbar
    tick_labels = cbar.ax.get_yticklabels()
    tick_values = cbar.get_ticks()
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_text(f"{int(tick_values[i] * 100)}%")
    cbar.ax.set_yticklabels(tick_labels)

    title = "Matriz de confusión con " + str(partition_amount) + " particiones y " + str(n_estimators) + \
        " estimadores" #+ "\nPrecisión: " + str(prec)

    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()

    path = "simulation_out/" + str(partition_amount) + "/" + \
        str(n_estimators) + "_estimators/"

    plt.savefig(path + "confusion_matrix.png")


def confusion_row_to_percent(row):
    total = row.sum()
    return row.apply(lambda x: (x / total).round(4))


def create_sim_out_dir(init_path):
    os.mkdir(init_path) if not os.path.exists(init_path) else None
    init_path += str(partitions) + "/"
    os.mkdir(init_path) if not os.path.exists(init_path) else None
    init_path += str(n_estimators) + "_estimators/"
    os.mkdir(init_path) if not os.path.exists(init_path) else None


if __name__ == "__main__":

    heart_df = pd.read_csv("../heart.csv")
    heart_df = prepare_dataset(heart_df)

    n_estimators, test_size, partitions = get_config_values()

    path = "simulation_out/"
    create_sim_out_dir(path)

    train_precs = []
    test_precs = []

    train_accuracies = []
    test_accuracies = []
    # train, test = train_test_split(heart_df, test_size=0.2)

    skf = StratifiedKFold(n_splits=partitions, shuffle=True)

    X = heart_df.drop("HDisease", axis=1)
    y = heart_df["HDisease"]

    confusion_matrixes = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        train_y = y.iloc[train_index]
        train_x = X.iloc[train_index]
        test_y = y.iloc[test_index]
        test_x = X.iloc[test_index]

        model = RandomForestClassifier(
            n_estimators=n_estimators)
        model.fit(train_x, train_y)

        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)

        train_precs.append(precision_score(train_y, train_predictions))
        test_precs.append(precision_score(test_y, test_predictions))

        train_accuracies.append(accuracy_score(train_y, train_predictions))
        test_accuracies.append(accuracy_score(test_y, test_predictions))

        confusion_matrixes.append(confusion_matrix(test_y, test_predictions))

    to_append = {"estimators": n_estimators, "mean_train_prec": np.mean(train_precs), "std_train_prec": np.std(
        train_precs), "mean_test_prec": np.mean(test_precs), "std_test_prec": np.std(test_precs), "mean_train_acc": np.mean(train_accuracies),
        "mean_test_acc": np.mean(test_accuracies), "std_train_acc": np.std(train_accuracies), "std_test_acc": np.std(test_accuracies),
        "max_test_prec": np.max(test_precs), "max_train_prec": np.max(train_precs)
    }

    precisions_path = "simulation_out/" + str(partitions) + "/" + \
        "precisions.csv"

    if not os.path.exists(precisions_path):
        pd.DataFrame([to_append]).to_csv(precisions_path)
    else:
        metric_df = pd.read_csv(precisions_path, usecols=["estimators", "mean_train_prec", "std_train_prec",
                                "mean_test_prec", "std_test_prec", "mean_train_acc", "mean_test_acc", "std_train_acc", "std_test_acc", "max_test_prec", "max_train_prec"])
        pd.concat([metric_df, pd.DataFrame([to_append])]
                  ).to_csv(precisions_path)

    best_configuration = {"partitions": partitions, "estimators": n_estimators,
                          "mean_test_prec": np.mean(test_precs), "std_test_prec": np.std(test_precs), "max_test_prec": np.max(test_precs)}

    best_configuration_path = "simulation_out/best_config.csv"
    if not os.path.exists(best_configuration_path):
        pd.DataFrame([best_configuration]).to_csv(best_configuration_path)
    else:
        best_partition_df = pd.read_csv(best_configuration_path, usecols=[
                                        "partitions", "estimators", "mean_test_prec", "std_test_prec", "max_test_prec"])
        pd.concat([best_partition_df, pd.DataFrame([best_configuration])]).to_csv(
            best_configuration_path)

    confusion_matrix = np.sum(confusion_matrixes, axis=0)

    # for better visualization
    true_negatives = confusion_matrix[0][0]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    true_positives = confusion_matrix[1][1]

    cm_df = pd.DataFrame({"0": [true_negatives, false_negatives], "1": [
                         false_positives, true_positives]})
    
    cm_df = cm_df.apply(confusion_row_to_percent, axis=1)
    plot_heatmap(cm_df, partitions, n_estimators, np.mean(test_precs))
