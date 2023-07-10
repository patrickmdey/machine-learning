import pandas as pd
import numpy as np
import seaborn as sns
import json, os, warnings
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def get_config_values():
    with open("config.json") as config_file:
        config = json.load(config_file)
        n_estimators = config["n_estimators"] if "n_estimators" in config else 50
        learning_rate = config["learning_rate"] if "learning_rate" in config else 1
        test_size = config["test_size"] if "test_size" in config else 0.2

        partitions = config["partitions"] if "partitions" in config else 10

    return n_estimators, learning_rate, test_size, partitions


def categorize_columns(df, columns):
    for column_name in columns:
        quartiles = df[column_name].quantile([0.25, 0.5, 0.75, 1])
        df[column_name] = pd.qcut(df[column_name], 4, labels=[0, 1, 2, 3])
    return df


def prepare_dataset(df):
    # Adaboost doesnt support categorical data so it needs to be converted to numerical
    mappings = {
        "Fbs": {">120": 1, "<=120": 0},
        "Sex": {"F": 0, "M": 1},
        "ChestPain": {"typical": 0, "asymptomatic": 1, "nonanginal": 2, "nontypical": 3},
        "RestECG": {"normal": 0, "abnormal": 1},
        "ExAng": {"No": 0, "Yes": 1},
        "Slope": {"down": 0, "level": 1, "up": 2},
        "Thal": {"normal": 0, "fixed": 1, "reversable": 2},
        "HDisease": {"No": 0, "Yes": 1}
    }

    for column, mapping in mappings.items():
        heart_df[column] = heart_df[column].map(mapping)

    # TODO
    # , "Oldpeak"] Oldpeak is not working
    columns = ["Age", "RestBP", "Chol", "MaxHR"]
    df = categorize_columns(df, columns)
    return df


def plot_heatmap(df, partition_amount, n_estimators, learning_rate):

    warnings.filterwarnings("ignore", category=UserWarning, message="FixedFormatter should only be used together with FixedLocator")
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

    title = "Matriz de confusión con " + str(partition_amount) + " particiones, " + str(n_estimators) + \
        " estimadores y " + str(learning_rate) + " de learning rate"

    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()

    path = "simulation_out/" + str(partition_amount) + "/" + \
        str(n_estimators) + "_estimators/" + \
        str(learning_rate) + "_learning_rate/"

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
    init_path += str(learning_rate) + "_learning_rate/"
    os.mkdir(init_path) if not os.path.exists(init_path) else None


if __name__ == "__main__":

    heart_df = pd.read_csv("heart.csv")
    heart_df = prepare_dataset(heart_df)

    n_estimators, learning_rate, test_size, partitions = get_config_values()

    path = "simulation_out/"
    create_sim_out_dir(path)

    train_scores = []
    test_scores = []
    for i in range(0, partitions):
        train, test = train_test_split(heart_df, test_size=0.2)

        train_y = train["HDisease"]
        train_x = train.drop("HDisease", axis=1)
        test_y = test["HDisease"]
        test_x = test.drop("HDisease", axis=1)

        model = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))

    to_append = {"estimators": n_estimators, "mean_train_score": np.mean(train_scores), "std_train_score": np.std(
        train_scores), "mean_test_score": np.mean(test_scores), "std_test_score": np.std(test_scores),
        "max_test_score": np.max(test_scores), "max_train_score": np.max(train_scores)
    }

    precisions_path = "simulation_out/" + str(partitions) + "/" + \
        "precisions_" + str(learning_rate) + ".csv"

    if not os.path.exists(precisions_path):
        pd.DataFrame([to_append]).to_csv(precisions_path)
    else:
        metric_df = pd.read_csv(precisions_path, usecols=["estimators", "mean_train_score", "std_train_score",
                                "mean_test_score", "std_test_score", "max_train_score", "max_test_score"])
        pd.concat([metric_df, pd.DataFrame([to_append])]).to_csv(precisions_path)

    confusion_matrix = confusion_matrix(test_y, model.predict(test_x))

    # for better visualization
    true_negatives = confusion_matrix[0][0]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    true_positives = confusion_matrix[1][1]

    cm_df = pd.DataFrame({"0": [true_negatives, false_negatives], "1": [
                         false_positives, true_positives]})

    cm_df = cm_df.apply(confusion_row_to_percent, axis=1)
    plot_heatmap(cm_df, partitions, n_estimators, learning_rate)
