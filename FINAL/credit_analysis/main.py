import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import os, json


def partition_dataset(df, partition_percentage):
    # shuffle dataframe rows
    df = df.sample(frac=1).reset_index(drop=True)

    partition_size = int(np.floor(len(df) * partition_percentage))
    partitions = []

    bottom = 0
    up = partition_size
    while bottom < len(df):

        partitions.append(df[bottom:up].copy())
        bottom += partition_size
        up += partition_size
        if up > len(df):
            up = len(df)

    if (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions


def categorize_columns(df, columns):
    for column_name in columns:
        quartiles = df[column_name].quantile([0.25, 0.5, 0.75, 1])
        # print(column_name, quartiles)
        df[column_name] = pd.qcut(df[column_name], 4, labels=[0, 1, 2, 3])
    return df


def get_prediction_dict(adaboost, to_predict_df, target_column, attribute_columns):
    df_to_csv = pd.DataFrame(columns=["predicted", "real"])

    for idx, instance in to_predict_df.iterrows():
        prediction = adaboost.predict([instance[attribute_columns]])
        df_to_csv = pd.concat(
            [df_to_csv, pd.DataFrame([{"predicted": prediction[0], "real": instance[target_column]}])],
            ignore_index=True)

    return df_to_csv


if __name__ == "__main__":
    csv_file = ""
    target_column = ""
    test_percentage = 0.2
    n_estimators = 50
    learing_rate = 1
    with open("config.json") as config_file:
        config = json.load(config_file)
        csv_file = config["file"]
        target_column = config["target"]
        test_percentage = config["test_percentage"] if "test_percentage" in config else 0.2
        n_estimators = config["n_estimators"] if "n_estimators" in config else 50
        learing_rate = config["learing_rate"] if "learing_rate" in config else 1

    df = pd.read_csv(csv_file)

    # Categorize
    columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    df = categorize_columns(df, columns)

    attribute_columns = df.loc[:, df.columns != target_column].columns.tolist()

    idx = 0
    partitions = partition_dataset(df, test_percentage)
    partitions_len = len(partitions)

    for partition in partitions:
        test = partition
        train = pd.concat([df for df in partitions if df is not partition])
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learing_rate)
        adaboost.fit(train[attribute_columns].values, train[target_column].values)

        node_path = "simulation_out/"
        os.mkdir(node_path) if not os.path.exists(node_path) else None

        node_path += str(partitions_len) + "/"
        os.mkdir(node_path) if not os.path.exists(node_path) else None

        node_path+= str(n_estimators)+"_estimators/"
        os.mkdir(node_path) if not os.path.exists(node_path) else None

        node_path += str(learing_rate) + "_learning_rate/"
        os.mkdir(node_path) if not os.path.exists(node_path) else None

        # TODO: adapt to new parameters
        # max_nodes_str = "no_max" if max_nodes == -1 else str(max_nodes)
        # node_path += max_nodes_str + "_nodes"
        # os.mkdir(node_path) if not os.path.exists(node_path) else None

        test_df_to_csv = get_prediction_dict(adaboost, test, target_column, attribute_columns)
        os.mkdir(node_path + "/test") if not os.path.exists(node_path + "/test") else None
        test_df_to_csv.to_csv(node_path + "/test/classification_" + str(idx) +
                              ".csv", index=False)

        train_df_to_csv = get_prediction_dict(adaboost, train, target_column, attribute_columns)
        os.mkdir(node_path + "/train") if not os.path.exists(node_path + "/train") else None
        train_df_to_csv.to_csv(node_path + "/train/classification_" + str(idx) +
                               ".csv", index=False)
        idx += 1
