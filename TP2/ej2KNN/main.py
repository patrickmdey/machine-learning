import json
import sys

import pandas as pd
import numpy as np
from KNN import *


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


def main():
    csv_file = ""
    k = 3
    test_size = 0.2
    with open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)
        csv_file = config["file"]
        k = config["k"]
        test_size = config["test_size"]

    df = pd.read_csv(csv_file, sep=';')
    # Ej a)
    df_one_star = df[df['Star Rating'] == 1]
    word_count = df_one_star['wordcount'].sum()
    print("Average word count for 1 star reviews:", word_count / len(df_one_star))

    # Ej c)
    # TODO: modularize
    df = df[['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']]
    df['titleSentiment'] = df['titleSentiment'].fillna(0)
    df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] >= 3), 'titleSentiment'] = 'positive'
    df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] < 3), 'titleSentiment'] = 'negative'

    df.loc[df['titleSentiment'] == 'positive', 'titleSentiment'] = 1
    df.loc[df['titleSentiment'] == 'negative', 'titleSentiment'] = 0

    partitions = partition_dataset(df, 0.2)
    knn = KNN(k)

    with open("post_processing/classification.csv", "w") as classifier_file:
        for partition in partitions:
            test = partition
            train = pd.concat([df for df in partitions if df is not partition])
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)
            knn.fit(train[['wordcount', 'titleSentiment', 'sentimentValue']].to_numpy(),
                    train[['Star Rating']].to_numpy())

            # TODO: move this up
            test = test[['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']]
            for instance in test.to_numpy():
                attributes = instance[0:-1]
                print(attributes)
                y = instance[-1]
                print("Prediction: ", knn.predict(attributes))


if __name__ == "__main__":
    main()
