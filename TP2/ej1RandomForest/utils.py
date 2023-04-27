import numpy as np
import pandas as pd


def calculate_entropy(df, target_column):
    relative_freqs = df[target_column].value_counts(normalize=True).array
    entropy = 0
    for freq in relative_freqs:
        if freq != 0:
            entropy -= freq * np.log2(freq)
    return entropy


def calculate_gains(df, columns):
    gains = {}
    h_s = calculate_entropy(df, 'Creditability')
    for column_name in columns:
        s_amount = len(df)
        gain = h_s
        h_sv = 0
        for value in df[column_name].unique():
            new_df = df[df[column_name] == value]
            sv_amount = len(new_df)
            h_sv += (sv_amount / s_amount) * calculate_entropy(new_df, 'Creditability')
        gain -= h_sv
        gains[column_name] = gain

    return gains

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

    if  (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions
