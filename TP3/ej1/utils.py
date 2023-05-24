import numpy as np
import pandas as pd

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

def random_points_within_range(x_min, x_max, y_min, y_max, n, error_rate=0):
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