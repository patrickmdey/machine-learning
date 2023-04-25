import pandas as pd
import numpy as np
from KNN import *

df = pd.read_csv("./reviews_sentiment.csv", sep=';')
# Ej a)
df_one_star = df[df['Star Rating'] == 1]
word_count = df_one_star['wordcount'].sum()
print("df len:", len(df))
print("Total words:", word_count)
print("Average word count for 1 star reviews:", word_count / len(df_one_star))


# Ej b)
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


# Ej c)
df = df[['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']]
df['titleSentiment'] = df['titleSentiment'].fillna(0)
df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] >= 3), 'titleSentiment'] = 'positive'
df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] < 3), 'titleSentiment'] = 'negative'

df.loc[df['titleSentiment'] == 'positive', 'titleSentiment'] = 1
df.loc[df['titleSentiment'] == 'negative', 'titleSentiment'] = 0

# Aca habria que usar las particiones pero era para probar
knn = KNN(3)
knn.fit(df[['wordcount', 'titleSentiment', 'sentimentValue']].to_numpy(),
        df[['Star Rating']].to_numpy())
instance = df.iloc[0].to_numpy()
X = instance[0:-1]
y = instance[-1]
print("Prediction: ", knn.predict(X))
