import csv
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from kohonen_network import KohonenNetwork
from kmeans import Kmeans

def plot_heatmap(matrix, title, labels=None):
    # Create a dataset
    df = pd.DataFrame(matrix)

    # Default heatmap
    p = sns.heatmap(df, annot=labels, fmt='')
    p.set_title(title)

    plt.show()

def run_kohonen(df, params):
    with open('movie_data.csv') as csvfile:

        rows = np.array(df.values.tolist())
        headers = df.columns.values.tolist()

        rows = rows.astype(float)

        csvfile.close()

        rows = StandardScaler().fit_transform(rows) #http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        grid_k = params['grid_k']
        radius = params['radius']

        network = KohonenNetwork(eta=params['eta'], grid_k=grid_k, radius=radius)

        network.solve(rows)

        winners_associated = np.zeros(dtype=int, shape=(grid_k, grid_k))
        winners = network.find_all_winners(rows)
        labels = np.empty(dtype="U256", shape=(grid_k, grid_k))
        # for idx, (row, col) in enumerate(winners):
        #     winners_associated[row][col] += 1
        #     labels[row][col] += countries[idx] + "\n"

        plot_heatmap(winners_associated, "Generos", winners_associated)
        plot_heatmap(winners_associated, "Agrupación de Generos", labels)
        plot_heatmap(network.u_matrix(), "Matriz U")


if __name__ == '__main__':
    params = {
        'eta': 0.1,
        'grid_k': 10,
        'radius': 5
    }

    # useful_cols = ["genres", "popularity", "revenue", "runtime", "vote_average", "vote_count"]
    useful_cols = ["popularity", "revenue"]

    df = pd.read_csv('movie_data.csv', sep=';', usecols=useful_cols)

    #remove rows with null values
    df = df.dropna()

    # normalize df
    df = (df - df.mean()) / (df.max() - df.min())

    # print(df.head())

    kmeans = Kmeans(3, df.values.tolist())

    centroids = kmeans.find_centroids()

    print(centroids)

    # plot scatter
    # plt.scatter(df['popularity'], df['revenue'], c='black')
    # plt.scatter(centroids[0][0], centroids[0][1], c='red')
    # plt.scatter(centroids[1][0], centroids[1][1], c='blue')
    # plt.scatter(centroids[2][0], centroids[2][1], c='green')
    # plt.xlabel('popularity')
    # plt.ylabel('revenue')
    # plt.show()


