import math
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hierarchical_alt import HierarchicalGroups
from utils import boxplot_column
from kohonen_network import KohonenNetwork
from kmeans import Kmeans

def test_heriarchy():
    data = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45, 0.3]])
    her = HierarchicalGroups(3, data)
    clusters = her.solve()
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster.elements}")


def plot_elbow(variations):
    plt.clf()
    plt.plot(variations, marker='o')
    plt.xlabel("Cantidad de clusters")
    plt.ylabel("Variación total")

    if not os.path.exists("./out/kmeans"):
        os.makedirs("./out/kmeans")
    plt.savefig("./out/kmeans/elbow.png")


def plot_heatmap(matrix, title, labels=None):
    plt.clf()
    # Create a dataset
    df = pd.DataFrame(matrix)

    # Default heatmap
    p = sns.heatmap(df, annot=labels, fmt='')
    p.set_title(title)

    plt.show()


def dataset_analysis(df, useful_cols):
    print("Amount of duplicated rows:", df.duplicated().sum())
    print("Amount of null values:", df.isnull().sum().sum())
    print("Amount of rows:", df.shape[0])
    print("Amount of columns:", df.shape[1])
    # print("Duplicated rows:")
    # print(df[df.duplicated()]["imdb_id"])

    for col in useful_cols:
        boxplot_column(df, col)

    # remove rows with null values
    df = df.dropna()
    df = df.drop_duplicates(subset=useful_cols, keep=False)

    years = df["release_date"].str.split("-", n=2, expand=True)[0]
    df["release_date"] = years

    trimed_df = df[useful_cols]

    # df.drop(["genres", "imdb_id", "original_title", "overview", "production_companies"], axis=1)
    # print(df["original_title"].head())

    # get standard deviation per column
    print(df.std)


def run_kohonen(df, params):
    rows = np.array(df.values.tolist())
    headers = df.columns.values.tolist()
    # group_by = df["genres"]

    # rows = rows.astype(float)

    # rows = StandardScaler().fit_transform(rows) #http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    grid_k = params['init_k']
    radius = params['init_r']

    network = KohonenNetwork(eta=params['eta'], grid_k=grid_k, radius=radius)

    network.solve(rows)

    winners_associated = np.zeros(dtype=int, shape=(grid_k, grid_k))
    winners = network.find_all_winners(rows)
    labels = np.empty(dtype="U256", shape=(grid_k, grid_k))
    for idx, (row, col) in enumerate(winners):
        winners_associated[row][col] += 1
        # labels[row][col] += group_by[idx] + "\n"

    plot_heatmap(winners_associated, "Generos", winners_associated)
    plot_heatmap(winners_associated, "Agrupación de Generos", labels)
    plot_heatmap(network.u_matrix(), "Matriz U")


if __name__ == '__main__':

    method = 'hierarchical'
    analysis_cols = ["budget", "popularity", "production_companies", "production_countries",
                   "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]

    # TODO: chequear exactamente que variables vamos a usar para comparar
    # useful_cols = analysis_cols + ["release_date"]

    # dataset_analysis(pd.read_csv("movie_data.csv", sep=';'), analysis_cols)


    df = pd.read_csv('movie_data.csv', sep=';', usecols=analysis_cols)

    # years = df["release_date"].str.split("-", n=2, expand=True)[0]
    # print(years)
    # df["release_date"] = years

    # remove rows with null values
    df = df.dropna()
    df = df.drop_duplicates(subset=analysis_cols, keep=False)

    # # normalize df
    # TODO: esto lo hicimos con un scaler, adaptarlo
    df = (df - df.mean()) / (df.max() - df.min())

    if method == 'kohonen':
        params = {
            'eta': 0.1,
            'init_k': 7,
            'init_r': 5
        }
        # run_kohonen(df, params)
        run_kohonen(df, params)

    elif method == 'kmeans':
        kmeans = Kmeans(3, df.values.tolist())
        centroids = kmeans.find_centroids()
        # print(centroids)

        variations = []

        for k in range(1, 10):
            kmeans = Kmeans(k, df.values.tolist())
            centroids, clusters = kmeans.find_centroids()
            # print(clusters)
            # print("K:", k, "SSE:", kmeans.calculate_variation(centroids))
            variations.append(kmeans.calculate_variation(clusters))

        print(variations)
        plot_elbow(variations)

    elif method == 'hierarchical':
        test_heriarchy()
        heriarchy = HierarchicalGroups(3, np.array(df.values.tolist()))
        clusters = heriarchy.solve()
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i + 1}: {cluster.elements}")


    exit(0)
