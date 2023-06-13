import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hierarchical_alt import HierarchicalGroups
from kohonen_network import KohonenNetwork
from kmeans import Kmeans


# TODO: agregarle a los clusters cuantas peliculas de cada tipo tienen!!!!


def test_heriarchy():
    data = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32],
                    [0.26, 0.19], [0.08, 0.41], [0.45, 0.3]])
    her = HierarchicalGroups(3, data)
    clusters = her.solve()
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster.elements}")


def plot_elbow(variations):
    plt.clf()
    plt.plot(variations, marker='o')
    plt.xlabel("Cantidad de clusters")
    plt.ylabel("Variación total")
    plt.title("Método del codo")

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


def run_kohonen(df, params, genres):
    headers = df.columns.values.tolist()
    # group_by = df["genres"]

    # rows = rows.astype(float)

    # rows = StandardScaler().fit_transform(rows) #http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    grid_k = params['init_k']
    radius = params['init_r']

    network = KohonenNetwork(eta=params['eta'], grid_k=grid_k, radius=radius, genres=genres)

    network.solve(df.values.tolist(), genres)


    # TODO: esto se puede hacer con los genres que quedaron del kohonen
    winners_associated = np.zeros(dtype=int, shape=(grid_k, grid_k))
    winners = network.find_all_winners(df.values.tolist())

    #plot
    labels = np.empty(dtype="U256", shape=(grid_k, grid_k))
    for idx, (row, col) in enumerate(winners):
        winners_associated[row][col] += 1
        # labels[row][col] += genres[idx] + "\n"
        # labels[row][col] += group_by[idx] + "\n"

    plot_heatmap(winners_associated, "Generos", winners_associated)
    plot_heatmap(winners_associated, "Agrupación de Generos", labels)
    plot_heatmap(network.u_matrix(), "Matriz U")


if __name__ == '__main__':

    method = 'hierarchical'
    if len(sys.argv) > 1:
        method = sys.argv[1]  # kohonen, kmeans, hierarchical, analysis

    analysis_cols = ["budget", "genres", "imdb_id", "popularity", "production_companies", "production_countries",
                     "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]
    # dataset_analysis(pd.read_csv("movie_data.csv", sep=';'), analysis_cols)

    float_analysis_cols = ["budget", "popularity", "production_companies", "production_countries",
                           "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]

    df = pd.read_csv('movie_data.csv', sep=';', usecols=analysis_cols)

    df = df.loc[(df["genres"] == "Drama") | (
        df["genres"] == "Comedy") | (df["genres"] == "Action")]
    
    genres = df["genres"]
    df = df.drop(["genres"], axis=1)
    df["genres"] = genres

    # remove rows with null values
    df = df.dropna()
    df = df.drop_duplicates(subset="imdb_id", keep=False)
    df = df.drop(["imdb_id"], axis=1)

    df.loc[:, float_analysis_cols] = StandardScaler().fit_transform(df[float_analysis_cols].values)

    if method == 'kohonen':
        params = {
            'eta': 0.1,
            'init_k': 7,
            'init_r': 5
        }
        run_kohonen(df, params, genres)

    elif method == 'kmeans':
        kmeans = Kmeans(3, df.values.tolist(), genres)
        centroids = kmeans.solve()
        # print(centroids)

        variations = []

        for k in range(1, 10):
            kmeans = Kmeans(k, df.values.tolist(), genres)
            centroids, clusters = kmeans.solve()
            variations.append(kmeans.calculate_variation(clusters))

        print(variations)
        plot_elbow(variations)

    elif method == 'hierarchical':
        # test_heriarchy()
        heriarchy = HierarchicalGroups(3, df.values.tolist(), genres)
        genre_count = {genre: 0 for genre in genres}
        clusters = heriarchy.solve()
        genre_count_per_cluster = {}
        for i, cluster in enumerate(clusters):
            genre_count_per_cluster[i] = {genre: 0 for genre in genres}
            for obs in cluster.elements:
                genre_count[obs[-1]] += 1
                genre_count_per_cluster[i][obs[-1]] += 1

        print(genre_count)
        print(genre_count_per_cluster)

    exit(0)
