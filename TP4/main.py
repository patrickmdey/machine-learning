import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hierarchical_alt import HierarchicalGroups
from kohonen_network import KohonenNetwork
from kmeans import Kmeans

# TODO: agregarle a los clusters cuantas peliculas de cada tipo tienen!!!!

def run_single_kmeans(train, test, cluster_amount, genres, confusion_matrix):
    print("Running single kmeans with " + str(cluster_amount) + " clusters")
    kmeans = Kmeans(cluster_amount, train.values.tolist(), genres)
    centroids, clusters = kmeans.solve()
    
    
    tot = 0
    for i, cluster in enumerate(kmeans.get_amount_of_genres_per_cluster()):
        total = sum(cluster.values())
        result = {key: str(value*100 / total) + '%' for key,
                  value in cluster.items()}
        print(f"{i}[{total}]: {result}")
        tot += sum(cluster.values())
    print(tot)

    correct = 0
    for idx, instance in test.iterrows():
        genre = kmeans.predict_genre(instance, centroids)
        confusion_matrix[instance[-1]][genre] += 1
        correct += 1 if genre == instance[-1] else 0
    
    return correct


def test_heriarchy():
    data = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32],
                    [0.26, 0.19], [0.08, 0.41], [0.45, 0.3]])
    her = HierarchicalGroups(3, data)
    clusters = her.solve()
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster.elements}")


def plot_elbow(variations):
    plt.clf()
    x = np.arange(1, len(variations)+1)
    y = np.array(variations)
    plt.plot(x, y, marker='o')
    # plt.plot(variations, marker='o')
    plt.xlabel("Cantidad de clusters")
    plt.ylabel("Variación total")
    plt.title("Método del codo")

    if not os.path.exists("./out/kmeans"):
        os.makedirs("./out/kmeans")
    plt.savefig("./out/kmeans/elbow.png")


def plot_heatmap(matrix, title, k, r, labels=None):
    plt.clf()
    # Create a dataset
    df = pd.DataFrame(matrix)

    # Default heatmap
    p = sns.heatmap(df, annot=labels, fmt='')
    p.set_title(title)

    save_path = "./out/kohonen/" + str(k) + "_" + str(r) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path+title+".png")

    # plt.show()


def run_kohonen(train, test, params, genres, confusion_matrix):
    headers = train.columns.values.tolist()

    grid_k = params['init_k']
    radius = params['init_r']

    network = KohonenNetwork(
        eta=params['eta'], grid_k=grid_k, radius=radius, genres=genres)

    network.solve(train.values.tolist(), genres)

    winners_associated = np.zeros(dtype=int, shape=(grid_k, grid_k))
    winners = network.find_all_winners(train.values.tolist())
    #print("ammount: "+ str(len(winners)))

    correct = 0

    for idx, instance in test.iterrows():
        winner, genre = network.predict_genre(instance)
        confusion_matrix[instance[-1]][genre] += 1
        correct += 1 if genre == instance[-1] else 0
    

    labels = np.empty(dtype="U256", shape=(grid_k, grid_k))
    i = 0
    for (row, col) in winners:
        #print(i)
        i+=1
        winners_associated[row][col] += 1
        # labels[row][col] += genres[idx] + "\n"
        # labels[row][col] += group_by[idx] + "\n"
    
    #print(np.array(winners_associated).sum())
    #print(winners_associated[0][0])

    plot_heatmap(winners_associated, "Activaciones de red Kohonen k=" +
                 str(grid_k), grid_k, radius, winners_associated)
    plot_heatmap(winners_associated, "Agrupación de Generos",
                 grid_k, radius, labels)
    plot_heatmap(network.u_matrix(), "Matriz U k="+str(grid_k), grid_k, radius)

    return correct


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
        partitions[-2] = pd.concat([partitions[-2],
                                   partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions


if __name__ == '__main__':

    method = 'hierarchical'
    if len(sys.argv) > 1:
        method = sys.argv[1]  # kohonen, kmeans, hierarchical, analysis

    analysis_cols = ["budget", "genres", "imdb_id", "revenue", "vote_count"]

    float_analysis_cols = ["budget", "revenue", "vote_count"]
    
    # analysis_cols = ["budget", "genres", "imdb_id", "popularity", "production_companies", "production_countries",
    #                 "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]
    # float_analysis_cols = ["budget", "popularity", "production_companies", "production_countries",
    #                       "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]

    df = pd.read_csv('movie_data.csv', sep=';', usecols=analysis_cols)

    df = df.loc[(df["genres"] == "Drama") | (
        df["genres"] == "Comedy") | (df["genres"] == "Action")]

    genres = df["genres"]
    df = df.drop(["genres"], axis=1)
    df["genres"] = genres

    df = df.dropna()
    df = df.drop_duplicates(subset="imdb_id", keep=False)
    df = df.drop(["imdb_id"], axis=1)

    df.loc[:, float_analysis_cols] = StandardScaler().fit_transform(df[float_analysis_cols].values)

    partitions = partition_dataset(df, 0.2)

    #confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in genres} for real_cat in genres}
    # print(confusion_matrix)
    #correct = 0

    #for partition in partitions:
        #test = partition
        #train = pd.concat([df for df in partitions if df is not partition])

        #train.reset_index(drop=True, inplace=True)
        #test.reset_index(drop=True, inplace=True)

    if method == 'kohonen':
        print(len(df))
        params = {
            'eta': 0.1,
            'init_k': 6,
            'init_r': 6
        }
        # correct += run_kohonen(train, test, params, genres, confusion_matrix)

    elif method == 'kmeans':
        # correct += run_single_kmeans(train, test, 4, genres, confusion_matrix)

        print(len(df))
        variations = []

        for k in range(1, 10):
            min_var = math.inf
            for _ in range(5):
                kmeans = Kmeans(k, df.values.tolist(), genres)
                centroids, clusters = kmeans.solve()
                variation = kmeans.calculate_variation(clusters)
                if variation < min_var:
                    min_var = variation
            variations.append(min_var)   
            #variations.append(kmeans.calculate_variation(clusters))

        print(variations)
        plot_elbow(variations)

        #run_single_kmeans(df, 4, genres)

    elif method == 'hierarchical':
        # test_heriarchy()
        print(len(df))
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
    
    #precision = correct / len(df)
    #print("Precision: " + str(precision))


    #plot confusion matrix
    # plt.clf()
    # sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, fmt='g')
    # plt.title('Matriz de confusión - ' + method)
    # out_path = "out/"+method+"_confusion_matrix.png"
    # plt.savefig(out_path)
