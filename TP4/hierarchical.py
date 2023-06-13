import numpy as np
from scipy.spatial.distance import cdist


class Cluster:
    def __init__(self, data_points):
        self.data_points = np.array(data_points)

    def add_point(self, data_point):
        np.append(data_point)


# def distance_matrix(data, distance='euclidean'):
#     matrix = np.zeros(shape=(len(data), len(data)))
#     for i in range(len(data)):
#         for j in range(i+1, len(data)):
#             if distance == 'manhattan':
#                 dist = np.sum(np.abs(data[i] - data[j]))
#                 matrix[i][j] = dist
#                 matrix[j][i] = dist
#             if distance == 'euclidean':
#                 dist = np.linalg.norm(data[i] - data[j])
#                 matrix[i][j] = dist
#                 matrix[j][i] = dist
#
#     return matrix

def hierarchical_clustering(data, num_clusters, linkage='single', distance_method='euclidean'):
    # Create a cluster for each data point
    clusters = [Cluster([x]) for x in data]

    # Linkage method
    if linkage == 'single':
        get_distance = np.min
    elif linkage == 'complete':
        get_distance = np.max
    elif linkage == 'average':
        get_distance = np.mean

    # Merge clusters until desired number of clusters is reached
    while len(clusters) > num_clusters:
        min_distance = np.inf
        merge_indices = (0, 0)

        # Find the closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                distances = cdist(cluster1.data_points, cluster2.data_points, metric=distance_method)
                distance = get_distance(distances)
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # Merge the closest pair of clusters
        cluster1 = clusters[merge_indices[0]]
        cluster2 = clusters[merge_indices[1]]
        merged_cluster = Cluster(np.append(cluster1.data_points, cluster2.data_points, axis=0))

        clusters.append(merged_cluster)

        # Remove the merged clusters
        clusters.remove(cluster1)
        clusters.remove(cluster2)

    # Return the final clusters
    return clusters


# Example usage
data = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32], [0.26, 0.19], [0.08, 0.41], [0.45, 0.3]])
clusters = hierarchical_clustering(data, num_clusters=3, linkage='complete')
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster.data_points}")
