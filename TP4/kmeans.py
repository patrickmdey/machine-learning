import math
import numpy as np
import random


class Kmeans:
    def __init__(self, k, observations):
        self.k = k
        self.observations = observations

    def check_centroids(self, centroids, new_centroids):
        return np.array_equal(centroids, new_centroids)

    def get_centroids(self, clusters):
        new_centroids = np.zeros(shape=(self.k, len(self.observations[0])))
        for cluster in clusters:
            new_centroids[cluster] = np.mean(clusters[cluster], axis=0)
        return new_centroids

    def fill_clusters(self, centroids):
        clusters = {cat: [] for cat in np.arange(self.k)}
        for observation in self.observations:
            min_dist = math.inf
            cluster_idx = None

            obs_array = np.array(observation)

            for idx, centroid in enumerate(centroids):
                # print("Observation: " + str(observation), "Centroid: " + str(centroid))
                centroid = np.array(centroid)
                dist = np.linalg.norm(obs_array - centroid)

                if dist < min_dist:
                    min_dist = dist
                    cluster_idx = idx
            clusters[cluster_idx].append(observation)
        return clusters

    # TODO: capaz aca pasarle el k para reutilizar las pruebas
    def find_centroids(self):
        # TODO: se supone que observations es un df? Aca elegimos de forma random
        centroids = random.sample(self.observations, self.k)
        clusters = self.fill_clusters(centroids)

        new_centroids = self.get_centroids(clusters)

        while not np.array_equal(centroids, new_centroids):
            centroids = new_centroids
            self.fill_clusters(centroids)
            new_centroids = self.get_centroids(clusters)

        return new_centroids, clusters

    def calculate_variation(self, clusters):
        total_variation = 0
        for cluster in clusters.values():
            cluster_variation = 0
            for i, obs_i in enumerate(cluster):
                for j, obs_j in enumerate(cluster):
                    cluster_variation += np.sum((np.array(obs_i) - np.array(obs_j))**2)
            
            total_variation += cluster_variation / len(cluster)
        return total_variation
