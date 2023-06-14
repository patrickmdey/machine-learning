import numpy as np
import math


class Group:
    def __init__(self, element, genres, use_centroid=True):
        self.elements = [element]
        self.genres = {genre: 0 for genre in genres}
        self.genres[element[-1]] = 1
        self.centroid = np.array(element[:-1], dtype=float) if use_centroid else None

    def add_element(self, group):
        # TODO: check!!!!!!!
        for element in group.elements:
            self.elements.append(element)
            for genre in self.genres:
                self.genres[genre] += group.genres[genre]
        # self.elements.extend(group.elements)
        # TODO: check!!!!
        
        masked_elements = [subarray[:-1] for subarray in self.elements]
        # print(masked_elements)
        self.centroid = np.mean(masked_elements, axis=0)
        #print(self.centroid)

    def calculate_distance_to(self, other_group):
        distance = np.linalg.norm(self.centroid - other_group.centroid)
        return distance

    def calculate_avg_centroid(self):
        return self.centroid


class HierarchicalGroups:
    def __init__(self, k, observations, genres):
        self.k = k
        self.observations = observations
        self.distances = {}
        self.genres = genres

    def predict_genre(self, observation, clusters):
        min_dist = math.inf
        cluster_idx = None

        obs_array = np.array(observation)

        min_dist = math.inf
        min_cluster = None
        for idx, cluster in enumerate(clusters):
            # print("Observation: " + str(observation), "Centroid: " + str(centroid))
            # TODO: check y ademas capaz podriamos tener una funcion que saque la distancia por afuera y listo
            distance = np.linalg.norm(cluster.centroid - observation[:-1])
            if distance < min_dist:
                min_dist = distance
                min_cluster = cluster
            centroid = np.array(centroid)

        return max(min_cluster.genres, key=min_cluster.genres.get)

    def solve(self):
        clusters = []
        for idx, observation in enumerate(self.observations):
            clusters.append(Group(observation, self.genres))

        # esta escrito asi porque es shape
        distances = np.zeros((len(clusters), len(clusters)))

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = clusters[i].calculate_distance_to(clusters[j])
                distances[i][j] = distance
                distances[j][i] = distance

        np.fill_diagonal(distances, math.inf)

        current_clusters = len(clusters)
        while current_clusters > self.k:
            if current_clusters % 100 == 0:
                print(current_clusters)
            min_distance = np.min(distances)
            min_distance_idxs = np.where(distances == min_distance)

            min_distance_idxs = min_distance_idxs[0]
            first_group = min(min_distance_idxs)
            second_group = max(min_distance_idxs)

            to_remove = clusters.pop(second_group)
            clusters[first_group].add_element(to_remove)

            distances = np.delete(distances, second_group,
                                  axis=0)  # Axis = 0 is row
            distances = np.delete(distances, second_group,
                                  axis=1)  # Axis = 1 is col

            for i, cluster in enumerate(clusters):
                distance = clusters[i].calculate_distance_to(
                    clusters[first_group])
                distances[i][first_group] = distance
                distances[first_group][i] = distance

            distances[first_group][first_group] = math.inf
            current_clusters -= 1

        return clusters
