import numpy as np
import math
class Group:
    id = 0
    def __init__(self, element):
        self.elements = [element]
        self.distances_to = {}
        self.id = Group.id
        Group.id += 1
        self.centroid = element # TODO: esto solo si lo hacemos con el average, podriamos hacer pruebas con max y min
    
    def add_element(self, group):
        # print("Adding", group.id, "to", self.id)
        # print("[" + str(self.elements) + "] + [" + str(group.elements) + "]")
        self.elements = np.vstack((self.elements, group.elements))
        self.centroid = np.mean(self.elements, axis=0)
        # print("Result", self.elements)
    
    def calculate_distance_to(self, other_group):
        distance = np.linalg.norm(self.centroid - other_group.centroid)
        self.distances_to[other_group.id] = np.linalg.norm(self.centroid - other_group.centroid)
        return distance
    
    def calculate_avg_centroid(self):
        return self.centroid

class HierarchicalGroups:
    def __init__(self, k, observations):
        self.k = k
        self.observations = observations
        self.distances = {}
    
    def calculate_and_group(self, groups):
        
        return
    
    def solve(self):
        clusters = [] # TODO: maybe map?
        for idx, observation in enumerate(self.observations):
            clusters.append(Group(observation))

        distances = np.zeros((len(clusters), len(clusters))) # esta escrito asi porque es shape

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = clusters[i].calculate_distance_to(clusters[j])
                distances[i][j] = distance
                distances[j][i] = distance

        np.fill_diagonal(distances, math.inf) # TODO: check

        current_clusters = len(clusters)
        while current_clusters > self.k:
            min_distance = np.min(distances)
            min_distance_idxs = np.where(distances == min_distance)
            
            min_distance_idxs = min_distance_idxs[0]
            first_group = min(min_distance_idxs)
            second_group = max(min_distance_idxs) 

            # TODO: chequear que te viene
            
            to_remove = clusters.pop(second_group)
            clusters[first_group].add_element(to_remove)

            # TODO: borrar el segundo grupo ya que fue anexado pero poder mantener los indices
            distances = np.delete(distances, second_group, axis=0) #Axis = 0 is row
            distances = np.delete(distances, second_group, axis=1) #Axis = 1 is col


            for i, cluster in enumerate(clusters):
                distance = clusters[i].calculate_distance_to(clusters[first_group])
                distances[i][first_group] = distance
                distances[first_group][i] = distance
            
            distances[first_group][first_group] = math.inf
            current_clusters -= 1

        return clusters
    

    def create_dendogram(self):
        groups = []
        for observation in self.observations:
            groups.append(observation)

        # TODO: debe haber una mejor forma de calcular la menor distancia
        # TODO: buscar alguna forma de no calcular de nuevo cuentas ya hechas
        # TODO: tambien necesitas las distancias desde afuera
        while len(groups) > 1:
            for first_group in groups:
                for second_group in groups:
                    if first_group.id != second_group.id:
                        first_group.calculate_distance_to(second_group)