import numpy as np
class Group:
    id = 0
    def __init__(self, element):
        self.elements = [element]
        self.distances_to = {}
        self.id = Group.id
        Group.id += 1
        self.centroid = element # TODO: esto solo si lo hacemos con el average, podriamos hacer pruebas con max y min
    
    def add_element(self, element):
        self.observations.append(element)
        self.centroid = np.mean(self.elements, axis=0)
    
    def calculate_distance_to(self, other_group):
        self.distances_to[other_group.id] = np.linalg.norm(self.centroid - other_group.centroid)
    
    def calculate_avg_centroid(self):
        return self.centroid
        
class HierarchialGroups:
    def __init__(self, k, observations):
        self.k = k
        self.observations = observations
        self.distances = {}
    
    def calculate_and_group(self, groups):
        
        return
    
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
                
        