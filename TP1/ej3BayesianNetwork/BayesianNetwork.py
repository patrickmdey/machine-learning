import pandas as pd
import numpy as np

class Node:
    def __init__(self, name, parents=[], values=[], cpt={}):
        self.name = name
        self.parents = parents
        self.values = values
        self.num_values = len(values)
        self.cpt = cpt

class BayesianNetwork:
    def __init__(self, nodes=[]):
        self.nodes = nodes

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        
        return None

    def get_node_index(self, name):
        for i, node in enumerate(self.nodes):
            if node.name == name:
                return i
        
        return None
    
    def get_parents_values(self, node, sample):
        values = []
        for parent in node.parents:
            parent_index = self.get_node_index(parent.name)
            parent_value = sample[parent_index]
            values.append(parent_value)
        
        return tuple(values)

  
    def train(self, data):
        counts = data.groupby([node.name for node in self.nodes]).size().reset_index(name='count')

        for node in self.nodes:
            if not node.parents:
                node.cpt = counts.groupby([node.name])[['count']].sum()
            else:
                node_vars = node.parents + [node.name]
                node_counts = counts.groupby(node_vars)[['count']].sum()
                parent_counts = counts.groupby(node.parents)[['count']].sum()
                node.cpt = node_counts / parent_counts

            node.cpt = node.cpt.to_dict()

    def query(self, query_vars, evidence):
        # Initialize probabilities of all values to 1
        probs = [1.0] * len(self.get_node(query_vars[0]).values)

        for i, value in enumerate(self.get_node(query_vars[0]).values):
            # Create a sample with the given evidence and query value
            sample = [None] * len(self.nodes)
            sample[self.get_node_index(query_vars[0])] = value
            for var, val in evidence.items():
                sample[self.get_node_index(var)] = val

            # Calculate the joint probability of the sample
            joint_prob = 1.0
            for node in self.nodes:
                node_prob = node.cpt['count'][node.values.index(sample[self.get_node_index(node.name)])]
                if node.parents:
                    parent_values = self.get_parents_values(node, sample)
                    parent_prob = node.cpt[parent_values]
                    node_prob = parent_prob[node.values.index(sample[self.get_node_index(node.name)])]
                joint_prob *= node_prob

            # Calculate the conditional probability of the query variable
            query_prob = joint_prob / sum([joint_prob for j, _ in enumerate(self.get_node(query_vars[0]).values) if j != i])
            probs[i] = query_prob

        return probs

if __name__ == '__main__':
    df = pd.read_csv('binary.csv')
 
    nodes = [
        Node('rank', values=[1, 2, 3, 4]),
        Node('gre', parents=['rank'], values=[0, 1]),
        Node('gpa', parents=['rank'], values=[0, 1]),
        Node('admit', parents=['rank', 'gre', 'gpa'], values=[0, 1])
    ]

    bn = BayesianNetwork(nodes)
    bn.train(df)

    query_vars = ['admit']
    evidence = {'rank': 3, 'gre': 1, 'gpa': 1}

    # Faltaria implementar esto
    probs = bn.query(query_vars, evidence)
    
    # Print the result
    print(f"P(Admit=1 | Rank=3, GRE=1, GPA=1) = {probs[1]:.4f}")
