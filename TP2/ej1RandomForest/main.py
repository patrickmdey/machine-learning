import json

import pandas as pd
from preanalysis import categorize_columns
from utils import *
from sklearn.model_selection import train_test_split
import numpy as np


class Node:
    attr_name = ""
    attr_value = None
    has_value = False
    children = {}
    parent = None
    conditional_df = None

    def __init__(self, parent, attr_name, attr_value=None, children=None, hasValue=None, conditional_df=None):
        if children is None:
            children = {}
        self.parent = parent
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.children = children
        self.has_value = hasValue
        self.conditional_df = conditional_df


def create_tree(df, columns, target_column, parent):
    if len(df[target_column].unique()) == 1:
        return Node(parent, target_column, df[target_column].unique()[0], {}, True, df[target_column])

    # TODO:check condition
    if columns is None or len(columns) == 0:
        return Node(parent, target_column, df[target_column].mode()[0], {}, True, df.copy())

    gains = calculate_gains(df, columns)
    max_gain_attr = max(gains, key=gains.get)
    # max_gain_value = max(gains.values())

    root = Node(parent, max_gain_attr, None)
    root.conditional_df = df.copy()

    for attr_value in df[max_gain_attr].unique():
        conditional_df = df[df[max_gain_attr] == attr_value]
        conditional_df = conditional_df.loc[:, conditional_df.columns != max_gain_attr]

        child = Node(root, max_gain_attr, attr_value, {}, True, conditional_df.copy())
        root.children[attr_value] = child

        new_cols = [column_name for column_name in conditional_df.columns.tolist() if column_name != target_column]

        new_child = create_tree(conditional_df, new_cols, target_column, parent)
        child.children[new_child.attr_name] = new_child

    return root

def rebuild_conditional_df(node, df, target_column):
    conditions = {}
    curr_node = node
    while curr_node.parent is not None:
        if curr_node.has_value:
            conditions[curr_node.attr_name] = curr_node.attr_value
        curr_node = curr_node.parent
    conditional_df = df
    for key in conditions:
        conditional_df = df[df[key == conditions[key]]]
    
    return conditional_df[target_column].mode()[0]

def classify_instance(node, instance, target_column, df):
    while node.attr_name != target_column:
        if not node.has_value:
            children = node.children
            #TODO: check
            if len(children) == 1 and next(iter(children)) == target_column:
                key = next(iter(node.children))
                return node.children[key].attr_value
            
            instance_value = instance[node.attr_name]
            if instance_value not in children:
                # print("ERROR: instance value not in children")
                # TODO: remove [] if it works
                return node.conditional_df[target_column].mode()[0]
                #return rebuild_conditional_df(node, df, target_column)
                
            node = children[instance_value]
        else:
            key = next(iter(node.children))
            node = node.children[key]

    return node.attr_value


# TODO: if it doesnt work, save conditional df to nodes
def prune_tree(root, max_nodes, df, target_column):
    amount = 0
    nodes = [root]
    prev_nodes = nodes
    while amount < max_nodes:
        children = []
        for node in nodes:
            if not node.has_value and node.attr_name != target_column:
                amount += 1
            for key, child in node.children.items():
                children.append(child)

        prev_nodes = nodes
        nodes = children
    
    for node in prev_nodes:
        if node.attr_name != target_column:
            # TODO: check
            if not node.has_value and node.parent is not None:
                node = node.parent

            most_common_class = node.conditional_df[target_column].mode()[0]
            #most_common_class = rebuild_conditional_df(node, df, target_column)
            node.children = {target_column: Node(node, target_column, most_common_class)}

def count_nodes(root, target_column):
    amount = 0
    nodes = [root]
    while len(nodes) > 0:
        children = []
        for node in nodes:
            if not node.has_value and node.attr_name != target_column:
                amount += 1
            for key, child in node.children.items():
                children.append(child)
        nodes = children
    
    return amount

def random_forest(df, target_column, test_percentage, examples_per_tree, tree_amount):
    train, test = train_test_split(df, test_size=test_percentage)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    trees = []
    for _ in range(0, tree_amount):
        tree = create_tree(train.sample(frac=1, replace=True), None, target_column, None)
        prune_tree(tree, 80, df, target_column)
        trees.append(tree)
        
    # TODO: change to apply
    most_voted = []
    for _, instance in test.iterrows():
        predictions = []
        for tree in trees:
            predictions.append(classify_instance(tree, instance, target_column, df))
        
        if instance[target_column] == 0:
            print("Predicciones " + str(predictions))
            most_voted.append(np.bincount(predictions).argmax())
            print("REAL: " + str(instance[target_column]))
            print("PREDICHOOOO: " + str(most_voted[-1]))


def id3(df, columns, target_column):
    partitions = partition_dataset(df, 0.1)
    idx = 0
    for partition in partitions:
        test = partition
        train = pd.concat([df for df in partitions if df is not partition])
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        root = create_tree(train, columns, target_column, None)

        # print("ANTES")
        # print(count_nodes(root, target_column))
        # # prune_tree(root, max_detph, df, target_column)
        # print("Despues")
        # print(count_nodes(root, target_column))

        df_to_csv = pd.DataFrame(columns=["predicted", "real"])
        
        for _, instance in test.iterrows():
            prediction = classify_instance(root, instance, target_column, df)
            real = instance[target_column]
            df_to_csv = pd.concat([df_to_csv, pd.DataFrame([{"predicted": prediction, "real": real}])], ignore_index=True)

        # TODO: add extension for node precision 
        df_to_csv.to_csv("post_processing/classification" + str(idx) +".csv", index=False)
        idx += 1

def main():
    csv_file = ""
    target_column = ""
    max_depth = 10
    do_forest = False
    with open("config.json") as config_file:#sys.argv[1], 'r') as config_file: #TODO: remove hardcode
        config = json.load(config_file)
        csv_file = config["file"]
        target_column = config["target"]
        max_depth = config["max_depth"]
        do_forest = config["do_forest"]

    df = pd.read_csv(csv_file)

    # Categorize
    columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    df = categorize_columns(df, columns)

    attribute_columns = df.loc[:, df.columns != target_column].columns.tolist()
    if do_forest:
        print("Random Forest")
        random_forest(df, target_column, 0.2, 80, 10)
    else:
        print("ID3")
        id3(df, attribute_columns, target_column)

if __name__ == "__main__":
    main()
