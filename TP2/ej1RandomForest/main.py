import json
import sys

import pandas as pd
import numpy as np
from preanalysis import categorize_columns
from utils import *


class Node:
    attr_name = ""
    attr_value = None
    has_value = False
    children = {}
    parent = None

    def __init__(self, parent, attr_name, attr_value=None, children=None, hasValue=None):
        if children is None:
            children = {}
        self.parent = parent
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.children = children
        self.has_value = hasValue


def create_tree(df, columns, target_column, nodes, max_depth=0):
    if len(df[target_column].unique()) == 1:
        return Node(None, target_column, df[target_column].unique()[0], {}, True)

    # TODO:check condition
    if columns is None or len(columns) == 0 or nodes == max_depth:
        return Node(None, target_column, df[target_column].value_counts().idxmax(), {}, True)

    nodes += 1

    gains = calculate_gains(df, columns)
    max_gain_attr = max(gains, key=gains.get)
    # max_gain_value = max(gains.values())

    root = Node(None, max_gain_attr, None)

    for attr_value in df[max_gain_attr].unique():
        conditional_df = df[df[max_gain_attr] == attr_value]
        conditional_df = conditional_df.loc[:, conditional_df.columns != max_gain_attr]

        child = Node(root, max_gain_attr, attr_value, {}, True)
        root.children[attr_value] = child

        new_cols = [column_name for column_name in conditional_df.columns.tolist() if column_name != target_column]

        new_child = create_tree(conditional_df, new_cols, target_column, nodes, max_depth)
        child.children[new_child.attr_value] = new_child

    return root


# Print the tree
def print_tree(node, to_print):
    to_print[node.attr_name] = {}
    for key in node.children.keys():
        if len(node.children[key].children) > 0:
            print_tree(node.children[key], to_print[node.attr_name])
        else:
            to_print[node.attr_name][key] = node.children[key].attr_value

    return to_print


def tree_to_dict(tree):
    dict_tree = {}
    dict_tree[tree.attr_name] = {}
    if len(tree.children) > 0:
        for key in tree.children.keys():
            dict_tree[tree.attr_name][key] = tree_to_dict(tree.children[key])
    else:
        dict_tree[tree.attr_name] = tree.attr_value

    return dict_tree

def classify_instance(node, instance, target_column):
    while node.attr_name != target_column:
        if not node.has_value:
            children = node.children
            instance_value = instance[node.attr_name]
            node = children[instance_value]
        else:
            key = next(iter(node.children))
            node = node.children[key]

    return node.attr_value


def main():
    csv_file = ""
    target_column = ""
    max_depth = 0
    with open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)
        csv_file = config["file"]
        target_column = config["target"]
        max_depth = config["max_depth"]

    df = pd.read_csv(csv_file)

    # Categorize
    columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    df = categorize_columns(df, columns)

    attribute_columns = df.loc[:, df.columns != target_column].columns.tolist()

    root = create_tree(df, attribute_columns, target_column, 0, max_depth)

    print(classify_instance(root, df.iloc[47], target_column))
    print("a")


if __name__ == "__main__":
    main()
