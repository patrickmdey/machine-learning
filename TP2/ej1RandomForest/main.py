import pandas as pd
import numpy as np
from preanalysis import categorize_columns

class Node:
    attr_name = ""
    attr_value = None
    has_value = False
    children = {}
    parent = None
    def __init__(self, parent, attr_name, attr_value = None, children = {}, hasValue = None):
        self.parent = parent
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.children = children
        self.has_value = hasValue


def calculate_entropy(df, target_column):
    relative_freqs = df[target_column].value_counts(normalize = True).array
    entropy = 0
    for freq in relative_freqs:
        if freq != 0:
            entropy -= freq * np.log2(freq)
    return entropy

def calculate_gains(df, columns):
    gains = {}
    h_s = calculate_entropy(df, 'Creditability')
    for column_name in columns:
        s_amount = len(df)
        gain = h_s
        h_sv = 0
        for value in df[column_name].unique():
            new_df = df[df[column_name] == value]
            sv_amount = len(new_df)
            h_sv += (sv_amount / s_amount) * calculate_entropy(new_df, 'Creditability') 
        gain -= h_sv
        gains[column_name] = gain
    
    return gains

def create_tree(df, columns):
    if len(df["Creditability"].unique()) == 1:
        
        return Node(None, "Creditability", df["Creditability"].unique()[0], {}, True)

    #TODO:empty node
    if columns == None or len(columns) == 0:
        return Node(None, "Creditability", df['Creditability'].value_counts().idxmax(), {}, True)
    
    gains = calculate_gains(df, columns)
    max_gain_attr = max(gains, key=gains.get)
    # max_gain_value = max(gains.values())

    root = Node(None, max_gain_attr, None)

    for attr_value in df[max_gain_attr].unique():
        conditional_df = df[df[max_gain_attr] == attr_value]
        conditional_df = conditional_df.loc[:, conditional_df.columns != max_gain_attr]
        child = Node(root, max_gain_attr, attr_value, {}, True)

        root.children[attr_value] = child

        #remove creditability from columns
        new_cols = [column_name for column_name in conditional_df.columns.tolist() if column_name != "Creditability"]

        new_child = create_tree(conditional_df, new_cols)
        child.children[new_child.attr_value] = new_child
    
    return root

#create a function that prints all children of the tree
def print_tree(tree):
    if len(tree.children) > 0:
        for key in tree.children.keys():
            print_tree(tree.children[key])
    else:
        print(tree.attr_value)


def tree_to_dict(tree):
    dict_tree = {}
    dict_tree[tree.attr_name] = {}
    if len(tree.children) > 0:
        for key in tree.children.keys():
            dict_tree[tree.attr_name][key] = tree_to_dict(tree.children[key])
    else:
        dict_tree[tree.attr_name] = tree.attr_value

    return dict_tree


def main():
    df = pd.read_csv("./german_credit.csv")

    # Categorize
    columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    df = categorize_columns(df, columns)


    atribute_columns = df.loc[:, df.columns != 'Creditability'].columns.tolist()

    ID3_tree = create_tree(df, atribute_columns)

    # print(ID3_tree.children.keys())

    # print_tree(ID3_tree)
        
    print(tree_to_dict(ID3_tree))

    print("a")

if __name__ == "__main__":
    main()