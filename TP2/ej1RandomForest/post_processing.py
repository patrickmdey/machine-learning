import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

def save_heatmap(df,method, partition_amount, tree_amount=None):
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)
    
    ax = sns.heatmap(df, cmap=cmap, 
                annot=True, fmt=".2%", xticklabels=["No Devuelve", "Devuelve"], yticklabels=["No Devuelve", "Devuelve"])
    
    cbar = ax.collections[0].colorbar
    tick_labels = cbar.ax.get_yticklabels()
    tick_values = cbar.get_ticks()
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_text(f"{int(tick_values[i] * 100)}%")
    cbar.ax.set_yticklabels(tick_labels)

    title = "Matriz de confusión para " + method.upper() + " con " + str(partition_amount) + " particiones"
    if tree_amount is not None:
        title += " y " + str(tree_amount) + " árboles"

    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()
    path = "out/"+method+ "/"+str(partition_amount)
    path += ("/" + str(tree_amount) + "_trees") if tree_amount is not None else ""
    plt.savefig(path + "/heatmap.png")

def confusion_row_to_percent(row):
    total = row.sum()
    return row.apply(lambda x: (x / total).round(4))

# TODO: this is not the tree precision
def compute_metrics(row, stats, confusion_matrix):
    predicted = row['predicted']
    real = row['real']
    if predicted == real:
        stats[real]["tp"] += 1
    else:
        stats[real]["fp"] += 1
    
    confusion_matrix[real][predicted] += 1

def calculate_metrics(df, confusion_matrix):
    stats_per_rating = {}
    for stars in range(0, 2):
        stats_per_rating[stars] = {"tp": 0, "tn":0, "fp": 0, "fn": 0}

    df.apply(lambda row: compute_metrics(row, stats_per_rating, confusion_matrix), axis=1)
    return stats_per_rating

def calculate_results(df, confusion_matrix=None):
    correct = 0
    for _, row in df.iterrows():
        predicted = row['predicted']
        real = row['real']
        if predicted == real:
            correct += 1
            
        if confusion_matrix is not None:
            confusion_matrix[real][predicted] += 1

    return correct

def get_metrics(method, max_nodes,partition_amount, tree_amount=None):    
    classes = range(0, 2)
    confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in classes} for real_cat in classes}

    pre_path = "post_processing/" + method + "/" + str(partition_amount) + "/"

    pre_path += (max_nodes if max_nodes != "-1" else "no_max") + "_nodes/"

    precision_per_partition = {"train": [], "test": []}

    for partition in range(partition_amount):
        # TODO: agregarle tambien a random forest el str partition amount
        post_path = ("_"+ (str(partition) + "_" + str(tree_amount) + "_trees") if method == "random_forest" 
                     else str(partition)) + ".csv"

        test_df = pd.read_csv(pre_path + "test/classification" + post_path)
        train_df = pd.read_csv(pre_path + "train/classification" + post_path)

        current_correct = calculate_results(test_df, confusion_matrix)
        precision_per_partition["test"].append(current_correct/len(test_df))

        # TODO: meterle un if para esto hacerlo solo en el caso de precision vs nodos
        current_correct = calculate_results(train_df)
        precision_per_partition["train"].append(current_correct/len(train_df))
    
    confusion_df = pd.DataFrame(confusion_matrix)
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)
    save_heatmap(confusion_df, method, partition_amount, tree_amount)

    mean_std_precision = {"train": 
                            {"mean": np.mean(precision_per_partition["train"]), 
                             "std": np.std(precision_per_partition["train"]), 
                                "max_precision": max(precision_per_partition['train'])}, 
                          "test": {"mean": np.mean(precision_per_partition["test"]), 
                                   "std": np.std(precision_per_partition["test"]), 
                                   "max_precision": max(precision_per_partition['test'])}}

    return mean_std_precision

def results_to_csv(precision, method, max_nodes, partition_amount, tree_amount=None):
    to_append = {
        "nodes": max_nodes, 
        "mean_test_precision": precision["test"]["mean"], "std_test_precision": precision["test"]["std"], "max_test_precision": precision["test"]["max_precision"],
        "mean_train_precision": precision["train"]["mean"], "std_train_precision": precision["train"]["std"], "max_train_precision": precision["train"]["max_precision"]
        }
    
    path = "out/" + method + "/" + str(partition_amount)
    if tree_amount is not None:
        path += "/" + str(tree_amount) + "_trees"
    
    path += "/precision_vs_nodes.csv"
                
    if not os.path.exists(path):
        pd.DataFrame([to_append]).to_csv(path)
    else:
        metric_df = pd.read_csv(path, usecols=["nodes", "mean_test_precision", "std_test_precision", "mean_train_precision", "std_train_precision", "max_test_precision", "max_train_precision"])
        pd.concat([metric_df, pd.DataFrame([to_append])]).to_csv(path)

def main():
    tree_amount = None
    method = sys.argv[1] if len(sys.argv) > 1 else "id3"
    if method not in ["id3", "random_forest"]:
        print("Invalid method")
        return
    
    max_nodes = sys.argv[2] if len(sys.argv) > 2 else -1

    partition_amount = int(sys.argv[3] if len(sys.argv) > 3 else 5)
    path = "out/"+method+"/"+str(partition_amount)+"/"
    os.mkdir(path) if not os.path.exists(path) else None
    
    if method == "random_forest":
        tree_amount = sys.argv[4] if len(sys.argv) > 4 else 5
        path += (str(tree_amount) + "_trees/")
        os.mkdir(path) if not os.path.exists(path) else None

    precision = get_metrics(method, max_nodes, partition_amount, tree_amount)

    os.remove(path + "/precision_vs_nodes.csv") if os.path.exists(path + "/precision_vs_nodes.csv") else None
    results_to_csv(precision, method, max_nodes, partition_amount, tree_amount)
            
if __name__ == "__main__":
    main()
