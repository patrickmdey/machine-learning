import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# TODO: remove extra zeros
def get_heatmap(df,method):
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)
    sns.heatmap(df, cmap=cmap, annot=True, fmt="%")
    plt.tight_layout()
    plt.savefig("out/"+method+"/heatmap.png")

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

def get_metrics(method, max_nodes,partition_amount, tree_amount=None):    
    classes = range(0, 2)
    confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in classes} for real_cat in classes}

    precision_per_class = [{real_cat: 0 for real_cat in classes}]

    pre_path = "post_processing/"+method+"/"
    pre_path += (max_nodes if max_nodes != "-1" else "no_max") + "_nodes/"


    for partition in range(partition_amount):
        post_path = ("_"+ (str(tree_amount) + "_trees") if method == "random_forest" else str(partition)) + ".csv"

        df = pd.read_csv(pre_path + "classification" + post_path)

        metrics = calculate_metrics(df, confusion_matrix)
        for key in metrics:
            precision_per_class[-1][key] = metrics[key]["tp"] / (metrics[key]["tp"] + metrics[key]["fp"])
        
        precision_per_class.append({real_cat: 0 for real_cat in classes})
    
    confusion_df = pd.DataFrame(confusion_matrix)
    
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)

    get_heatmap(confusion_df, method)

    metrics_per_class = {"mean": {cat: 0 for cat in classes}, "std": {cat: 0 for cat in classes}}
    for real_cat in classes:
        metrics_per_class["mean"][real_cat] = np.mean([precision_per_class[i][real_cat] for i in range(len(precision_per_class))])
        metrics_per_class["std"][real_cat] = np.std([precision_per_class[i][real_cat] for i in range(len(precision_per_class))])

    pd.DataFrame(metrics_per_class["mean"], index=[0]).to_csv("out/"+method+"/mean_metrics.csv")
    pd.DataFrame(metrics_per_class["std"], index=[0]).to_csv("out/"+method+"/std_metrics.csv")


def main():
    tree_amount = None
    method = sys.argv[1] if len(sys.argv) > 1 else "id3"
    if method not in ["id3", "random_forest"]:
        print("Invalid method")
        return
    
    max_nodes = sys.argv[2] if len(sys.argv) > 2 else -1

    if method == "id3":
        partition_amount = int(sys.argv[3] if len(sys.argv) > 3 else 5)
    else:
        partition_amount = 1
        tree_amount = sys.argv[3] if len(sys.argv) > 3 else 5

    get_metrics(method, max_nodes, partition_amount, tree_amount)
        
            
if __name__ == "__main__":
    main()
