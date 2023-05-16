import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def save_heatmap(df):
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)
    
    ax = sns.heatmap(df, cmap=cmap, 
                annot=True, fmt=".2%")
    
    cbar = ax.collections[0].colorbar
    tick_labels = cbar.ax.get_yticklabels()
    tick_values = cbar.get_ticks()
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_text(f"{int(tick_values[i] * 100)}%")
    cbar.ax.set_yticklabels(tick_labels)

    title = "Matriz de confusión"
    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()
    path = "out/"
    
    plt.savefig(path + "/heatmap.png")

# TODO: remove extra zeros
def get_heatmap(df):
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)
    sns.heatmap(df, cmap=cmap, annot=True, fmt="%")
    plt.tight_layout()
    plt.savefig("out/heatmap.png")

def confusion_row_to_percent(row):
    total = row.sum()
    return row.apply(lambda x: (x / total).round(4))

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
    for stars in range(1, 6):
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


def main():
    # FIXME

    file_amount = 10
    
    confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in range(1, 6)} for real_cat in range(1, 6)}

    precision_per_partition = [0 for _ in range(file_amount)]

    for partition in range(file_amount):
        df = pd.read_csv("post_processing/knn/classification" + str(partition) + ".csv")
        current_correct = calculate_results(df, confusion_matrix)
        precision_per_partition[partition] = current_correct/len(df)
    
    confusion_df = pd.DataFrame(confusion_matrix)
    
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)

    save_heatmap(confusion_df)

    mean_std_precision = {"mean": np.mean(precision_per_partition), 
                             "std": np.std(precision_per_partition), 
                                "max_precision": max(precision_per_partition)}


    pd.DataFrame([mean_std_precision]).to_csv("out/mean_metrics.csv")
            
if __name__ == "__main__":
    main()
