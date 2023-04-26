import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: PASAR A % LA MATRIZ DE CONFUSION
def get_heatmap(df):
    plt.clf()
    sns.heatmap(df, annot=True, fmt='.0f')
    plt.tight_layout()
    plt.savefig("out/heatmap.png")


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


def main():
    file_amount = 5
    confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in range(1, 6)} for real_cat in range(1, 6)}

    precision_per_class = [{real_cat: 0 for real_cat in range(1, 6)}] #* file_amount
    
    for partition in range(file_amount):
        df = pd.read_csv("post_processing/classification" + str(partition) + ".csv")
        # print("Partition: " + str(partition))
        # print(df.head())
        metrics = calculate_metrics(df, confusion_matrix)
        print(metrics[1]["tp"])
        for key in metrics:
            precision_per_class[-1][key] = metrics[key]["tp"] / (metrics[key]["tp"] + metrics[key]["fp"])
        
        precision_per_class.append({real_cat: 0 for real_cat in range(1, 6)})
    
    print(precision_per_class)
    confusion_df = pd.DataFrame(confusion_matrix)
    get_heatmap(confusion_df)

    metrics_per_class = {"mean": {cat: 0 for cat in range(1, 6)}, "std": {cat: 0 for cat in range(1, 6)}}
    for real_cat in range(1, 6):
        metrics_per_class["mean"][real_cat] = np.mean([precision_per_class[i][real_cat] for i in range(len(precision_per_class))])
        metrics_per_class["std"][real_cat] = np.std([precision_per_class[i][real_cat] for i in range(len(precision_per_class))])
        
    
    # print(metrics_per_class["mean"]) = {1: 0.9166666666666666, 2: 0.75, 3: 0.9333333333333333, 4: 1.0, 5: 0.8}
        
    pd.DataFrame(metrics_per_class["mean"], index=[0]).to_csv("out/mean_metrics.csv")
    pd.DataFrame(metrics_per_class["std"], index=[0]).to_csv("out/std_metrics.csv")
        

            
if __name__ == "__main__":
    main()
