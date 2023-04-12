import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_categories():
    with open ("post_processing/classification.csv", "r") as c_file:
        return c_file.readline().split(",")[1:-1]        

def get_heatmap(df):
    plt.clf()
    sns.heatmap(df, annot=True, fmt='.0f')
    plt.tight_layout()
    plt.savefig("post_processing/heatmap.png")


# prediction,Nacional,Salud,Economia,Destacadas,Ciencia y Tecnologia,Deportes,Internacional,Entretenimiento,real
def calculate_metrics(categories):
    confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
    with open ("post_processing/classification.csv", "r") as c_file:
        c_file.readline()
        # categories = c_file.readline().split(",")[1:-1]
        stats_per_class = [{real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}]
        # confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
        #idx = 0
        for line in c_file:
            if "prediction" in line: #new train-test partition
                #idx += 1
                #confusion_matrix[idx] = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
                stats_per_class.append({real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories})
                continue 
            values = line.split(",")
            predicted_cat = values[0]
            real_cat = values[-1][:-1] # remove \n
            tn_cats = []
            if predicted_cat == real_cat: 
                stats_per_class[-1][predicted_cat]["tp"] += 1 #it really is a hit
                tn_cats = [cat for cat in categories if cat != predicted_cat]
            else:
                stats_per_class[-1][predicted_cat]["fp"] += 1 #for the other cat shouldn't be a hit but it is
                stats_per_class[-1][real_cat]["fn"] += 1 #for the cat its a hit but it shouldn't be
                tn_cats = [cat for cat in categories if cat != predicted_cat and cat != real_cat]

            for cat in tn_cats:
                stats_per_class[-1][cat]["tn"] += 1 #for all the other cat, it wont be a hit for sure

            confusion_matrix[real_cat][predicted_cat] += 1
        
        metrics_per_iter = []
        for stats in stats_per_class:
            metrics_per_iter.append({cat: {"accuracy": 0, "precision":0, "fpr": 0, "tpr": 0, "f1": 0} for cat in categories})
            for cat in categories: 
                metrics_per_iter[-1][cat]["accuracy"] = (stats[cat]["tp"] + stats[cat]["tn"]) / (stats[cat]["tp"] + stats[cat]["tn"]+ stats[cat]["fp"] + stats[cat]["fn"])
                metrics_per_iter[-1][cat]["precision"] = stats[cat]["tp"] / (stats[cat]["tp"] + stats[cat]["fp"])
                metrics_per_iter[-1][cat]["fpr"] = (stats[cat]["fp"]) / (stats[cat]["fp"] + stats[cat]["tn"])
                metrics_per_iter[-1][cat]["tpr"] = (stats[cat]["tp"]) / (stats[cat]["tp"] + stats[cat]["fn"])
                metrics_per_iter[-1][cat]["f1"] = 2 * (metrics_per_iter[-1][cat]["precision"] * metrics_per_iter[-1][cat]["tpr"]) / (metrics_per_iter[-1][cat]["precision"] + metrics_per_iter[-1][cat]["tpr"])
            
        
        metrics_per_class = {"mean": {cat: {"accuracy": 0, "precision":0, "fpr": 0, "tpr": 0, "f1": 0} for cat in categories}, 
                                   "std": {cat: {"accuracy": 0, "precision":0, "fpr": 0, "tpr": 0, "f1": 0} for cat in categories}}
        
        metrics_labels = ["accuracy", "precision", "fpr", "tpr", "f1"]
        for cat in categories:
            for label in metrics_labels:
                for cat in categories:
                    metrics_per_class["mean"][cat][label] = np.mean([metrics_per_iter[i][cat][label] for i in range(len(metrics_per_iter))])
                    metrics_per_class["std"][cat][label] = np.std([metrics_per_iter[i][cat][label] for i in range(len(metrics_per_iter))])

        
        
        pd.DataFrame(metrics_per_class["mean"]).to_csv("out/mean_metrics.csv")
        pd.DataFrame(metrics_per_class["std"]).to_csv("out/std_metrics.csv")
        
    c_file.close()
    return confusion_matrix
        
# TODO: separar por iteracion y capaz mostrar un grafico con el desvio
def calculate_roc(threshold, categories):
    stats_per_class = {}
    with open ("post_processing/classification.csv", "r") as c_file:
        c_file.readline()
        # categories = sorted(c_file.readline().split(",")[1:-1])
        stats_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}
        for line in c_file:
            if "prediction" in line:
                continue #we consider all partitions 
            values = line.split(",")
            predicted_cat = values[0]
            real_cat = values[len(values) - 1][:-1]

            for idx, cat in enumerate(categories):
                if float(values[idx + 1]) > threshold:
                    if real_cat == cat:
                        stats_per_class[cat]["tp"] += 1
                    else:
                        stats_per_class[cat]["fp"] += 1
                else:
                    if real_cat != cat:
                        stats_per_class[cat]["tn"] += 1
                    else:
                        stats_per_class[cat]["fn"] += 1

    c_file.close()    
    return stats_per_class

def graph_roc(rocs, categories, xticks):
    plt.clf()
    for i, cat in enumerate(categories):
        x = [rocx["fpr"] for rocx in rocs[i]]
        y = [rocy["tpr"] for rocy in rocs[i]]
        plt.plot(x,y, 'o-', label=cat, markersize=4)
        plt.xticks(xticks)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
    plt.plot([0,1], [0,1], '--', color='grey')
    plt.legend()
    plt.title("Curva ROC")
    plt.tight_layout()
    plt.savefig("out/roc.png")

def get_full_metric_csv():

    mean_df = pd.read_csv("out/mean_metrics.csv")
    std_df = pd.read_csv("out/std_metrics.csv")
    mean_df = mean_df.drop(columns=["Unnamed: 0"])
    std_df = std_df.drop(columns=["Unnamed: 0"])
    full_df = pd.DataFrame(columns=mean_df.columns, index=mean_df.index)

    for cat in mean_df.index:
        full_df.loc[cat] = [f"{round(mean_df.loc[cat][col], 3)} +- 0.00" + f"{std_df.loc[cat][col]:.3}"[-1] for col in mean_df.columns]
    
    full_df.insert(0, "metric", ["accuracy", "precision", "fpr", "tpr", "f1"])
    full_df.to_csv("out/full_metrics.csv")
    

def main():
    categories = get_categories()
    confusion_matrix_dict = calculate_metrics(categories)
    
    get_full_metric_csv()

    get_heatmap(pd.DataFrame(confusion_matrix_dict))

    rocs = []
    roc_for_graph = []

    plt.clf()
    step = 0.1 # TODO: maybe read it from config
    thresholds = np.arange(0, 1 + step, step)
    for idx, cat in enumerate(categories):
        rocs.clear()
        for threshold in thresholds:
            roc = calculate_roc(threshold, categories)
            fpr = roc[cat]["fp"] / (roc[cat]["fp"] + roc[cat]["tn"]) 
            tpr = roc[cat]["tp"] / (roc[cat]["tp"] + roc[cat]["fn"])
            rocs.append({"fpr": fpr, "tpr": tpr})
        roc_for_graph.append(rocs.copy())

    graph_roc(roc_for_graph,categories, thresholds)

if __name__ == "__main__":
    main()
