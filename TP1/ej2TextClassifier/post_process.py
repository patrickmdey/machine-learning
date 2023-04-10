import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_categories():
    with open ("post_processing/classification.csv", "r") as c_file:
        # prediction,Nacional,Salud,Economia,Destacadas,Ciencia y Tecnologia,Deportes,Internacional,Entretenimiento,real
        return c_file.readline().split(",")[1:-1]        

def get_heatmap(df):
    plt.clf()
    sns.heatmap(df, annot=True, fmt='.0f')
    plt.tight_layout()
    plt.savefig("post_processing/heatmap.png")

def get_graphs():
    df = pd.read_csv("post_processing/confusion_matrix.csv")
    get_heatmap(df)


# prediction,Nacional,Salud,Economia,Destacadas,Ciencia y Tecnologia,Deportes,Internacional,Entretenimiento,real
def calculate_metrics():
    with open ("post_processing/classification.csv", "r") as c_file:
        categories = c_file.readline().split(",")[1:-1]
        multiple_metrics_per_class = []
        metrics_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}
        multiple_confusion_matrix=[]
        confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
        idx = 0
        for line in c_file:
            if "prediction" in line: #new train-test partition
                
                confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
                multiple_confusion_matrix.append(confusion_matrix)
                metrics_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}
                multiple_metrics_per_class.append(metrics_per_class)
                idx += 1
                continue 
            values = line.split(",")
            predicted_cat = values[0]
            real_cat = values[-1][:-1] # remove \n
            # for i in range(1, len(values) - 1):
            tn_cats = []
            if predicted_cat == real_cat: 
                metrics_per_class[predicted_cat]["tp"] += 1 #it really is a hit
                tn_cats = [cat for cat in categories if cat != predicted_cat]
            else:
                metrics_per_class[predicted_cat]["fp"] += 1 #for the other cat shouldn't be a hit but it is
                metrics_per_class[real_cat]["fn"] += 1 #for the cat its a hit but it shouldn't be
                tn_cats = [cat for cat in categories if cat != predicted_cat and cat != real_cat]

            for cat in tn_cats:
                metrics_per_class[cat]["tn"] += 1 #for all the other cat, it wont be a hit for sure

            confusion_matrix[idx][real_cat][predicted_cat] += 1

    c_file.close()    
        
def calculate_roc(threshold):
    metrics_per_class = {}
    with open ("post_processing/classification.csv", "r") as c_file:
        categories = sorted(c_file.readline().split(",")[1:-1])
        metrics_per_class = {real_cat: {"tp": 0, "tn":0, "fp": 0, "fn": 0} for real_cat in categories}
        # confusion_matrix = {real_cat: {pred_cat: 0 for pred_cat in categories} for real_cat in categories}
        for line in c_file:
            if "prediction" in line:
                continue 
            values = line.split(",")
            predicted_cat = values[0]
            real_cat = values[len(values) - 1][:-1]

            # print(categories)

            for idx, cat in enumerate(categories):
                if float(values[idx + 1]) > threshold:
                    # print("PROB " + str(float(values[idx + 1])) + " THRE " + str(threshold) + " CAT " + cat + " REAL " + real_cat)
                    if real_cat == cat:
                        # print("llegue")
                        metrics_per_class[cat]["tp"] += 1
                    else:
                        metrics_per_class[cat]["fp"] += 1
                else:
                    if real_cat != cat:
                        metrics_per_class[cat]["tn"] += 1
                    else:
                        metrics_per_class[cat]["fn"] += 1

    c_file.close()    
    return metrics_per_class

def graph_roc(rocs, cat, xticks, color_idx):

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey", "black"]

    x = [rocx["fpr"] for rocx in rocs]
    y = [rocy["tpr"] for rocy in rocs]
    
    # for value in rocs:
    plt.plot(x,y, 'o-', label=cat, c=colors[color_idx])
    plt.xticks(xticks)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

def graph_roc2(rocs, categories, xticks):
    plt.clf()
    for i, cat in enumerate(categories):
        
        x = [rocx["fpr"] for rocx in rocs[i]]
        y = [rocy["tpr"] for rocy in rocs[i]]

        # print(y)
        # for value in rocs:
        plt.plot(x,y, 'o-', label=cat)
        plt.xticks(xticks)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
    plt.legend()
    plt.savefig("out/roc.png")

def main():
    categories = get_categories()
    calculate_metrics()
    rocs = []
    roc_for_graph = []

    plt.clf()
    step = 0.1 # TODO: maybe read it from config
    thresholds = np.arange(0, 1 + step, step)
    for idx, cat in enumerate(categories):
        rocs.clear()
        for threshold in thresholds:
            roc = calculate_roc(threshold)
            fpr = roc[cat]["fp"] / (roc[cat]["fp"] + roc[cat]["tn"]) 
            tpr = roc[cat]["tp"] / (roc[cat]["tp"] + roc[cat]["fn"])
            rocs.append({"fpr": fpr, "tpr": tpr})
        roc_for_graph.append(rocs.copy())

    graph_roc2(roc_for_graph,categories, thresholds)

        # graph_roc(rocs, cat, thresholds, idx)
    
    plt.savefig("out/roc.png")
if __name__ == "__main__":
    main()
