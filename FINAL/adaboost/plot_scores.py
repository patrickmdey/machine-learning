from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys, os

def get_precision_graph(df, partition_amount, learning_rate=None):
    plt.clf()
    plt.plot(df["estimators"], df["mean_test_prec"], label="test")
    plt.plot(df["estimators"], df["mean_train_prec"], label="train")
    plt.xlabel("Cantidad de estimadores")
    plt.ylabel("Precisión")
    plt.legend(loc="upper right")
    plt.title("Precisión vs Cantidad de estimadores - " + str(partition_amount) + " particiones y " + str(learning_rate) + " de learning rate")

    out_path = "simulation_out/" + str(partition_amount) + "/" + \
        "precision_vs_estimator_amount_" + str(learning_rate) + ".png"
    plt.savefig(out_path)

def get_optimal_confusion_matrix():
    return

if __name__ == "__main__":

    partition_amount = int(sys.argv[1] if len(sys.argv) > 1 else None)
    learning_rate = sys.argv[2] if len(sys.argv) > 2 else None

    path = "simulation_out/"+str(partition_amount) + "/"
    path += "precisions_" + str(learning_rate) + ".csv"

    if not os.path.exists(path) or partition_amount is None:
        print("Error al tratar de leer", path)
        exit()

    df = pd.read_csv(path)

    get_precision_graph(df, partition_amount, learning_rate)


