from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys, os

def get_precision_graph(df, partition_amount):
    plt.clf()
    plt.plot(df["estimators"], df["mean_test_prec"], label="test")
    plt.plot(df["estimators"], df["mean_train_prec"], label="train")
    plt.xlabel("Cantidad de estimadores")
    plt.ylabel("Precisión")
    plt.legend(loc="upper right")
    plt.title("Precisión vs Cantidad de estimadores")

    out_path = "simulation_out/" + str(partition_amount) + "/" + \
        "precision_vs_estimator_amount.png"
    plt.savefig(out_path)

if __name__ == "__main__":

    partition_amount = int(sys.argv[1] if len(sys.argv) > 1 else None)

    path = "simulation_out/"+str(partition_amount) + "/"
    path += "precisions.csv"

    if not os.path.exists(path) or partition_amount is None:
        print("Error al tratar de leer", path)
        exit()

    df = pd.read_csv(path)

    get_precision_graph(df, partition_amount)
