import sys
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_precision_graph(df, partition_amount, learning_rate=None):
    plt.clf()

    plt.plot(df["estimators"], df["mean_test_precision"], label="test")
    plt.plot(df["estimators"], df["mean_train_precision"], label="train")
    plt.xlabel("Cantidad de estimadores")
    plt.ylabel("Precisión")
    plt.legend(loc="upper right")
    plt.title("Precisión vs Cantidad de estimadores")

    out_path = "out/" + str(partition_amount) + "/" + \
        "precision_vs_estimator_amount_" + str(learning_rate) + ".png"
    print(out_path)
    plt.savefig(out_path)


if __name__ == "__main__":

    partition_amount = int(sys.argv[1] if len(sys.argv) > 1 else None)
    learning_rate = sys.argv[2] if len(sys.argv) > 2 else None

    path = "out/"+str(partition_amount) + "/"
    path += "precision" + "_learning_rate_" + str(learning_rate) + ".csv"

    if not os.path.exists(path) or partition_amount is None:
        print("Error al tratar de leer", path)
        exit()

    df = pd.read_csv(path)

    get_precision_graph(df, partition_amount, learning_rate)
