import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_precision_graph(df, method, partition_amount, tree_amount=None):
    plt.clf()

    plt.plot(df["nodes"],df["mean_test_precision"], label="test")
    plt.plot(df["nodes"],df["mean_train_precision"], label="train")
    plt.xlabel("Cantidad de nodos")
    plt.ylabel("Precisión")
    plt.legend(loc="upper right")
    plt.title("Precisión vs Cantidad de nodos")

    out_path = "out/"+method+ "/" + str(partition_amount) + "/" + ((str(tree_amount) + "_trees") if tree_amount is not None else "") + "/precision_vs_nodes.png"
    print(out_path)
    plt.savefig(out_path)

def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "id3"
    if method not in ["id3", "random_forest"]:
        print("Invalid method")
        return
    
    partition_amount = int(sys.argv[2] if len(sys.argv) > 2 else None)
    tree_amount = sys.argv[3] if len(sys.argv) > 3 else None

    
    path = "out/"+method+"/"+str(partition_amount) + "/" 
    path += (str(tree_amount) + "_trees/") if tree_amount is not None else ""
    path += "precision_vs_nodes.csv"
    if not os.path.exists(path) or partition_amount is None:
        print("Error al tratar de leer el archivo")
        return
    
    df = pd.read_csv(path)

    get_precision_graph(df.tail(-1), method, partition_amount, tree_amount)

if __name__ == "__main__":
    main()




