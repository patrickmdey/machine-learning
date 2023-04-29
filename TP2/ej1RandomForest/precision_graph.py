import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_precision_graph(df, method):

    
    df["nodes" == -1] = 1000

    plt.plot(df["nodes"],df["mean_test_precision"], label="test")
    plt.plot(df["nodes"],df["mean_train_precision"], label="train")
    plt.xlabel("Cantidad de nodos")
    plt.ylabel("Precisión")
    plt.legend(loc="upper right")
    plt.title("Precisión vs Cantidad de nodos")
    plt.savefig("out/"+method+"/precision_vs_nodes.png")
    plt.clf()

def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "id3"
    if method not in ["id3", "random_forest"]:
        print("Invalid method")
        return
    
    if not os.path.exists("out/"+method+"/precision_vs_nodes.csv"):
        print("No existe el archivo de precisiones")
        return
    
    df = pd.read_csv("out/"+method+"/precision_vs_nodes.csv")

    get_precision_graph(df, method)

if __name__ == "__main__":
    main()




