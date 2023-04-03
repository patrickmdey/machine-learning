import pandas as pd
from node import MyNode

def main():
    df = pd.read_csv("./binary.csv")

    node = MyNode(df, "rank")
    

    

if __name__ == "__main__":
    main()