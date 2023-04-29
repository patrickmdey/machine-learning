from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def boxplot_column(df, column_name, method=""):
    plt.clf()
    boxplot = df.boxplot(column=column_name, grid=False)
    boxplot.set_ylabel(column_name)
    boxplot.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    str = column_name + "_with_" + method if method != "" else column_name
    boxplot.figure.savefig("./out/" + str + "_boxplot.png")

def bar_column(df, column_name):
    plt.clf()
    y = df[column_name].value_counts()
    plt.bar(y.index, y)
    plt.xlabel(column_name)
    plt.ylabel("Cantidad")
    plt.savefig("./out/"+column_name + "_bar.png")

def categorize_columns(df, columns):
    for column_name in columns:
        quartiles = df[column_name].quantile([0.25, 0.5, 0.75, 1])
        # print(column_name, quartiles)

        df[column_name] = pd.qcut(df[column_name], 4, labels=[0, 1, 2, 3])
    return df


if __name__ == "__main__":
    df = pd.read_csv("./german_credit.csv")
    
    columns = ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
    for column_name in columns:
        boxplot_column(df, column_name)
        bar_column(df, column_name)
    
    categorize_columns(df, columns)