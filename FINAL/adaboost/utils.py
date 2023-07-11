from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def partition_dataset(df, partition_percentage):
    # shuffle dataframe rows
    df = df.sample(frac=1).reset_index(drop=True)

    partition_size = int(np.floor(len(df) * partition_percentage))
    partitions = []

    bottom = 0
    up = partition_size
    while bottom < len(df):

        partitions.append(df[bottom:up].copy())
        bottom += partition_size
        up += partition_size
        if up > len(df):
            up = len(df)

    if (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2],
                                   partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions


def categorize_columns(df, columns):

    quantile_dict = {"column": [], "q1": [], "q2": [], "q3": [], "q4": []}

    for column_name in columns:
        quartiles = df[column_name].quantile([0.25, 0.5, 0.75, 1])

        if column_name == "Oldpeak":
            df[column_name] = pd.qcut(df[column_name], 3, labels=[0, 1, 2])
        else:
            df[column_name] = pd.qcut(df[column_name], 4, labels=[0, 1, 2, 3])

        if not os.path.exists("./dataset_out/quantiles.csv"):
            quantile_dict["column"].append(column_name)
            quantile_dict["q1"].append(quartiles[0.25])
            quantile_dict["q2"].append(quartiles[0.5])
            quantile_dict["q3"].append(quartiles[0.75])
            quantile_dict["q4"].append(quartiles[1])

    pd.DataFrame(quantile_dict).to_csv("./dataset_out/quantiles.csv",
                                       index=False) if len(quantile_dict["column"]) > 0 else None

    return df


def prepare_dataset(df):
    # Adaboost doesnt support categorical data so it needs to be converted to numerical
    mappings = {
        "Fbs": {"<=120": 0, ">120": 1},
        "Sex": {"F": 0, "M": 1},
        "ChestPain": {"typical": 0, "asymptomatic": 1, "nonanginal": 2, "nontypical": 3},
        "RestECG": {"normal": 0, "abnormal": 1},
        "ExAng": {"No": 0, "Yes": 1},
        "Slope": {"down": 0, "level": 1, "up": 2},
        "Thal": {"normal": 0, "fixed": 1, "reversable": 2},
        "HDisease": {"No": 0, "Yes": 1}
    }

    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)

    # columns = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]
    # df = categorize_columns(df, columns)
    return df


def replace_missing_values(df, column, value, method="mean"):
    mean_df = df.copy()
    median_df = df.copy()
    # drops the rows that have Alcohol >= value
    df.drop(df[df[column] >= value].index, inplace=True)
    df_mean = df[column].mean()
    df_median = df[column].median()

    # print()
    if (method == "mean"):
        # print("Replacing for mean...")
        mean_df[column].replace(value, df_mean, inplace=True)
        print("New " + column + " mean:", mean_df[column].mean())
        print("New " + column + " median:", mean_df[column].median())
        return mean_df
    elif (method == "median"):
        # print("Replacing for median...")
        median_df[column].replace(value, df_median, inplace=True)
        print("New " + column + " mean:", median_df[column].mean())
        print("New " + column + " median:", median_df[column].median())
        return median_df


def cov_analysis(df):
    print(df.groupby("Sexo").cov())


def bar_column(df, column_name):
    plt.clf()
    y = df[column_name].value_counts()
    plt.bar(y.index, y)
    plt.xlabel(column_name)
    plt.ylabel("Cantidad")
    plt.savefig("./dataset_out/"+column_name + "_bar.png")


def kurtosis_column(df, column):
    print("Kurtosis:", df[column].kurtosis())


def skewness_column(df, column):
    print("Skewness:", df[column].skew())


def plot_pie(df, col):
    plt.clf()
    plt.pie(df[col].value_counts(),
            labels=df[col].value_counts().index.tolist(), autopct='%1.1f%%')
    plt.title(col)
    plt.tight_layout()
    plt.savefig("dataset_out/"+col+"_pie.png")


def boxplot_column(df, column_name, method=""):
    plt.clf()
    boxplot = df.boxplot(column=column_name, grid=False)
    boxplot.set_ylabel("Cantidad")
    boxplot.yaxis.grid(True, linestyle='-', which='major',
                       color='lightgrey', alpha=0.5)
    file_name = column_name + "_with_" + method if method != "" else column_name
    boxplot.figure.tight_layout()
    boxplot.figure.savefig("./dataset_out/" + file_name + "_boxplot.png")


def boxplot_grouped_column(df, column_name, group_by):
    plt.clf()
    boxplot = sns.boxplot(x=group_by, y=column_name, data=df)
    boxplot.set_ylabel(column_name)

    os.mkdir("./dataset_out/by_" + group_by +
             "/") if not os.path.exists("./dataset_out/by_" + group_by+"/") else None

    # boxplot.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    boxplot.figure.savefig("./dataset_out/by_" + group_by +
                           "/"+column_name + "_boxplot.png")


def scatter_category(df, x, y):
    plt.clf()

    scatter = df.plot.scatter(x, y)

    scatter.set_xlabel(x)
    scatter.set_ylabel("Calorías")
    scatter.figure.savefig("./dataset_out/"+x + "_vs_" + y + "_scatter.png")
