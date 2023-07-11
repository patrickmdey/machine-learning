import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from utils import *

def dataset_analysis(df):

    useful_cols = ["Age",
                   "RestBP",
                   "Chol",
                   "MaxHR",
                   "Oldpeak"]

    print("Amount of rows:", df.shape[0])
    # print("Amount of columns:", df.shape[1])

    # remove rows with null values
    # df = df.dropna()

    for col in useful_cols:
        boxplot_grouped_column(df, col, "Sex")

    for col in useful_cols:
        boxplot_column(df, col)
    
    plot_pie(df, "HDisease")
    plot_pie(df, "Sex")
    plot_pie(df, "HDisease")

    


if __name__ == '__main__':

    df = pd.read_csv('../heart.csv', sep=',')

    # df = prepare_dataset(df)

    dataset_analysis(df)
