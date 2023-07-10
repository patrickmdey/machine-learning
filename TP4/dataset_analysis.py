import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from utils import *


def dataset_analysis(df):

    useful_cols = ["budget", "popularity", "production_companies", "production_countries",
                   "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]

    print("Amount of duplicated rows:", df.duplicated(["imdb_id"]).sum())
    print("Amount of null values:", df.isnull().sum().sum())
    print("Amount of rows:", df.shape[0])
    print("Amount of columns:", df.shape[1])
    # print("Duplicated rows:")
    # print(df[df.duplicated()]["imdb_id"])

    # remove rows with null values
    df = df.dropna()
    df = df.drop_duplicates(subset="imdb_id", keep=False)

    for col in useful_cols:
        boxplot_grouped_column(df, col, "genres")

    df.drop(["genres", "imdb_id", "original_title", "overview"], axis=1)

    for col in useful_cols:
        boxplot_column(df, col)

    trimed_df = df[useful_cols]

    print(trimed_df.std().sort_values(ascending=False))

if __name__ == '__main__':

    df = pd.read_csv('movie_data.csv', sep=';')
    df = df.loc[(df["genres"] == "Drama") | (
        df["genres"] == "Comedy") | (df["genres"] == "Action")]
    
    print("Drama",len(df.loc[df["genres"] == "Drama"])*100/len(df), "%")
    print("Comedia",len(df.loc[df["genres"] == "Comedy"])*100/len(df), "%")
    print("Acción", len(df.loc[df["genres"] == "Action"])*100/len(df), "%")

    dataset_analysis(df)
