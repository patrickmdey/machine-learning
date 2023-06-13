import pandas as pd
from utils import boxplot_column

def dataset_analysis(df, useful_cols):
    print("Amount of duplicated rows:", df.duplicated().sum())
    print("Amount of null values:", df.isnull().sum().sum())
    print("Amount of rows:", df.shape[0])
    print("Amount of columns:", df.shape[1])
    # print("Duplicated rows:")
    # print(df[df.duplicated()]["imdb_id"])

    for col in useful_cols:
        print("Column:", col)
        boxplot_column(df, col)

    # remove rows with null values
    df = df.dropna()
    df = df.drop_duplicates(subset="imdb_id", keep=False)

    years = df["release_date"].str.split("-", n=2, expand=True)[0]
    df["release_date"] = years

    # trimed_df = df[useful_cols]

    # df.drop(["genres", "imdb_id", "original_title", "overview", "production_companies"], axis=1)
    # print(df["original_title"].head())

    # get standard deviation per column
    # TODO: ver std
    # print(df.std)

if __name__ == '__main__':
    analysis_cols = ["budget", "genres", "imdb_id", "popularity", "production_companies", "production_countries",
                     "revenue", "runtime", "spoken_languages", "vote_average", "vote_count"]
    
    dataset_analysis(pd.read_csv("movie_data.csv", sep=';'), analysis_cols)