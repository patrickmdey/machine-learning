import pandas as pd
from utils import *

def main():
    df = pd.read_csv("./Datos1.csv", usecols=['Grasas_sat','Alcohol','Calorías','Sexo'])
    method = "mean"


    df["Calorías"] = df["Calorías"].str.replace(",", "").astype(int)

    df = replace_missing_values(df, "Alcohol", 999.99, method)
    df = replace_missing_values(df, "Grasas_sat", 999.99, method)

    # boxplot de todas las variables
    for column in ['Grasas_sat','Alcohol','Calorías']: 
        print()
        print(column)
        boxplot_column(df, column, method)
        boxplot_grouped_column(df, column, "Sexo")
        kurtosis_column(df, column)
        skewness_column(df, column)

    bar_column(df, "Sexo")

    # analisis de covarianza
    input_df = df.loc[:,df.columns != "Sexo"]

    # normalized_df=(input_df-input_df.mean())/input_df.std()

    input_df["Sexo"] = df["Sexo"]
    cov_analysis(input_df)

    # categorizando calorias
    categorized_df = categorize_calories_col(df)
    scatter_category(categorized_df, "Alcohol", "Categorías")
    boxplot_grouped_column(categorized_df, 'Alcohol', "Categorías")
    


if __name__ == "__main__":
    main()