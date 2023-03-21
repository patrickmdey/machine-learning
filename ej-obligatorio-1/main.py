import pandas as pd
from utils import *


def main():
    df = pd.read_csv("./Advertising.csv", usecols=['TV','Radio','Newspaper','Sales'])
    
    print("Running correlation analysis...\n")
    corr_analysis(df)
    print("Running simple linear regression...")
    # col_len = len(df['TV'])
    # linear_model(df, df['TV'].values.reshape(col_len, 1), df['Sales'].values.reshape(col_len, 1))
    simple_linear_models(df)

    print("Running multiple linear regression...")
    multiple_linear_models(df)

    print("Running simple linear regression test...")
    simple_test(df)

    print("Running multiple linear regression test...")
    multiple_test(df)


def simple_linear_models(df):
    col_len = len(df['TV'])
    for col in ['TV','Radio','Newspaper']:
        linear_model(df[col].values.reshape(col_len, 1), df['Sales'].values.reshape(col_len, 1), False, col, 'Sales')

def multiple_linear_models(df):
    trimmed_df = df[["TV", "Radio", "Newspaper"]]
    linear_model(trimmed_df, df['Sales'], True)


def simple_test(df):
    for col in ['TV','Radio','Newspaper']:
        test_simple_linear_model(df, col, 0.2)

def multiple_test(df):
    test_multple_linear_model(df, 0.2)


if __name__ == "__main__":
    main()
