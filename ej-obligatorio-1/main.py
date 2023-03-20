import pandas as pd
from utils import *


def main():
    df = pd.read_csv("./Advertising.csv", usecols=['TV','Radio','Newspaper','Sales'])
    
    # corr_analysis(df)
    # scatter_category(df, 'TV', 'Sales', 'Sales')
    # col_len = len(df['TV'])
    # linear_model(df, df['TV'].values.reshape(col_len, 1), df['Sales'].values.reshape(col_len, 1))
    # simple_linear_models(df)

    # multiple_linear_models(df)
    test(df)


def simple_linear_models(df):
    col_len = len(df['TV'])

    for col in ['TV','Radio','Newspaper']:
        linear_model(df[col].values.reshape(col_len, 1), df['Sales'].values.reshape(col_len, 1), True, col, 'Sales')

def multiple_linear_models(df):
    trimmed_df = df[["TV", "Radio", "Newspaper"]]
    linear_model(trimmed_df, df['Sales'], True)


def test(df):
    test_linear_model(df, 0.2)


    
    
    

if __name__ == "__main__":
    main()
