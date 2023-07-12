import pandas as pd
import numpy as np

if __name__ == "__main__":

    df = pd.read_csv("simulation_out/best_config.csv")

    max_config_idx = df["mean_test_prec"].idxmax()

    best_config_df = pd.DataFrame(df.iloc[max_config_idx]).transpose()
    best_config_df = best_config_df.drop("Unnamed: 0", axis=1)
    print(best_config_df)
