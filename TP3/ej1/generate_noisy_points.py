from utils import *
import sys
from main import *

file_name_1 = "input/no_noise/TP3-1"
file_name_2 = "input/with_noise/TP3-2"


for points in [60,120]:
    pre_path = "input/no_noise/TP3-1-" + str(points)

    df = pd.read_csv(pre_path+ ".csv")
    line_df = pd.read_csv(pre_path+"-line.csv")

    X, y, weights = run_perceptron(df, 10000, 0.01)


    line_y, upper_points, lower_points, optimal_point = optimize_perceptron(
        X, y, weights,3, 10)

    noisy_df = df.copy()

    random_values = np.random.random(len(df))
    should_multiply = random_values < 0.5

    noisy_df["class"] = np.where(df["x"].isin(upper_points[:,0]) & df["y"].isin(upper_points[:,1]), df["class"], df["class"])
    noisy_df["class"] = np.where(df["x"].isin(lower_points[:,0]) & df["y"].isin(lower_points[:,1]), np.where(should_multiply, df["class"] * -1, df["class"]), df["class"])

    file_name_2 += "-"+str(points)
    line_df.to_csv(file_name_2+"-line.csv", index=False)
    noisy_df.to_csv(file_name_2+".csv", index=False)
