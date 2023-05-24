from utils import *
import sys
from main import *

x_min, x_max, y_min, y_max = 0, 5, 0, 5
file_name_1 = "input/no_noise/TP3-1"
file_name_2 = "input/with_noise/TP3-2"

# # file_name = sys.argv[1] if len(sys.argv) > 1 else file_name

point_amount = 30
# error_rate = 0

# if len(sys.argv) > 3:
#     print("Generating", str(sys.argv[2]), "points and", str(
#         sys.argv[3]), "error rate")
#     point_amount = int(sys.argv[2])
#     error_rate = float(sys.argv[3])

# data, line = random_points_within_range(
#     x_min, x_max, y_min, y_max, point_amount, error_rate)
# df = pd.DataFrame(data, columns=["x", "y", "class"])
# file_name_1 += "-"+str(point_amount)
# df.to_csv(file_name_1+".csv", index=False)
# line_df = pd.DataFrame([line], columns=["m", "b"])
# line_df.to_csv(file_name_1+"-line.csv", index=False)

df = pd.read_csv("input/no_noise/TP3-1-30.csv")
line_df = pd.read_csv("input/no_noise/TP3-1-30-line.csv")

X, y, weights = run_perceptron(df, 10000, 0.01)


line_y, upper_points, lower_points, optimal_point = optimize_perceptron(
    X, y, weights, 3, 2)

noisy_df = df.copy()

print(upper_points)

noisy_df["class"] = np.where(df["x"].isin(upper_points[:,0]) & df["y"].isin(upper_points[:,1]), df["class"] * -1, df["class"])
noisy_df["class"] = np.where(df["x"].isin(lower_points[:,0]) & df["y"].isin(lower_points[:,1]), df["class"] * -1, df["class"])

file_name_2 += "-"+str(point_amount)
line_df.to_csv(file_name_2+"-line.csv", index=False)
noisy_df.to_csv(file_name_2+".csv", index=False)
