from utils import *
import sys

x_min, x_max, y_min, y_max = 0, 5, 0, 5
file_name = "input/no_noise/TP3-1"

file_name = sys.argv[1] if len(sys.argv) > 1 else file_name

point_amount = 30
error_rate = 0

if len(sys.argv) > 3:
    print("Generating", str(sys.argv[2]), "points and ", str(sys.argv[3]), "error rate")
    point_amount = int(sys.argv[2])
    error_rate = float(sys.argv[3])

data, line = random_points_within_range(x_min, x_max, y_min, y_max, point_amount, error_rate)
df = pd.DataFrame(data, columns=["x", "y", "class"])
file_name+="-"+str(point_amount)
df.to_csv(file_name+".csv", index=False)
line_df = pd.DataFrame([line], columns=["m", "b"])
line_df.to_csv(file_name+"-line.csv", index=False)